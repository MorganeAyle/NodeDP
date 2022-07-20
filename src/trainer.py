import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from src import autograd_hacks

from src.models import create_model
from src.constants import DP_METHODS, NON_DP_METHODS

import warnings


class Trainer:
    def __init__(self, training_args, model_args, feats, class_arr, use_cuda, minibatch, out, sampler_args):
        self.feats = torch.from_numpy(feats.astype(np.float32))
        self.labels = torch.from_numpy(class_arr.astype(np.float32))
        self.use_cuda = use_cuda
        if self.use_cuda:
            self.feats = self.feats.cuda()
            self.labels = self.labels.cuda()
        self.sampler_args = sampler_args

        # minibatch
        self.minibatch = minibatch

        # Loss
        if training_args['loss'] == 'sigmoid':
            self.sigmoid_loss = True
        else:
            assert training_args['loss'] == 'softmax'
            self.sigmoid_loss = False

        # Create model
        in_channels = self.feats.shape[1]
        out_channels = self.labels.shape[1]
        self.model = create_model(in_channels, out_channels, model_args)
        autograd_hacks.add_hooks(self.model)  # for the fast DP method
        self.num_param = len([p for p in self.model.parameters()])

        if self.use_cuda:
            self.model.to('cuda')

        # Optimizer
        if training_args['optim'].lower() == 'adam':
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=training_args['lr'])
        elif training_args['optim'].lower() == 'sgd':
            self.optimizer = torch.optim.SGD(self.model.parameters(), lr=training_args['lr'])
        else:
            raise NotImplementedError

        # Get gradient norms
        if sampler_args['method'] in NON_DP_METHODS:
            self.clip_norm = training_args['clip_norm']
        elif sampler_args['method'] in DP_METHODS:
            clip_norm = training_args['clip_norm']
            init_grad_norms = self.estimate_init_grad_norms(*self.minibatch.sample_one_batch(out, mode='train'))
            if np.sqrt(sum(init_grad_norms ** 2)) < clip_norm:
                warnings.warn("Specified clip norm is larger than initial model's clip norm. Use smaller clip_norm"
                              "for better privacy guarantees.")
            grad_weights = torch.nn.functional.softmax(init_grad_norms, dim=0)
            self.per_param_clip_norm = np.sqrt(grad_weights * (clip_norm ** 2))
        else:
            raise NotImplementedError

    def predict(self, preds):
        return nn.Sigmoid()(preds) if self.sigmoid_loss else F.softmax(preds, dim=1)

    def _loss(self, preds, labels):
        """
        The predictor performs sigmoid (for multi-class) or softmax (for single-class)
        """
        if self.sigmoid_loss:
            return nn.BCEWithLogitsLoss(reduction='mean')(preds, labels)
        else:
            return nn.CrossEntropyLoss(reduction='mean')(preds, labels)

    def estimate_init_grad_norms(self, nodes, adj, roots=None):
        grad_norms = []
        i = 0
        count = 0
        while count < 10:
            i += 1
            if nodes[0][i] in roots[0]:
                count += 1
                self.model.zero_grad()
                preds = self.model(self.feats[nodes[0]], adj[0])
                loss = self._loss(preds[i].unsqueeze(0), self.labels[nodes[0]][i].unsqueeze(0))
                grads = torch.autograd.grad(loss, list(self.model.parameters()))
                grad_norms.append([grad.norm() for grad in grads])
        return torch.tensor(grad_norms).mean(0)

    def train_step(self, nodes, adj, roots=None):
        self.model.train()
        self.optimizer.zero_grad()
        preds = self.model(self.feats[nodes], adj)

        loss = self._loss(preds, self.labels[nodes])
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_norm)
        self.optimizer.step()
        return loss, self.predict(preds), self.labels[nodes]

    def dp_train_step_fast(self, all_nodes, all_adj, all_roots=None, sigma=1.):
        all_grads = []
        num_nodes = sum([len(roots) for roots in all_roots])

        if type(all_adj) != list:
            all_nodes = [all_nodes]
            all_adj = [all_adj]
            all_roots = [all_roots]

        for nodes, adj, roots in zip(all_nodes, all_adj, all_roots):
            self.model.train()
            self.optimizer.zero_grad()
            autograd_hacks.clear_backprops(self.model)
            for param in self.model.parameters():
                if hasattr(param, 'grad1'):
                    del param.grad1

            # compute loss only on training nodes
            idx = [i for i, v in enumerate(nodes) if v in roots]
            preds = self.model(self.feats[nodes], adj)
            self._loss(preds[idx], self.labels[nodes][idx]).backward(retain_graph=True)
            autograd_hacks.compute_grad1(self.model)

            # compute new noisy gradients
            for i, param in enumerate(self.model.parameters()):
                grad_norm = param.grad1.norm(dim=tuple(np.arange(len(param.grad1.shape))[1:]))[idx].flatten()
                denom = torch.maximum(torch.tensor(1, device=param.grad1.device),
                                      grad_norm / self.per_param_clip_norm[i])
                for _ in range(len(param.grad1.shape) - 1):
                    denom = denom.unsqueeze(-1)

                new_grad = (param.grad1[idx] / denom).sum(0)
                if len(all_grads) < self.num_param:
                    all_grads.append(new_grad)
                else:
                    all_grads[i] += new_grad

        for i, param in enumerate(self.model.parameters()):
            noise = torch.empty(param.shape, device=param.device).normal_(mean=0.0, std=sigma)
            param.grad = ((all_grads[i] + noise) / num_nodes)  # .cuda()

        self.optimizer.step()
