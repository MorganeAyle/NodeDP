import copy

import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import collections
import scipy
from src import autograd_hacks

from src.models import create_model
from src.utils import _coo_scipy2torch
from src.constants import DP_METHODS, NON_DP_METHODS


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
            self.grad_norms = self.estimate_init_grad_norms(*self.minibatch.sample_one_batch(out, mode='train'))
            # self.C = training_args['C%'] * np.sqrt(sum(self.grad_norms ** 2))
            self.grad_norms = training_args['C%'] * self.grad_norms
            self.C = np.sqrt(sum(self.grad_norms ** 2))
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
        # preds = self.model(self.feats[nodes], adj)
        # loss = self._loss(preds, self.labels[nodes])
        # grads = torch.autograd.grad(loss, list(self.model.parameters()))
        # grad_norms = ([grad.norm() for grad in grads])
        # return torch.tensor(grad_norms)

        grad_norms = []
        i = 0
        count = 0
        while count < 10:
            i += 1
            if nodes[i] in roots:
                count += 1
                self.model.zero_grad()
                preds = self.model(self.feats[nodes], adj)
                loss = self._loss(preds[i].unsqueeze(0), self.labels[nodes][i].unsqueeze(0))
                grads = torch.autograd.grad(loss, list(self.model.parameters()))
                grad_norms.append([grad.norm() for grad in grads])
        return torch.tensor(grad_norms).mean(0)
        #
        # grad_norms = []
        # self.model.train()
        # self.model.zero_grad()
        # autograd_hacks.clear_backprops(self.model)
        # idx = []
        # for i, v in enumerate(nodes):
        #     if v in roots:
        #         idx.append(i)
        # preds = self.model(self.feats[nodes] adj)
        # self._loss(preds[idx], self.labels[nodes][idx]).backward(retain_graph=True)
        # autograd_hacks.compute_grad1(self.model)
        # for param in self.model.parameters():
        #     del param.grad
        #     grad_norms.append(param.grad1.norm(dim=tuple(np.arange(len(param.grad1.shape))[1:])).mean())
        # return torch.tensor(grad_norms)

    def train_step(self, nodes, adj, roots=None):
        self.model.train()
        self.optimizer.zero_grad()
        preds = self.model(self.feats[nodes], adj)

        loss = self._loss(preds, self.labels[nodes])
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_norm)
        self.optimizer.step()
        return loss, self.predict(preds), self.labels[nodes]

    def dp_train_step_fast(self, nodes, adj, roots=None, sigma=1.):
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
        num_nodes = len(roots)
        autograd_hacks.compute_grad1(self.model)

        # to avoid memory problems
        for param in self.model.parameters():
            del param.grad
            param.grad1 = param.grad1.cpu()

        # compute new noisy gradients
        for i, param in enumerate(self.model.parameters()):
            param_grad = param.grad1
            grad_norm = param_grad.norm(dim=tuple(np.arange(len(param_grad.shape))[1:]))[idx]
            denom = torch.maximum(torch.tensor([1] * num_nodes, device=grad_norm.device),
                                  grad_norm.flatten() / self.grad_norms[i])
            for _ in range(len(param_grad.shape) - 1):
                denom = denom.unsqueeze(-1)

            new_grad = param_grad[idx] / denom  # clip gradients
            new_grad = new_grad.sum(0)
            mean = torch.zeros_like(new_grad)
            # std = torch.ones_like(new_grad) * (sigma * self.grad_norms[i]) ** 2
            std = torch.ones_like(new_grad) * sigma ** 2
            noise = torch.normal(mean, std)

            del param.grad1
            param.grad = ((new_grad + noise) / num_nodes).cuda()

        self.optimizer.step()

