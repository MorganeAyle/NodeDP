import pdb

import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import collections
import scipy
from src import autograd_hacks

from src.models import create_model
from src.utils import _coo_scipy2torch

from functorch import make_functional_with_buffers, vmap, grad


class Trainer:
    def __init__(self, training_args, model_args, feats, class_arr, use_cuda, minibatch, out, sampler_args):
        self.feats = torch.from_numpy(feats.astype(np.float32))
        self.labels = torch.from_numpy(class_arr.astype(np.float32))
        if use_cuda:
            self.feats = self.feats.cuda()
            self.labels = self.labels.cuda()
        self.use_cuda = use_cuda
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

        autograd_hacks.add_hooks(self.model)

        if use_cuda:
            self.model.to('cuda')

        # Optimizer
        if training_args['optim'].lower() == 'adam':
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=training_args['lr'])
        elif training_args['optim'].lower() == 'sgd':
            self.optimizer = torch.optim.SGD(self.model.parameters(), lr=training_args['lr'])
        else:
            raise NotImplementedError

        # Get gradient norms
        if training_args['method'] == 'normal':
            self.clip_norm = training_args['clip_norm']
        if training_args['method'] in ['ours', 'node_dp_max_degree']:
            self.grad_norms = torch.tensor(self.estimate_init_grad_norms(*self.minibatch.sample_one_batch(out, mode='train')))
            self.C = training_args['C%'] * np.sqrt(sum(self.grad_norms ** 2))
            self.grad_norms = training_args['C%'] * self.grad_norms

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
        preds = self.model(self.feats[nodes], adj)
        loss = self._loss(preds, self.labels[nodes])
        grads = torch.autograd.grad(loss, list(self.model.parameters()))
        return [grad.norm() for grad in grads]

    def train_step(self, nodes, adj, roots=None):
        self.model.train()
        self.optimizer.zero_grad()
        preds = self.model(self.feats[nodes], adj)

        loss = self._loss(preds, self.labels[nodes])
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_norm)
        self.optimizer.step()
        return loss, self.predict(preds), self.labels[nodes]

    def dp_train_step(self, nodes, adj, roots=None, sigma=1.):
        self.model.train()
        self.optimizer.zero_grad()

        total_preds = []
        total_labels = []
        processed_nodes = []
        node_subgraph_dict = {}
        for i, node in enumerate(nodes):
            node_subgraph_dict[node] = i

        grads = []
        total_loss = 0
        for i, node in enumerate(nodes):
            if node not in processed_nodes:
                walk_nodes = []
                nodes_to_visit = collections.deque([node])
                while len(nodes_to_visit):
                    n = nodes_to_visit.popleft()
                    walk_nodes.append(n)
                    for u in adj[node_subgraph_dict[n]]._indices()[0]:
                        node_u = nodes[int(u)]
                        if node_u not in walk_nodes and node_u not in nodes_to_visit:
                            nodes_to_visit.append(node_u)

                processed_nodes.extend(walk_nodes)

                new_adj = np.zeros((len(walk_nodes), len(walk_nodes)))
                for j, n in enumerate(walk_nodes):
                    for idx_u, u in enumerate(adj[node_subgraph_dict[n]]._indices()[0]):
                        node_u = nodes[int(u)]
                        k = walk_nodes.index(node_u)
                        val = adj[node_subgraph_dict[n]]._values()[idx_u]
                        # val = 1.
                        new_adj[j, k] = val
                        new_adj[k, j] = val

                walk_nodes = np.array(walk_nodes)
                new_adj = scipy.sparse.csr_matrix(new_adj)
                new_adj = _coo_scipy2torch(new_adj.tocoo())
                if self.use_cuda:
                    new_adj = new_adj.cuda()

                # compute gradient for each sample in walk and clip it
                for inode, walk_node in enumerate(walk_nodes):
                    if self.sampler_args["only_roots"] and walk_node not in roots:
                        continue
                    self.optimizer.zero_grad()
                    preds = self.model(self.feats[walk_nodes], new_adj)
                    pred, label = preds[inode].unsqueeze(0), self.labels[walk_node].unsqueeze(0)
                    loss = self._loss(pred, label)

                    model_grads = torch.autograd.grad(loss, list(self.model.parameters()), allow_unused=True)
                    clipped_model_grads = []
                    for ti, t in enumerate(model_grads):
                        if t is not None:
                            clipped_model_grads.append(t / max(1, t.norm() / self.grad_norms[ti]))

                    grads.append(clipped_model_grads)
                    total_loss += loss
                    total_preds.extend(pred)
                    total_labels.extend(label)

        # sum al gradients and add noise
        grads = zip(*grads)
        new_grads = []
        for i, shards in enumerate(grads):
            if shards[0] is not None:
                grad_sum = torch.stack(shards).sum(0)
                new_grads.append((grad_sum + torch.normal(torch.zeros_like(grad_sum),
                                                         torch.ones_like(grad_sum) * (sigma * self.grad_norms[i]) ** 2)) / len(shards))

        # set gradients of model to new gradients and do optimizer step
        for i, p in enumerate(self.model.parameters()):
            p.grad = new_grads[i]
        self.optimizer.step()

        return total_loss, self.predict(torch.stack(total_preds)), torch.stack(total_labels)

    def dp_train_step_fast(self, nodes, adj, roots=None, sigma=1.):
        self.model.train()
        self.optimizer.zero_grad()

        self.fmodel, params, buffers = make_functional_with_buffers(self.model)
        ft_compute_grad = grad(self.compute_loss_stateless_model)
        ft_compute_sample_grad = vmap(ft_compute_grad, in_dims=(None, None, None, None, 0, 0), randomness='same')

        if not (self.sampler_args["method"] == "drw" and self.sampler_args["only_roots"]):
            ft_per_sample_grads = ft_compute_sample_grad(params, buffers, self.feats[nodes], adj,
                                                         self.labels[nodes], torch.arange(nodes.size))
        else:
            idx = []
            for i, v in enumerate(nodes):
                if v in roots:
                    idx.append(i)
            ft_per_sample_grads = ft_compute_sample_grad(params, buffers, self.feats[nodes], adj,
                                                         self.labels[nodes][idx], torch.arange(nodes.size)[idx])

        grads_norms = torch.zeros((ft_per_sample_grads[0].shape[0], self.grad_norms.shape[0]),
                                  device='cuda' if self.use_cuda else 'cpu')

        for i, layer in enumerate(ft_per_sample_grads):
            grads_norms[:, i] = layer.norm(dim=tuple(np.arange(len(layer.shape))[1:]))

        new_grads = []
        for i, layer in enumerate(ft_per_sample_grads):
            denom = torch.maximum(torch.tensor([1] * len(grads_norms), device=grads_norms.device),
                                  grads_norms[:, i].flatten() / self.grad_norms[i])
            for _ in range(len(layer.shape) - 1):
                denom = denom.unsqueeze(-1)

            new_grad = layer / denom
            new_grad = new_grad.sum(0)
            mean = torch.zeros_like(new_grad)
            std = torch.ones_like(new_grad) * (sigma * self.grad_norms[i]) ** 2
            noise = torch.normal(mean, std)
            new_grads.append((new_grad + noise) / len(grads_norms))

        # new_grads = []
        # for layer in ft_per_sample_grads:
        #     new_grads.append(layer.mean(0))

        for i, p in enumerate(self.model.parameters()):
            p.grad = new_grads[i]
        self.optimizer.step()

    def dp_train_step_fast2(self, nodes, adj, roots=None, sigma=1.):
        self.model.train()
        self.optimizer.zero_grad()
        autograd_hacks.clear_backprops(self.model)
        if not (self.sampler_args["method"] == "drw" and self.sampler_args["only_roots"]):
            preds = self.model(self.feats[nodes], adj)
            self._loss(preds, self.labels[nodes]).backward(retain_graph=True)
            num_nodes = len(nodes)
            idx = list(range(len(nodes)))
        else:
            idx = []
            for i, v in enumerate(nodes):
                if v in roots:
                    idx.append(i)
            preds = self.model(self.feats[nodes], adj)
            self._loss(preds[idx], self.labels[nodes][idx]).backward(retain_graph=True)
            num_nodes = len(roots)
        autograd_hacks.compute_grad1(self.model)

        # for param in self.model.parameters():
        #     print((param.grad1.mean(dim=0) - param.grad).norm())
        #     assert (torch.allclose(param.grad1.mean(dim=0), param.grad))

        grads_norms = torch.zeros((num_nodes, self.grad_norms.shape[0]))

        for i, param in enumerate(self.model.parameters()):
            param_grad = param.grad1.cpu()
            grads_norms[:, i] = param_grad.norm(dim=tuple(np.arange(len(param_grad.shape))[1:]))[idx]

        new_grads = []
        for i, param in enumerate(self.model.parameters()):
            param_grad = param.grad1.cpu()
            denom = torch.maximum(torch.tensor([1] * len(grads_norms), device=grads_norms.device),
                                  grads_norms[:, i].flatten() / self.grad_norms[i])
            for _ in range(len(param_grad.shape) - 1):
                denom = denom.unsqueeze(-1)

            new_grad = param_grad[idx] / denom
            new_grad = new_grad.sum(0)
            mean = torch.zeros_like(new_grad)
            std = torch.ones_like(new_grad) * (sigma * self.grad_norms[i]) ** 2
            noise = torch.normal(mean, std)
            new_grads.append((new_grad + noise) / len(grads_norms))

        for i, p in enumerate(self.model.parameters()):
            p.grad = new_grads[i].cuda()
        self.optimizer.step()

    def compute_loss_stateless_model(self, params, buffers, x, adj, target, idx):
        targets = target.unsqueeze(0)
        index = idx.unsqueeze(0)

        predictions = self.fmodel(params, buffers, x, adj)
        loss = self._loss(predictions[index], targets)
        return loss
