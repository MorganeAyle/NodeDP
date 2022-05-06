import pdb

import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import collections
import scipy

from src.models import create_model
from src.utils import _coo_scipy2torch

from functorch import make_functional_with_buffers, vmap, grad


class Trainer:
    def __init__(self, training_args, model_args, feats, class_arr, use_cuda, minibatch, out, only_roots):
        self.feats = torch.from_numpy(feats.astype(np.float32))
        self.labels = torch.from_numpy(class_arr.astype(np.float32))
        if use_cuda:
            self.feats = self.feats.cuda()
            self.labels = self.labels.cuda()
        self.use_cuda = use_cuda
        self.only_roots = only_roots

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

        if use_cuda:
            self.model.to('cuda')

        # Optimizer
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=training_args['lr'])

        # Get gradient norms
        if training_args['method'] == 'normal':
            self.clip_norm = training_args['clip_norm']
        if training_args['method'] in ['ours', 'node_dp_max_degree']:
            self.grad_norms = torch.tensor(self.estimate_init_grad_norms(*self.minibatch.sample_one_batch(out, mode='train')))
            self.C = np.sqrt(sum(self.grad_norms ** 2))

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

    def dp_train_step(self, nodes, adj, roots=None, clip_norm=0.002, sigma=1.):
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

                self.optimizer.zero_grad()
                preds = self.model(self.feats[walk_nodes], new_adj)
                loss = self._loss(preds, self.labels[walk_nodes])
                tmp = torch.autograd.grad(loss, list(self.model.parameters()), allow_unused=True)
                total_norm = 0
                for t in tmp:
                    if t is not None:
                        total_norm += t.norm() ** 2
                tmpp = [t / max(1, torch.sqrt(total_norm) / clip_norm) for t in tmp if t is not None]
                grads.append(tmpp)
                total_loss += loss
                total_preds.extend(preds)
                total_labels.extend(self.labels[walk_nodes])

        grads = zip(*grads)
        new_grads = []
        for shards in grads:
            if shards[0] is not None:
                grad_sum = torch.stack(shards).sum(0)
                new_grads.append(grad_sum + torch.normal(torch.zeros_like(grad_sum),
                                                         torch.ones_like(grad_sum) * (sigma * clip_norm) ** 2) / len(
                    shards))

        for i, p in enumerate(self.model.parameters()):
            p.grad = new_grads[i]
            if i == len(new_grads) - 1:
                break
        self.optimizer.step()
        return total_loss, self.predict(torch.stack(total_preds)), torch.stack(total_labels)

    def dp_train_step_fast(self, nodes, adj, roots=None, sigma=1.):
        self.model.train()
        self.optimizer.zero_grad()

        self.fmodel, params, buffers = make_functional_with_buffers(self.model)
        ft_compute_grad = grad(self.compute_loss_stateless_model)
        ft_compute_sample_grad = vmap(ft_compute_grad, in_dims=(None, None, None, None, 0, 0))

        if not self.only_roots:
            ft_per_sample_grads = ft_compute_sample_grad(params, buffers, self.feats[nodes], adj.to_dense(),
                                                         self.labels[nodes], torch.arange(nodes.size))
        else:
            idx = []
            for i, v in enumerate(nodes):
                if v in roots:
                    idx.append(i)
            ft_per_sample_grads = ft_compute_sample_grad(params, buffers, self.feats[nodes], adj.to_dense(),
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

    def compute_loss_stateless_model(self, params, buffers, x, adj, target, idx):
        targets = target.unsqueeze(0)
        index = idx.unsqueeze(0)

        predictions = self.fmodel(params, buffers, x, adj)
        if not self.only_roots:
            loss = self._loss(predictions[index], targets)
        else:
            loss = self._loss(predictions[index], targets)
        return loss
