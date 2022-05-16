import pdb
import time
import random
import copy

from src.sampling.graph_samplers import DisjointRandomWalks, RandomWalks, NodesUniformMaxDegree
from src.utils import adj_norm, adj_add_self_loops, bound_adj_degree

import numpy as np
import torch
import scipy.sparse as sp
from src.utils import _coo_scipy2torch


class Minibatch:
    def __init__(self, adj_full, adj_train, role, num_par_sampler, samples_per_proc, use_cuda, sampler_args):
        self.num_par_sampler = num_par_sampler
        self.samples_per_proc = samples_per_proc
        self.use_cuda = use_cuda
        self.sampler_method = sampler_args["method"]
        self.sampler_args = sampler_args
        self.batch_num = 0

        self.node_train = np.array(role['tr'])
        self.node_val = np.array(role['va'])
        self.node_test = np.array(role['te'])

        self.adj_full_norm_with_sl = _coo_scipy2torch(adj_norm(adj_add_self_loops(adj_full)).tocoo())
        self.adj_train = adj_train  # scipy sparse csr format
        self.deg_train = np.array(adj_add_self_loops(adj_train).sum(1).flatten().tolist()[0])

        self.subgraphs_remaining_indptr = []
        self.subgraphs_remaining_indices = []
        self.subgraphs_remaining_data = []
        self.subgraphs_remaining_nodes = []
        self.subgraphs_remaining_edge_index = []
        self.subgraphs_remaining_roots = []

        self.set_sampler()

    def set_sampler(self):
        self.subgraphs_remaining_indptr = []
        self.subgraphs_remaining_indices = []
        self.subgraphs_remaining_data = []
        self.subgraphs_remaining_nodes = []
        self.subgraphs_remaining_edge_index = []
        self.subgraphs_remaining_roots = []

        if self.sampler_method == 'drw':
            self.graph_sampler = DisjointRandomWalks(
                self.adj_train,
                self.node_train,
                int(self.sampler_args['num_root']),
                int(self.sampler_args['depth']),
                self.num_par_sampler,
                self.samples_per_proc
            )
        elif self.sampler_method == 'rw':
            self.graph_sampler = RandomWalks(
                self.adj_train,
                self.node_train,
                int(self.sampler_args['num_root']),
                int(self.sampler_args['depth']),
                self.num_par_sampler,
                self.samples_per_proc
            )
        elif self.sampler_method == 'nodes_max':
            self.graph_sampler = NodesUniformMaxDegree(
                self.adj_train,
                self.node_train,
                int(self.sampler_args['num_nodes']),
                int(self.sampler_args['max_degree']),
                self.num_par_sampler,
                self.samples_per_proc
            )
        else:
            raise NotImplementedError

    def sample_subgraphs(self, out):
        out("Sampling subgraphs...")
        if self.sampler_method == 'drw':
            new_adj = bound_adj_degree(copy.deepcopy(self.adj_train), self.sampler_args['max_degree'])
            self.deg_train = np.array(adj_add_self_loops(new_adj).sum(1).flatten().tolist()[0])
            self.graph_sampler = DisjointRandomWalks(
                new_adj,
                self.node_train,
                int(self.sampler_args['num_root']),
                int(self.sampler_args['depth']),
                self.num_par_sampler,
                self.samples_per_proc
            )
            _indptr, _indices, _data, _v, _edge_index, _roots = self.graph_sampler.par_sample()

        else:
            _indptr, _indices, _data, _v, _edge_index = self.graph_sampler.par_sample()
        self.subgraphs_remaining_indptr.extend(_indptr)
        self.subgraphs_remaining_indices.extend(_indices)
        self.subgraphs_remaining_data.extend(_data)
        self.subgraphs_remaining_nodes.extend(_v)
        self.subgraphs_remaining_edge_index.extend(_edge_index)
        if self.sampler_method == 'drw':
            self.subgraphs_remaining_roots.extend(_roots)
        out(f"Done sampling {len(_indptr)} subgraphs.")

    def sample_one_batch(self, out, mode='train'):
        if mode in ['val', 'test', 'valtest']:
            self.node_subgraph = np.arange(self.adj_full_norm_with_sl.shape[0])
            adj = self.adj_full_norm_with_sl
            root_subgraph = None
        else:
            assert mode == 'train'
            if len(self.subgraphs_remaining_nodes) == 0:
                self.sample_subgraphs(out)
                print()

            self.node_subgraph = self.subgraphs_remaining_nodes.pop()
            self.size_subgraph = len(self.node_subgraph)

            if self.sampler_method == 'drw':
                root_subgraph = self.subgraphs_remaining_roots.pop()

            adj = sp.csr_matrix(
                (
                    self.subgraphs_remaining_data.pop(),
                    self.subgraphs_remaining_indices.pop(),
                    self.subgraphs_remaining_indptr.pop()),
                    shape=(self.size_subgraph,self.size_subgraph,
                )
            )
            adj = _coo_scipy2torch(adj_norm(adj_add_self_loops(adj), self.deg_train[self.node_subgraph]).tocoo())

            if self.use_cuda:
                adj = adj.cuda()

            self.batch_num += 1

            out("Number of nodes in batch: " + str(adj.shape[0]))

        if self.sampler_method == 'drw':
            return self.node_subgraph, adj, root_subgraph
        return self.node_subgraph, adj
