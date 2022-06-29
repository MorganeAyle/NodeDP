import copy
import pdb

from src.graph_samplers import DisjointRandomWalks, RandomWalks, PreDisjointRandomWalks, Baseline, \
    PreDisjointRandomWalksWithsRestarts
from src.utils import adj_norm, adj_add_self_loops, bound_adj_degree, sample_rws, sample_rws_w_restarts
from src.constants import BOUND_DEGREE_METHODS, DP_METHODS

import numpy as np
import scipy.sparse as sp
from src.utils import _coo_scipy2torch


class Minibatch:
    def __init__(self, adj_full, adj_train, role, use_cuda, sampler_args, model_args, fout):
        self.use_cuda = use_cuda
        self.sampler_method = sampler_args["method"]
        self.sampler_args = sampler_args
        self.model_args = model_args
        self.batch_num = 0
        self.fout = fout

        self.node_train = np.array(role['tr'])
        self.node_val = np.array(role['va'])
        self.node_test = np.array(role['te'])

        self.adj_full_norm_with_sl = _coo_scipy2torch(adj_norm(adj_add_self_loops(adj_full)).tocoo())
        self.adj_val_norm_with_sl = _coo_scipy2torch(adj_norm(adj_add_self_loops(
            adj_full[self.node_val][:, self.node_val])).tocoo())
        # bound degree of adjacency matrix if required by method
        if self.sampler_method in BOUND_DEGREE_METHODS:
            adj_train = bound_adj_degree(adj_train, self.sampler_args['max_degree'])
        self.adj_train = adj_train  # should be in scipy sparse csr format

        self.subgraphs_remaining_indptr = []
        self.subgraphs_remaining_indices = []
        self.subgraphs_remaining_data = []
        self.subgraphs_remaining_nodes = []
        self.subgraphs_remaining_edge_index = []
        self.subgraphs_remaining_roots = []
        self.subgraphs_remaining_degrees = []
        self.subgraphs_remaining_depths = []

        self.graph_sampler = None
        self.set_sampler()

        self.size_subgraph = 0
        self.node_subgraph = []

    def set_sampler(self):
        self.subgraphs_remaining_indptr = []
        self.subgraphs_remaining_indices = []
        self.subgraphs_remaining_data = []
        self.subgraphs_remaining_nodes = []
        self.subgraphs_remaining_edge_index = []
        self.subgraphs_remaining_roots = []

        if self.sampler_method == 'drw':
            self.graph_sampler = DisjointRandomWalks(self.adj_train, self.node_train,
                                                     int(self.sampler_args['depth']),
                                                     int(self.sampler_args['num_root']))

        elif self.sampler_method == 'rw':
            self.graph_sampler = RandomWalks(
                self.adj_train,
                self.node_train,
                int(self.sampler_args['depth']),
                int(self.sampler_args['num_root'])
            )
        elif self.sampler_method == 'baseline':
            self.graph_sampler = Baseline(
                self.adj_train,
                self.node_train,
                int(self.model_args['num_layers']),
                int(self.sampler_args['num_root']),
            )
        elif self.sampler_method == 'pre_drw':
            self.fout("Sampling random walks...")
            rws, roots = sample_rws(self.adj_train, self.node_train, int(self.sampler_args['depth']))
            self.fout("Done sampling.")
            self.graph_sampler = PreDisjointRandomWalks(
                rws, roots, int(self.sampler_args['num_root'])
            )
        elif self.sampler_method == 'pre_drw_w_restarts':
            self.fout("Sampling random walks with restarts...")
            rws, edges, roots = sample_rws_w_restarts(self.adj_train, self.node_train,
                                                      int(self.sampler_args['depth']),
                                                      int(self.sampler_args['restarts']))
            self.fout("Done sampling.")
            self.graph_sampler = PreDisjointRandomWalksWithsRestarts(
                rws, edges, roots, int(self.sampler_args['num_root'])
            )
        else:
            raise NotImplementedError

    def sample_one_batch(self, out, mode='train'):
        if mode == 'val':
            self.node_subgraph = self.node_val
            adj = self.adj_val_norm_with_sl
            root_subgraph = None

        elif mode in ['test', 'valtest']:
            self.node_subgraph = np.arange(self.adj_full_norm_with_sl.shape[0])
            adj = self.adj_full_norm_with_sl
            root_subgraph = None

        else:
            assert mode == 'train'
            adj, root_subgraph, self.node_subgraph = self.graph_sampler.sample()
            self.size_subgraph = len(self.node_subgraph)

            if self.model_args['arch'] == 'GraphSAGE':
                adj = _coo_scipy2torch(adj_norm(adj).tocoo())
            else:
                adj = _coo_scipy2torch(adj_norm(adj_add_self_loops(adj)).tocoo())

            if self.use_cuda:
                adj = adj.cuda()

            self.batch_num += 1
            out("Number of nodes in batch: " + str(adj.shape[0]))

        return self.node_subgraph, adj, root_subgraph
