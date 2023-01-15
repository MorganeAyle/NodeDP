import copy
import pdb

from src.graph_samplers import DisjointRandomWalks, RandomWalks, PreDisjointRandomWalks, Baseline, \
    PreDisjointRandomWalksWithsRestarts, UniformNodes
from src.utils import adj_norm, adj_add_self_loops, bound_adj_degree, sample_rws, sample_rws_w_restarts, DFS
from src.constants import BOUND_DEGREE_METHODS, DP_METHODS

import numpy as np
import scipy.sparse as sp
import networkx as nx
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
        # bound degree of adjacency matrix if required by method
        if self.sampler_method in BOUND_DEGREE_METHODS:
            adj_train = bound_adj_degree(adj_train, self.sampler_args['max_degree'])
        self.adj_train = adj_train  # should be in scipy sparse csr format
        self.G = nx.from_scipy_sparse_matrix(self.adj_train)

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

    def set_pre_drw_sampler(self):
        self.fout("Sampling random walks...")
        rws, roots = sample_rws(self.G,
                                copy.deepcopy(self.node_train),
                                int(self.sampler_args['depth']))
        self.fout("Done sampling.")
        self.graph_sampler = PreDisjointRandomWalks(rws,
                                                    roots,
                                                    int(self.sampler_args['num_root']),
                                                    self.sampler_args['split_lots_into_batches']
                                                    )

    def set_pre_drw_w_restarts_sampler(self):
        self.fout("Sampling random walks with restarts...")
        rws, edges, roots = sample_rws_w_restarts(self.G,
                                                  self.node_train,
                                                  int(self.sampler_args['depth']),
                                                  int(self.sampler_args['restarts']))
        self.fout("Done sampling.")
        self.graph_sampler = PreDisjointRandomWalksWithsRestarts(rws,
                                                                 edges,
                                                                 roots,
                                                                 int(self.sampler_args['num_root']),
                                                                 self.sampler_args['split_lots_into_batches']
                                                                 )

    def set_sampler(self):
        self.subgraphs_remaining_indptr = []
        self.subgraphs_remaining_indices = []
        self.subgraphs_remaining_data = []
        self.subgraphs_remaining_nodes = []
        self.subgraphs_remaining_edge_index = []
        self.subgraphs_remaining_roots = []

        if self.sampler_method == 'uniform':
            self.graph_sampler = UniformNodes(
                self.adj_train,
                self.node_train,
                int(self.sampler_args['num_root'])
            )
        elif self.sampler_method == 'rw':
            self.graph_sampler = RandomWalks(
                self.adj_train,
                self.node_train,
                int(self.sampler_args['depth']),
                int(self.sampler_args['num_root'])
            )
        elif self.sampler_method == 'baseline':
            graph = nx.from_scipy_sparse_matrix(self.adj_train)  # convert to networkx graph
            dfs = DFS(self.node_train, int(self.model_args['num_layers']), graph)
            dfs.sample_all_neighborhoods()
            self.graph_sampler = Baseline(
                self.adj_train,
                self.node_train,
                int(self.model_args['num_layers']),
                int(self.sampler_args['num_root']),
                self.sampler_args['split_lots_into_batches'],
                dfs.neighborhoods,
                dfs.neighborhoods_edges
            )
        elif self.sampler_method == 'pre_drw':
            self.set_pre_drw_sampler()
        elif self.sampler_method == 'pre_drw_w_restarts':
            self.set_pre_drw_w_restarts_sampler()
        elif self.sampler_method == 'drw':
            self.graph_sampler = DisjointRandomWalks(self.adj_train,
                                                     self.node_train,
                                                     int(self.sampler_args['depth']),
                                                     int(self.sampler_args['num_root']),
                                                     self.sampler_args['split_lots_into_batches'])
        else:
            raise NotImplementedError

    def sample_one_batch(self, out, mode='train'):
        if mode == 'val':
            self.node_subgraph = np.arange(self.adj_full_norm_with_sl.shape[0])
            adj = self.adj_full_norm_with_sl
            root_subgraph = self.node_val

        elif mode == 'test':
            self.node_subgraph = np.arange(self.adj_full_norm_with_sl.shape[0])
            adj = self.adj_full_norm_with_sl
            root_subgraph = self.node_test

        else:
            assert mode == 'train'

            if self.sampler_method == 'pre_drw' and \
                    self.sampler_args['preprocess_graph_every'] != -1 and \
                    self.batch_num % self.sampler_args['preprocess_graph_every'] == 0 and \
                    self.batch_num > 0:
                self.set_pre_drw_sampler()
            elif self.sampler_method == 'pre_drw_w_restarts' and \
                    self.sampler_args['preprocess_graph_every'] != -1 and \
                    self.batch_num % self.sampler_args['preprocess_graph_every'] == 0 and \
                    self.batch_num > 0:
                self.set_pre_drw_w_restarts_sampler()

            adj, root_subgraph, self.node_subgraph = self.graph_sampler.sample()
            self.size_subgraph = sum([len(subgraph) for subgraph in self.node_subgraph])

            if self.model_args['arch'] == 'GraphSAGE':
                adj = [_coo_scipy2torch(adj_norm(a).tocoo()) for a in adj]
            else:
                adj = [_coo_scipy2torch(adj_norm(adj_add_self_loops(a)).tocoo()) for a in adj]

            if self.use_cuda:
                adj = [a.cuda() for a in adj]

            self.batch_num += 1
            out("Number of nodes in batch: " + str(self.size_subgraph))

        return self.node_subgraph, adj, root_subgraph
