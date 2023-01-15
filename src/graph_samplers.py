import pdb
import random
import time

import networkx as nx
import copy
import numpy as np

from src.constants import MAX_BATCH_SIZE
from src.utils import binary_search
from src.graph_samplers_utils import divide_chunks, map_sampled_nodes_to_new_idx, get_sampled_nodes_from_rws, \
    convert_sampled_nodes_to_adj, convert_neighborhood_edges_to_adj, convert_rws_to_adj


class UniformNodes:
    def __init__(self, adj, nodes, num_roots):
        self.adj = adj
        self.nodes = nodes
        self.num_roots = num_roots
        self.graph = nx.from_scipy_sparse_matrix(self.adj)

    def sample(self):
        sampled_nodes = np.unique(random.sample(list(self.nodes), self.num_roots))

        new_idx, count = map_sampled_nodes_to_new_idx(sampled_nodes)

        adj = convert_sampled_nodes_to_adj(sampled_nodes, self.graph, new_idx, count)

        assert not len((adj != adj.T).data)
        assert not adj.diagonal().sum()

        return [adj], [sampled_nodes], [sampled_nodes]


class RandomWalks:
    def __init__(self, adj, nodes, depth, num_roots):
        self.nodes = nodes
        self.depth = depth
        self.num_roots = num_roots
        self.graph = nx.from_scipy_sparse_matrix(adj)

    def sample(self):
        sampled_nodes = []
        sampled_roots = []

        for _ in range(self.num_roots):
            # choose a random root
            root_idx = random.randint(0, len(self.nodes) - 1)
            node = self.nodes[root_idx]
            sampled_nodes.append(node)
            sampled_roots.append(node)
            # construct random walk
            for _ in range(self.depth):
                valid_neighbors = [v for v in self.graph.neighbors(node)]
                if len(valid_neighbors):
                    neighbor = random.choice(valid_neighbors)
                    sampled_nodes.append(neighbor)
                    node = neighbor
                else:
                    break

        sampled_nodes = np.unique(sampled_nodes)

        new_idx, count = map_sampled_nodes_to_new_idx(sampled_nodes)

        adj = convert_sampled_nodes_to_adj(sampled_nodes, self.graph, new_idx, count)

        sampled_training_nodes = [node for node in sampled_nodes if node in self.nodes]

        assert not len((adj != adj.T).data)
        assert not adj.diagonal().sum()

        return [adj], [sampled_training_nodes], [sampled_nodes]


class Baseline:
    def __init__(self, adj, nodes, depth, num_roots, split_lots_into_batches, neighborhoods, neighborhoods_edges):
        self.adj = adj
        self.nodes = nodes
        self.depth = depth
        self.num_roots = num_roots
        self.split_lots_into_batches = split_lots_into_batches
        self.neighborhoods = neighborhoods
        self.neighborhoods_edges = neighborhoods_edges

    def sample(self):
        sampled_roots = random.sample(list(self.nodes), self.num_roots)  # sample roots uniformly at random
        if self.split_lots_into_batches:
            batched_roots = list(divide_chunks(sampled_roots, MAX_BATCH_SIZE))
        else:
            batched_roots = [sampled_roots]

        all_adj = []
        all_idx = []
        all_roots = []

        for roots in batched_roots:
            sampled_nodes = copy.deepcopy(roots)
            edges = {}
            for root in roots:
                neighborhood, neighborhood_edges = self.neighborhoods[root], self.neighborhoods_edges[root]
                edges[root] = neighborhood_edges
                sampled_nodes.extend(neighborhood)

            sampled_nodes = np.unique(sampled_nodes)

            new_idx, count = map_sampled_nodes_to_new_idx(sampled_nodes)

            adj = convert_neighborhood_edges_to_adj(roots, edges, new_idx, count)

            all_adj.append(adj)
            all_roots.append(roots)
            all_idx.append(list(new_idx.keys()))

            assert not len((adj != adj.T).data)
            assert not adj.diagonal().sum()

        return all_adj, all_roots, all_idx


class PreDisjointRandomWalks:
    def __init__(self, rws, roots, num_roots, split_lots_into_batches):
        self.rws = rws
        self.roots = roots
        self.num_roots = num_roots
        self.split_lots_into_batches = split_lots_into_batches

    def sample(self):
        sampled_roots = random.sample(self.roots, self.num_roots)
        if self.split_lots_into_batches:
            batched_roots = list(divide_chunks(sampled_roots, MAX_BATCH_SIZE))
        else:
            batched_roots = [sampled_roots]

        all_adj = []
        all_idx = []
        all_roots = []

        for roots in batched_roots:
            roots, sampled_nodes = get_sampled_nodes_from_rws(roots, self.rws)

            new_idx, count = map_sampled_nodes_to_new_idx(sampled_nodes)

            adj = convert_rws_to_adj(roots, self.rws, new_idx, count)

            all_adj.append(adj)
            all_roots.append(roots)
            all_idx.append(sampled_nodes)

            assert not len((adj != adj.T).data)
            assert not adj.diagonal().sum()

        return all_adj, all_roots, all_idx


class PreDisjointRandomWalksWithsRestarts:
    def __init__(self, rws, edges, roots, num_roots, split_lots_into_batches):
        self.rws = rws
        self.edges = edges
        self.roots = roots
        self.num_roots = num_roots
        self.split_lots_into_batches = split_lots_into_batches

    def sample(self):
        sampled_roots = random.sample(self.roots, self.num_roots)
        if self.split_lots_into_batches:
            batched_roots = list(divide_chunks(sampled_roots, MAX_BATCH_SIZE))
        else:
            batched_roots = [sampled_roots]

        all_adj = []
        all_idx = []
        all_roots = []

        for roots in batched_roots:
            roots, sampled_nodes = get_sampled_nodes_from_rws(roots, self.rws)

            new_idx, count = map_sampled_nodes_to_new_idx(sampled_nodes)

            adj = convert_neighborhood_edges_to_adj(roots, self.edges, new_idx, count)

            all_adj.append(adj)
            all_roots.append(roots)
            all_idx.append(sampled_nodes)

            assert not len((adj != adj.T).data)
            assert not adj.diagonal().sum()

        return all_adj, all_roots, all_idx


class DisjointRandomWalks:
    def __init__(self, adj, nodes, depth, num_roots, split_lots_into_batches):
        self.adj = adj
        self.nodes = nodes
        self.depth = depth
        self.num_roots = num_roots
        self.graph = nx.from_scipy_sparse_matrix(self.adj)
        self.split_lots_into_batches = split_lots_into_batches

    def sample(self):
        rws = {}
        sampled_roots = random.sample(list(self.nodes), self.num_roots)
        if self.split_lots_into_batches:
            batched_roots = list(divide_chunks(sampled_roots, MAX_BATCH_SIZE))
        else:
            batched_roots = [sampled_roots]
        sampled_nodes = set(copy.deepcopy(sampled_roots))

        all_adj = []
        all_idx = []
        all_roots = []

        for roots in batched_roots:
            batch_sampled_nodes = []
            for root in roots:
                batch_sampled_nodes.append(root)
                rw = [root]
                node = copy.deepcopy(root)
                for _ in range(self.depth):
                    valid_neighbors = [v for v in self.graph.neighbors(node) if v not in sampled_nodes]
                    if len(valid_neighbors):
                        neighbor = random.choice(valid_neighbors)
                        rw.append(neighbor)
                        sampled_nodes.add(neighbor)
                        batch_sampled_nodes.append(neighbor)
                        node = copy.deepcopy(neighbor)
                    else:
                        break
                rws[root] = rw

            roots = np.unique(roots)
            batch_sampled_nodes = np.unique(batch_sampled_nodes)

            new_idx, count = map_sampled_nodes_to_new_idx(batch_sampled_nodes)

            adj = convert_rws_to_adj(roots, rws, new_idx, count)

            all_adj.append(adj)
            all_roots.append(roots)
            all_idx.append(batch_sampled_nodes)

            assert not len((adj != adj.T).data)
            assert not adj.diagonal().sum()

        return all_adj, all_roots, all_idx
