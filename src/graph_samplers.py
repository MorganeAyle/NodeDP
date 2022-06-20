import pdb
import random

import scipy.sparse as sp
import networkx as nx
import copy


class RandomWalks:
    def __init__(self, adj, nodes, depth, num_roots):
        self.adj = adj
        self.nodes = nodes
        self.depth = depth
        self.num_roots = num_roots
        self.graph = nx.from_scipy_sparse_matrix(self.adj)

    def sample(self):
        sampled_nodes = set()

        i = 0
        while i <= self.num_roots:
            i += 1
            # choose a random root
            root_idx = random.randint(0, len(self.nodes) - 1)
            node = self.nodes[root_idx]
            sampled_nodes.add(node)
            # construct random walk
            for _ in range(self.depth):
                valid_neighbors = [v for v in self.graph.neighbors(node)]
                if len(valid_neighbors):
                    neighbor = random.choice(valid_neighbors)
                    sampled_nodes.add(neighbor)
                    node = neighbor
                else:
                    break

        new_idx = {}
        count = 0

        rows = []
        cols = []
        vals = []

        for node in sampled_nodes:
            # create new idx for node
            new_idx[node] = count
            count += 1

        for node in sampled_nodes:
            for neighbor in self.graph.neighbors(node):
                if neighbor in sampled_nodes:
                    rows.append(new_idx[node])
                    cols.append(new_idx[neighbor])
                    vals.append(True)

        adj = sp.csr_matrix((vals, (rows, cols)), shape=(count, count))

        return adj, None, list(new_idx.keys())


class Baseline:
    def __init__(self, adj, nodes, depth, num_roots):
        self.adj = adj
        self.nodes = nodes
        self.depth = depth
        self.num_roots = num_roots
        self.graph = nx.from_scipy_sparse_matrix(self.adj)  # convert to networkx graph

        self.neighborhood = []
        self.edges = set()

    def depth_first_search(self, node, depth, graph):
        self.neighborhood.append(node)
        if depth == 0:
            return
        else:
            for neighbor in graph.neighbors(node):
                if neighbor not in self.neighborhood:
                    if (node, neighbor) not in self.edges:
                        self.edges.add((node, neighbor))
                        self.edges.add((neighbor, node))
                    return self.depth_first_search(neighbor, depth - 1, graph)

    def sample(self):
        edges = {}
        sampled_nodes = []
        roots = random.sample(list(self.nodes), self.num_roots)  # sample roots uniformly at random
        sampled_nodes.extend(roots)

        # TODO: do this step in pre-processed way
        for root in roots:
            node = copy.deepcopy(root)
            self.neighborhood = []
            self.edges = set()
            self.depth_first_search(node, self.depth, self.graph)
            neighborhood, neighborhood_edges = self.neighborhood, self.edges
            edges[root] = neighborhood_edges
            sampled_nodes.extend(neighborhood)

        new_idx = {}
        count = 0

        rows = []
        cols = []
        vals = []

        for node in sampled_nodes:
            if node not in new_idx:
                # create new idx for root
                new_idx[node] = count
                count += 1

        for root in roots:
            neighborhood_edges = edges[root]
            for (u, v) in neighborhood_edges:
                rows.append(new_idx[u])
                cols.append(new_idx[v])
                vals.append(True)
                cols.append(new_idx[v])
                rows.append(new_idx[u])
                vals.append(True)

        adj = sp.csr_matrix((vals, (rows, cols)), shape=(count, count))

        return adj, roots, list(new_idx.keys())


class PreDisjointRandomWalks:
    def __init__(self, rws, roots, num_roots):
        self.rws = rws
        self.roots = roots
        self.num_roots = num_roots

    def sample(self):
        roots = random.sample(self.roots, self.num_roots)
        new_idx = {}
        count = 0

        rows = []
        cols = []
        vals = []
        added_edges = set()

        for root in roots:
            # create new idx for root
            new_idx[root] = count
            count += 1

            # get neighbors
            neighbors = self.rws[root]
            for neighbor in neighbors[1:]:
                if neighbor not in new_idx:
                    new_idx[neighbor] = count
                    count += 1

        for root in roots:
            rw = self.rws[root]
            for irw in range(len(rw) - 1):  # rw has to include root
                if (rw[irw], rw[irw]+1) not in added_edges:
                    rows.append(new_idx[rw[irw]])
                    cols.append(new_idx[rw[irw+1]])
                    vals.append(True)
                    cols.append(new_idx[rw[irw]])
                    rows.append(new_idx[rw[irw+1]])
                    vals.append(True)

                    added_edges.add((rw[irw], rw[irw+1]))
                    added_edges.add((rw[irw+1], rw[irw]))

        adj = sp.csr_matrix((vals, (rows, cols)), shape=(count, count))

        return adj, roots, list(new_idx.keys())


class DisjointRandomWalks:
    def __init__(self, adj, nodes, depth, num_roots):
        self.adj = adj
        self.nodes = nodes
        self.depth = depth
        self.num_roots = num_roots
        self.graph = nx.from_scipy_sparse_matrix(self.adj)

    def sample(self):
        rws = {}
        sampled_nodes = set()
        roots = random.sample(list(self.nodes), self.num_roots)
        for root in roots:
            sampled_nodes.add(root)

        for root in roots:
            rw = [root]
            node = copy.deepcopy(root)
            for _ in range(self.depth):
                valid_neighbors = [v for v in self.graph.neighbors(node) if v not in sampled_nodes]
                if len(valid_neighbors):
                    neighbor = random.choice(valid_neighbors)
                    rw.append(neighbor)
                    sampled_nodes.add(neighbor)
                    node = copy.deepcopy(neighbor)
                else:
                    break
            rws[root] = rw

        new_idx = {}
        count = 0

        rows = []
        cols = []
        vals = []
        added_edges = set()

        for root in roots:
            # create new idx for root
            new_idx[root] = count
            count += 1

            # get neighbors
            neighbors = rws[root]
            for neighbor in neighbors[1:]:
                if neighbor not in new_idx:
                    new_idx[neighbor] = count
                    count += 1

        for root in roots:
            rw = rws[root]
            for irw in range(len(rw) - 1):  # rw has to include root
                if (rw[irw], rw[irw] + 1) not in added_edges:
                    rows.append(new_idx[rw[irw]])
                    cols.append(new_idx[rw[irw + 1]])
                    vals.append(True)
                    cols.append(new_idx[rw[irw]])
                    rows.append(new_idx[rw[irw + 1]])
                    vals.append(True)

                    added_edges.add((rw[irw], rw[irw + 1]))
                    added_edges.add((rw[irw + 1], rw[irw]))

        adj = sp.csr_matrix((vals, (rows, cols)), shape=(count, count))

        return adj, roots, list(new_idx.keys())


class PreDisjointRandomWalksWithsRestarts:
    def __init__(self, rws, edges, roots, num_roots):
        self.rws = rws
        self.edges = edges
        self.roots = roots
        self.num_roots = num_roots

    def sample(self):

        roots = random.sample(self.roots, self.num_roots)

        new_idx = {}
        count = 0

        rows = []
        cols = []
        vals = []

        for root in roots:
            # create new idx for root
            new_idx[root] = count
            count += 1

            # get neighbors
            neighbors = self.rws[root]
            for neighbor in neighbors[1:]:
                if neighbor not in new_idx:
                    new_idx[neighbor] = count
                    count += 1

        for root in roots:
            neighborhood_edges = self.edges[root]
            for (u, v) in neighborhood_edges:
                rows.append(new_idx[u])
                cols.append(new_idx[v])
                vals.append(True)
                cols.append(new_idx[v])
                rows.append(new_idx[u])
                vals.append(True)

        adj = sp.csr_matrix((vals, (rows, cols)), shape=(count, count))

        return adj, roots, list(new_idx.keys())
