import copy

import scipy.sparse as sp
import numpy as np

from src.utils import binary_search

import time


def divide_chunks(l, n):
    # looping till length l
    for i in range(0, len(l), n):
        yield l[i:i + n]


def map_sampled_nodes_to_new_idx(sampled_nodes):
    new_idx = {}
    count = 0

    for node in sampled_nodes:
        # create new idx for node
        new_idx[node] = count
        count += 1
    return new_idx, count


def get_sampled_nodes_from_rws(roots, rws):
    sampled_nodes = copy.deepcopy(roots)
    for root in roots:
        for neighbor in rws[root][1:]:
            sampled_nodes.append(neighbor)
    roots = np.unique(roots)
    sampled_nodes = np.unique(sampled_nodes)
    return roots, sampled_nodes


def convert_sampled_nodes_to_adj(sampled_nodes, graph, new_idx, count):
    rows = []
    cols = []
    vals = []

    for node in sampled_nodes:
        for neighbor in graph.neighbors(node):
            if binary_search(sampled_nodes, 0, len(sampled_nodes)-1, neighbor) != -1:
                rows.append(new_idx[node])
                cols.append(new_idx[neighbor])
                vals.append(True)

    adj = sp.csr_matrix((vals, (rows, cols)), shape=(count, count))
    return adj


def convert_rws_to_adj(roots, rws, new_idx, count):
    rows = []
    cols = []
    vals = []
    added_edges = set()

    for root in roots:
        rw = rws[root]
        for irw in range(len(rw) - 1):  # rw has to include root at index 0
            if (rw[irw], rw[irw + 1]) not in added_edges:
                rows.append(new_idx[rw[irw]])
                cols.append(new_idx[rw[irw + 1]])
                vals.append(True)
                cols.append(new_idx[rw[irw]])
                rows.append(new_idx[rw[irw + 1]])
                vals.append(True)

                added_edges.add((rw[irw], rw[irw + 1]))
                added_edges.add((rw[irw + 1], rw[irw]))

    adj = sp.csr_matrix((vals, (rows, cols)), shape=(count, count))
    return adj


def convert_neighborhood_edges_to_adj(roots, edges, new_idx, count):
    rows = []
    cols = []
    vals = []

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
    return adj