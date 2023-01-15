import copy
import pdb

import scipy
import numpy as np
import json
from sklearn.preprocessing import StandardScaler
import scipy.sparse as sp
import torch
from math import comb
import random
import os

from src.constants import TRANSDUCTIVE_DATASETS


def configure_seeds(seed, device):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if device == 'cuda':
        torch.cuda.manual_seed_all(seed)


def create_save_path(save_model_dir, data_path, num_iterations, seed, sampler_args, training_args, model_args, out):
    model_name = f'data={data_path.split("/")[-1]}_iter={num_iterations}_seed={seed}'
    for key, val in sampler_args.items():
        key = key.split('_')[0]
        model_name += '_' + str(key) + '=' + str(val)
    for key, val in training_args.items():
        key = key.split('_')[0]
        model_name += '_' + str(key) + '=' + str(val)
    for key, val in model_args.items():
        key = key.split('_')[0]
        model_name += '_' + str(key) + '=' + str(val)
    save_model_path = os.path.join(save_model_dir, model_name)
    out("Model path: " + save_model_path)
    return save_model_path


def load_data(data_path, out):
    out("Loading data...")
    # Full adjacency
    adj_full = scipy.sparse.load_npz('./{}/adj_full.npz'.format(data_path)).astype(float)
    assert not len((adj_full != adj_full.T).data)
    adj_full = (adj_full - scipy.sparse.diags([adj_full.diagonal()], [0])).astype(bool)
    assert not adj_full.diagonal().any()

    # Training adjacency
    if data_path.split('/')[-1] in TRANSDUCTIVE_DATASETS:
        adj_train = copy.deepcopy(adj_full)
    else:
        adj_train = scipy.sparse.load_npz('./{}/adj_train.npz'.format(data_path)).astype(float)
        assert not len((adj_train != adj_train.T).data)
        adj_train = (adj_train - scipy.sparse.diags([adj_train.diagonal()], [0])).astype(bool)
        assert not adj_train.diagonal().any()

    role = json.load(open('./{}/role.json'.format(data_path)))
    feats = np.load('./{}/feats.npy'.format(data_path))
    class_map = json.load(open('./{}/class_map.json'.format(data_path)))
    class_map = {int(k): v for k, v in class_map.items()}
    assert len(class_map) == feats.shape[0]

    # normalize features
    out("Normalizing data...")
    train_nodes = np.array(role['tr'])
    train_feats = feats[train_nodes]
    scaler = StandardScaler()
    scaler.fit(train_feats)
    feats = scaler.transform(feats)

    num_vertices = adj_full.shape[0]
    if isinstance(list(class_map.values())[0], list):
        num_classes = len(list(class_map.values())[0])
        class_arr = np.zeros((num_vertices, num_classes))
        for k, v in class_map.items():
            class_arr[k] = v
    else:
        num_classes = max(class_map.values()) - min(class_map.values()) + 1
        class_arr = np.zeros((num_vertices, num_classes))
        offset = min(class_map.values())
        for k, v in class_map.items():
            class_arr[k][v - offset] = 1
    out("Done loading data.")
    return adj_full, adj_train, feats, class_arr, role


def adj_add_self_loops(adj):
    ret_adj = copy.deepcopy(adj).tolil()
    ret_adj.setdiag(1)
    return ret_adj.tocsr()


def adj_norm(adj, deg=None, sort_indices=True):
    diag_shape = (adj.shape[0], adj.shape[1])
    D = adj.sum(1).flatten().squeeze() if deg is None else deg
    # D[D == 0.] = 1.
    # norm_diag = sp.dia_matrix((1 / D, 0), shape=diag_shape)
    norm_diag = sp.dia_matrix((1 / np.sqrt(D), 0), shape=diag_shape)
    # adj_norm = norm_diag.dot(adj)
    normalized_adj = norm_diag.dot(adj).dot(norm_diag)
    if sort_indices:
        normalized_adj.sort_indices()
    return normalized_adj


def _coo_scipy2torch(adj):
    """
    convert a scipy sparse COO matrix to torch
    """
    values = adj.data
    indices = np.vstack((adj.row, adj.col))
    i = torch.LongTensor(indices)
    v = torch.FloatTensor(values)
    return torch.sparse.FloatTensor(i, v, torch.Size(adj.shape))


def compute_hypergeometric(N, d, m):
    return [comb(d, i) * comb(N - d, m - i) / comb(N, m) for i in range(d + 1)]


def binary_search(arr, low, high, x):
    # Check base case
    if high >= low:

        mid = (high + low) // 2

        # If element is present at the middle itself
        if arr[mid] == x:
            return mid

        # If element is smaller than mid, then it can only
        # be present in left subarray
        elif arr[mid] > x:
            return binary_search(arr, low, mid - 1, x)

        # Else the element can only be present in right subarray
        else:
            return binary_search(arr, mid + 1, high, x)

    else:
        # Element is not present in the array
        return -1


def bound_adj_degree(adj, max_degree):
    new_row = []
    new_col = []
    new_data = []
    new_neigh = {v: [] for v in range(adj.shape[0])}

    nodes = list(range(adj.shape[0]))
    random.shuffle(nodes)
    for v in nodes:
        neigh_len = adj.indptr[v + 1] - adj.indptr[v]
        existing_neigh = new_neigh[v]
        if len(existing_neigh) >= max_degree:
            continue
        else:
            neigh_indices = list(range(neigh_len))
            random.shuffle(neigh_indices)
            for i in neigh_indices:
                u = adj.indices[adj.indptr[v] + i]
                if len(new_neigh[u]) >= max_degree:
                    continue
                if len(new_neigh[v]) >= max_degree:
                    break
                new_data.append(True)
                new_row.append(v)
                new_col.append(u)
                new_data.append(True)
                new_row.append(u)
                new_col.append(v)
                new_neigh[v].append(u)
                new_neigh[u].append(v)
    new_adj = sp.csr_matrix((new_data, (new_row, new_col)), shape=adj.shape)

    return new_adj


def sample_rws(G, nodes, depth):
    rws = {}
    nodes = np.array(nodes)
    sampled_nodes = set()
    sampled_roots = []

    while len(nodes) != 0:
        iroot = random.randint(0, len(nodes)-1)
        root = nodes[iroot]
        node = copy.deepcopy(root)
        index = np.argwhere(nodes == node)
        nodes = np.delete(nodes, index)
        sampled_nodes.add(node)
        sampled_roots.append(node)

        rw = [node]
        for _ in range(depth):
            valid_neighbors = [v for v in G.neighbors(node) if v not in sampled_nodes]
            if len(valid_neighbors):
                node = random.choice(valid_neighbors)
                rw.append(node)
                if node not in sampled_nodes:
                    sampled_nodes.add(node)
                index = np.argwhere(nodes == node)
                nodes = np.delete(nodes, index)
            else:
                break
        rws[root] = rw

    return rws, sampled_roots


def sample_rws_w_restarts(G, nodes, depth, restarts):
    rws = {}
    edges = {}
    nodes = np.array(nodes)
    sampled_nodes = set()
    sampled_roots = []

    while len(nodes) != 0:
        iroot = random.randint(0, len(nodes) - 1)
        root = nodes[iroot]
        index = np.argwhere(nodes == root)
        nodes = np.delete(nodes, index)
        sampled_nodes.add(root)
        sampled_roots.append(root)

        rw = [root]
        redges = []
        for _ in range(restarts):
            node = copy.deepcopy(root)
            for _ in range(depth):
                valid_neighbors = [v for v in G.neighbors(node) if v not in sampled_nodes]
                if len(valid_neighbors):
                    neighbor = random.choice(valid_neighbors)
                    rw.append(neighbor)
                    redges.append((node, neighbor))
                    redges.append((neighbor, node))
                    if neighbor not in sampled_nodes:
                        sampled_nodes.add(neighbor)
                    index = np.argwhere(nodes == neighbor)
                    nodes = np.delete(nodes, index)
                    node = copy.deepcopy(neighbor)
                else:
                    break
            rws[root] = rw
            edges[root] = redges

    return rws, edges, sampled_roots


class DFS:
    def __init__(self, nodes, depth, graph):
        self.nodes = nodes
        self.depth = depth
        self.graph = graph

        self.neighborhoods = {}
        self.neighborhoods_edges = {}

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

    def sample_all_neighborhoods(self):
        for root in self.nodes:
            node = copy.copy(root)
            self.neighborhood = []
            self.edges = set()
            self.depth_first_search(node, self.depth, self.graph)
            neighborhood, neighborhood_edges = self.neighborhood, self.edges
            self.neighborhoods[root] = neighborhood
            self.neighborhoods_edges[root] = neighborhood_edges
