import copy
import scipy
import numpy as np
import json
from sklearn.preprocessing import StandardScaler
import scipy.sparse as sp
import torch
from math import comb
import random
import networkx as nx


def configure_seeds(seed, device):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if device == 'cuda':
        torch.cuda.manual_seed_all(seed)


def load_data(data_path, out):
    out("Loading data...")
    adj_full = scipy.sparse.load_npz('./{}/adj_full.npz'.format(data_path)).astype(float)
    assert not len((adj_full != adj_full.T).data)
    adj_full = (adj_full - scipy.sparse.diags([adj_full.diagonal()], [0])).astype(bool)
    assert not adj_full.diagonal().any()

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
    train_nodes = np.array(list(set(adj_train.nonzero()[0])))
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


def sample_rws(adj, nodes, depth):
    rws = {}
    G = nx.from_scipy_sparse_matrix(adj)
    tot_num_nodes = len(nodes)
    nodes = list(nodes)
    random.shuffle(nodes)
    sampled_nodes = set()
    sampled_roots = []

    while len(sampled_nodes) != tot_num_nodes:
        root = nodes.pop()
        while root in sampled_nodes:
            root = nodes.pop()
        sampled_roots.append(root)
        sampled_nodes.add(root)
        rw = [root]
        node = copy.deepcopy(root)
        for _ in range(depth):
            valid_neighbors = [v for v in G.neighbors(node) if v not in sampled_nodes]
            if len(valid_neighbors):
                neighbor = random.choice(valid_neighbors)
                rw.append(neighbor)
                sampled_nodes.add(neighbor)
                node = copy.deepcopy(neighbor)
            else:
                break
        rws[root] = rw

    return rws, sampled_roots


def sample_rws_w_restarts(adj, nodes, depth, restarts):
    rws = {}
    edges = {}
    G = nx.from_scipy_sparse_matrix(adj)
    tot_num_nodes = len(nodes)
    nodes = list(nodes)
    random.shuffle(nodes)
    sampled_nodes = set()
    sampled_roots = []

    while len(sampled_nodes) != tot_num_nodes:
        root = nodes.pop()
        while root in sampled_nodes:
            root = nodes.pop()
        sampled_roots.append(root)
        sampled_nodes.add(root)
        rw = [root]
        redges = []
        for _ in range(restarts):
            node = copy.deepcopy(root)
            for _ in range(depth):
                valid_neighbors = [v for v in G.neighbors(node) if v not in sampled_nodes]
                if len(valid_neighbors):
                    neighbor = random.choice(valid_neighbors)
                    rw.append(neighbor)
                    sampled_nodes.add(neighbor)
                    redges.append((node, neighbor))
                    redges.append((neighbor, node))
                    node = copy.deepcopy(neighbor)
                else:
                    break
        rws[root] = rw
        edges[root] = redges

    return rws, edges, sampled_roots
