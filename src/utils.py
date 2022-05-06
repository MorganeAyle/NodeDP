import copy
import pdb
import time

import scipy
import numpy as np
import json
from sklearn.preprocessing import StandardScaler
import scipy.sparse as sp
import torch
from math import comb


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

    # ---- normalize feats ----
    out("Normalizing data...")
    train_nodes = np.array(list(set(adj_train.nonzero()[0])))
    train_feats = feats[train_nodes]
    scaler = StandardScaler()
    scaler.fit(train_feats)
    feats = scaler.transform(feats)
    # -------------------------

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
        for k,v in class_map.items():
            class_arr[k][v-offset] = 1
    out("Done loading data.")
    return adj_full, adj_train, feats, class_arr, role


def define_additional_args(num_subgraphs, num_par_samplers, out):
    assert num_subgraphs % num_par_samplers == 0
    num_subgraphs_per_sampler = num_subgraphs // num_par_samplers
    out(f"Number of subgraphs per sampler: {num_subgraphs_per_sampler}")
    return num_subgraphs_per_sampler


def adj_add_self_loops(adj):
    ret_adj = copy.deepcopy(adj).tolil()
    ret_adj.setdiag(1)
    return ret_adj.tocsr()


def adj_norm(adj, deg=None, sort_indices=True):
    diag_shape = (adj.shape[0], adj.shape[1])
    D = adj.sum(1).flatten().squeeze() if deg is None else deg
    norm_diag = sp.dia_matrix((1/D, 0), shape=diag_shape)
    adj_norm = norm_diag.dot(adj)
    if sort_indices:
        adj_norm.sort_indices()
    return adj_norm


def _coo_scipy2torch(adj):
    """
    convert a scipy sparse COO matrix to torch
    """
    values = adj.data
    indices = np.vstack((adj.row, adj.col))
    i = torch.LongTensor(indices)
    v = torch.FloatTensor(values)
    return torch.sparse.FloatTensor(i,v, torch.Size(adj.shape))


def compute_hypergeometric(N, d, m):
    return [comb(d, i) * comb(N - d, m - i) / comb(N, m) for i in range(d + 1)]