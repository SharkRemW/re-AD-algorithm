import numpy as np
import scipy.sparse as sp
import torch
import scipy.io as sio
import random
from pygod.utils import load_data


def load_anomaly_detection_dataset(dataset, datadir='data'):
    data = load_data("inj_cora")
    print("data: ", data)
    print("data.edge_index: ", data.edge_index)
    print("data.x: ", data.x)
    print("data.y: ", data.y)

    feat = data.x
    label = data.y

    num_nodes = feat.shape[0]

    edge_index = data.edge_index
    adj = torch.zeros((num_nodes, num_nodes), dtype=edge_index.dtype)
    adj[edge_index[0], edge_index[1]] = 1

    
    # data_mat = sio.loadmat(f'{datadir}/{dataset}.mat')
    # # print("data_mat: ", data_mat)

    # adj = data_mat['Network']
    # adj = (adj + sp.eye(adj.shape[0])).toarray()
    # feat = data_mat['Attributes'].toarray()
    # label = data_mat['Label'].flatten()
    
    return adj, feat, label, normalize_adj(adj.numpy())



def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj + sp.eye(adj.shape[0]))
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return d_mat_inv_sqrt.dot(adj).dot(d_mat_inv_sqrt).tocoo().toarray()