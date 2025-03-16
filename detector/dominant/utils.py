import numpy as np
import scipy.sparse as sp
import torch
import scipy.io as sio
import random
from pygod.utils import load_data


def load_anomaly_detection_dataset(dataset, datadir='data'):
    # data = load_data(dataset)

    data_mat = sio.loadmat(f'{datadir}/{dataset}.mat')
    print("data_mat: ", data_mat)

    adj = data_mat['Network']
    adj = (adj + sp.eye(adj.shape[0])).toarray()
    feat = data_mat['Attributes'].toarray()
    truth = data_mat['Label'].flatten()
    
    return adj, feat, truth, normalize_adj(adj)



def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj + sp.eye(adj.shape[0]))
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return d_mat_inv_sqrt.dot(adj).dot(d_mat_inv_sqrt).tocoo().toarray()