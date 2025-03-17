import torch.nn as nn
import torch.nn.functional as F
import torch
from torch_geometric.nn import GCNConv


class Encoder(nn.Module):
    def __init__(self, nfeat, nhid, drop_prob):
        super().__init__()

        self.gc1 = GCNConv(nfeat, nhid)
        self.gc2 = GCNConv(nhid, nhid)
        self.drop_prob = drop_prob

    def forward(self, x, edge_index):
        x = F.relu(self.gc1(x, edge_index))
        x = F.dropout(x, self.drop_prob, training=self.training)
        x = F.relu(self.gc2(x, edge_index))

        return x

class Attribute_Decoder(nn.Module):
    def __init__(self, nfeat, nhid, drop_prob):
        super().__init__()

        self.gc1 = GCNConv(nhid, nhid)
        self.gc2 = GCNConv(nhid, nfeat)
        self.drop_prob = drop_prob

    def forward(self, x, edge_index):
        x = F.relu(self.gc1(x, edge_index))
        x = F.dropout(x, self.drop_prob, training=self.training)
        x = F.relu(self.gc2(x, edge_index))

        return x

class Structure_Decoder(nn.Module):
    def __init__(self, nhid, drop_prob):
        super().__init__()

        self.gc1 = GCNConv(nhid, nhid)
        self.drop_prob = drop_prob

    def forward(self, x, edge_index):
        x = F.relu(self.gc1(x, edge_index))
        x = F.dropout(x, self.drop_prob, training=self.training)
        x = x @ x.T

        return x

class Dominant(nn.Module):
    def __init__(self, feat_size, hidden_size, drop_prob):
        super().__init__()
        
        self.shared_encoder = Encoder(feat_size, hidden_size, drop_prob)
        self.attr_decoder = Attribute_Decoder(feat_size, hidden_size, drop_prob)
        self.struct_decoder = Structure_Decoder(hidden_size, drop_prob)
    
    def forward(self, x, edge_index):

        # encode
        x = self.shared_encoder(x, edge_index)
        # decode feature matrix
        X_hat = self.attr_decoder(x, edge_index)
        # decode adjacency matrix
        A_hat = self.struct_decoder(x, edge_index)
        # return reconstructed matrices
        return A_hat, X_hat
    
