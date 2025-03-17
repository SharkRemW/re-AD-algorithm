import torch.nn as nn
import torch.nn.functional as F
import torch
from torch_geometric.nn import GCNConv

from pygod.detector import CONAD


class GRL(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, hidden_num=1, head_num=1) -> None:
        super(GRL, self).__init__()
        self.shared_encoder = nn.ModuleList(
            GraphConv(
                in_dim if i==0 else hidden_dim,
                (out_dim if i == hidden_num - 1 else hidden_dim),
                activation=torch.sigmoid
            )
            for i in range(hidden_num)
        )
        self.attr_decoder = GraphConv(
            in_feats=out_dim,
            out_feats=in_dim,
            activation=torch.sigmoid,
        )
        self.struct_decoder = nn.Sequential(
            Reconstruct(),
            nn.Sigmoid()
        )
        self.dense = nn.Sequential(nn.Linear(out_dim, out_dim))

    def embed(self, g, h):
        for layer in self.shared_encoder:
            h = layer(g, h).view(h.shape[0], -1)
        # h = self.project(g, h).view(h.shape[0], -1)
        # return h.div(torch.norm(h, p=2, dim=1, keepdim=True))
        return self.dense(h)
    
    def reconstruct(self, g, h):
        struct_reconstructed = self.struct_decoder(h)
        x_hat = self.attr_decoder(g, h).view(h.shape[0], -1)
        return struct_reconstructed, x_hat

    def forward(self, g, h):
        # encode
        for layer in self.shared_encoder:
            h = layer(g, h).view(h.shape[0], -1)
        # decode adjacency matrix
        struct_reconstructed = self.struct_decoder(h)
        # decode feature matrix
        x_hat = self.attr_decoder(g, h).view(h.shape[0], -1)
        # return reconstructed matrices
        return struct_reconstructed, x_hat