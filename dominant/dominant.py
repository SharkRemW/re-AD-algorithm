import torch.nn as nn
import torch.nn.functional as F
import torch
from torch_geometric.nn import GCNConv


from pygod.utils import load_data
from sklearn.metrics import roc_auc_score
import torch


from torch_geometric.utils import to_dense_adj

from pygod.detector import DOMINANT 

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
    
def loss_func(adj, A_hat, attrs, X_hat, alpha):
    # Attribute reconstruction loss
    diff_attribute = torch.pow(X_hat - attrs, 2)
    attribute_reconstruction_errors = torch.sqrt(torch.sum(diff_attribute, 1) + 1e-9)
    attribute_cost = torch.mean(attribute_reconstruction_errors)

    # structure reconstruction loss
    diff_structure = torch.pow(A_hat - adj, 2)
    structure_reconstruction_errors = torch.sqrt(torch.sum(diff_structure, 1) + 1e-9)
    structure_cost = torch.mean(structure_reconstruction_errors)


    cost =  alpha * attribute_reconstruction_errors + (1-alpha) * structure_reconstruction_errors

    return cost, structure_cost, attribute_cost

def train_dominant(args):
    data = load_data(args.dataset)
    print("data: ", data)

    # 生成邻接矩阵
    data.s = to_dense_adj(data.edge_index)[0]

    x = data.x
    s = data.s
    edge_index = data.edge_index

    model = Dominant(feat_size = x.shape[1], hidden_size = args.hidden_dim, drop_prob = args.dropout)

    if args.device == 'cuda':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        x = x.to(device)
        s = s.to(device)
        edge_index = edge_index.to(device)
        model = model.to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # train
    for epoch in range(args.epoch):
        model.train()
        s_, x_ = model(x, edge_index)
        loss, struct_loss, feat_loss = loss_func(s, s_, x, x_, args.alpha)
        loss = torch.mean(loss)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if epoch % 10 == 0 or epoch == args.epoch - 1:
            print("Epoch:", '%04d' % (epoch), "train_loss=", "{:.5f}".format(loss.item()), "train_struct_loss=", "{:.5f}".format(struct_loss.item()),"train_feat_loss=", "{:.5f}".format(feat_loss.item()))

    # evaluation
    model.eval()
    s_, x_ = model(x, edge_index)
    loss, struct_loss, feat_loss = loss_func(s, s_, x, x_, args.alpha)
    score = loss.detach().cpu().numpy()
    print('Auc: ', roc_auc_score(data.y.bool().numpy(), score))