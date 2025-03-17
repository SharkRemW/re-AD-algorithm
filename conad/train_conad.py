from pygod.utils import load_data
from sklearn.metrics import roc_auc_score
import torch

from torch_geometric.utils import to_dense_adj

from conad import Conad
from pygod.detector import CONAD

from copy import deepcopy

from torch_geometric.utils import dense_to_sparse

from tqdm import tqdm

def graph_augmentation(adj, feat, rate=.1, clique_size=30, sourround=50, scale_factor=10):
    adj_aug, feat_aug = deepcopy(adj), deepcopy(feat)
    label_aug = torch.zeros(adj.shape[0])

    assert(adj_aug.shape[0]==feat_aug.shape[0])

    num_nodes = adj_aug.shape[0]
    import torch
from copy import deepcopy

def graph_augmentation(adj, feat, rate=0.1, clique_size=30, surround=50, scale_factor=10):
    # Deep copy the input adjacency and feature matrices
    adj_aug, feat_aug = deepcopy(adj), deepcopy(feat)
    label_aug = torch.zeros(adj.shape[0], dtype=torch.int32)  # Initialize labels

    assert adj_aug.shape[0] == feat_aug.shape[0], "Adjacency and feature matrices must have the same number of nodes."
    
    num_nodes = adj_aug.shape[0]  # Number of nodes in the graph

    for i in range(num_nodes):
        prob = torch.rand(1).item()  # Random probability for anomaly injection
        if prob > rate:
            continue  # Skip if no anomaly is injected
        
        label_aug[i] = 1  # Mark node as anomalous
        
        one_fourth = torch.randint(0, 4, (1,)).item()  # Randomly choose one of four anomaly types
        
        if one_fourth == 0:
            # Add clique: Connect the node to a random subset of other nodes
            new_neighbors = torch.randperm(num_nodes)[:clique_size]  # Randomly select clique_size nodes
            adj_aug[new_neighbors, i] = 1
            adj_aug[i, new_neighbors] = 1

        elif one_fourth == 1:
            # Drop all connections: Remove all edges connected to the node
            neighbors = torch.nonzero(adj_aug[i]).squeeze()  # Find neighbors of node i
            if neighbors.numel() == 0:  # Skip if no neighbors exist
                continue
            adj_aug[i, neighbors] = 0
            adj_aug[neighbors, i] = 0

        elif one_fourth == 2:
            # Deviated attributes: Replace node's feature with the most distant candidate
            candidates = torch.randperm(num_nodes)[:surround]  # Randomly select surround nodes
            max_dev, max_idx = 0, i
            for c in candidates:
                dev = torch.sum(torch.square(feat_aug[i] - feat_aug[c]))
                if dev > max_dev:
                    max_dev = dev
                    max_idx = c
            feat_aug[i] = feat_aug[max_idx]

        else:
            # Scale attributes: Multiply or divide the node's features by scale_factor
            prob_scale = torch.rand(1).item()
            if prob_scale > 0.5:
                feat_aug[i] *= scale_factor
            else:
                feat_aug[i] /= scale_factor
    edge_index_aug = dense_to_sparse(adj_aug)[0]
    # print("edge_index_aug: ", edge_index_aug)
    return feat_aug, edge_index_aug, label_aug

def loss_func(s, s_, x, x_, alpha):
    # Attribute reconstruction loss
    diff_attribute = torch.pow(x - x_, 2)
    attribute_reconstruction_errors = torch.sqrt(torch.sum(diff_attribute, 1) + 1e-9)
    attribute_cost = torch.mean(attribute_reconstruction_errors)

    # structure reconstruction loss
    diff_structure = torch.pow(s - s_, 2)
    structure_reconstruction_errors = torch.sqrt(torch.sum(diff_structure, 1) + 1e-9)
    structure_cost = torch.mean(structure_reconstruction_errors)


    cost =  alpha * attribute_reconstruction_errors + (1-alpha) * structure_reconstruction_errors

    return cost, structure_cost, attribute_cost

def train_conad(args):
    data = load_data(args.dataset)
    print("data: ", data)

    # 生成邻接矩阵
    data.s = to_dense_adj(data.edge_index)[0]

    x = data.x
    s = data.s
    edge_index = data.edge_index

    model = Conad(feat_size = x.shape[1], hidden_size = args.hidden_dim, drop_prob = args.dropout)

    # criterion = torch.nn.TripletMarginLoss()
    margin = 0.5
    margin_loss_func = lambda z, z_hat, l: torch.square(z - z_hat) * (l==0) - l * torch.square(z - z_hat) + margin
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    x = x.to(device)
    s = s.to(device)
    edge_index = edge_index.to(device)
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # graph augmentation
    x_aug, edge_index_aug, label_aug = graph_augmentation(s, x, rate=.2, clique_size=15)
    label_aug = label_aug.unsqueeze(-1).float()
    x_aug = x_aug.to(device)
    edge_index_aug = edge_index_aug.to(device)
    label_aug = label_aug.to(device)

    epoch = args.epoch
    # train encoder with supervised contrastive learning
    for i in tqdm(range(epoch)):
        model.train()
        
        z = model.embed(x, edge_index)
        z_aug = model.embed(x_aug, edge_index_aug)
        s_, x_ = model(x, edge_index)

        margin_loss = margin_loss_func(z, z_aug, label_aug)
        margin_loss = margin_loss.mean()

        recon_loss, struct_loss, feat_loss = loss_func(s, s_, x, x_, alpha=args.alpha)
        recon_loss = recon_loss.mean()

        loss = 0.5 * margin_loss + 0.5 * recon_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
        # evaluate
        if i % 10 == 0 or i == args.epoch - 1:
            with torch.no_grad():
                model.eval()

                s_, x_ = model(x, edge_index)
                recon_loss, struct_loss, feat_loss = loss_func(s, s_, x, x_, alpha=.7)

                score = recon_loss.detach().cpu().numpy()

                print('AUC: %.4f' % roc_auc_score(data.y.bool().numpy(), score))
                # for k in [50, 100, 200, 300]:
                #     print('Precision@%d: %.4f' % (k, precision_at_k(label, score, k)))