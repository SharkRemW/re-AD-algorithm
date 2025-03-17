from pygod.utils import load_data
from sklearn.metrics import roc_auc_score
import torch


from torch_geometric.utils import to_dense_adj
from dominant import Dominant

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