from pygod.utils import load_data
from sklearn.metrics import roc_auc_score
import torch


from torch_geometric.utils import to_dense_adj
from adagad import ADAGAD

def train_adagad(args):
    data = load_data(args.dataset)

    model = ADAGAD(feat_size = data.x.shape[1], hidden_size = args.num_hidden, 
                   lr=args.lr,
                   epoch=args.max_epoch, device=args.device,
                   aggr_f=args.aggr_f)
    
    print("Auc: ", model(data))