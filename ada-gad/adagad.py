import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch
from torch_geometric.nn import GCNConv
from torch_geometric.nn.models import GCN
from torch_geometric.utils import to_dense_adj, dense_to_sparse
from torch_geometric.utils import dropout_edge, add_self_loops, sort_edge_index, degree
from torch_geometric.utils.num_nodes import maybe_num_nodes
from sklearn.utils.validation import check_is_fitted

from torch_geometric.loader import NeighborLoader

from sklearn.metrics import roc_auc_score

from typing import Optional, Tuple
from torch import Tensor
try:
    import torch_cluster  # noqa
    random_walk = torch.ops.torch_cluster.random_walk
except ImportError:
    random_walk = None

def dropout_subgraph(edge_index: Tensor, p: float = 0.2, walks_per_node: int = 1,
                 walk_length: int = 3, num_nodes: Optional[int] = None,
                 is_sorted: bool = False,
                 training: bool = True,return_subgraph:bool=True) -> Tuple[Tensor, Tensor]:
    r"""Drops edges from the adjacency matrix :obj:`edge_index`
    based on random walks. The source nodes to start random walks from are
    sampled from :obj:`edge_index` with probability :obj:`p`, following
    a Bernoulli distribution.

    The method returns (1) the retained :obj:`edge_index`, (2) the edge mask
    indicating which edges were retained.

    Args:
        edge_index (LongTensor): The edge indices.
        p (float, optional): Sample probability. (default: :obj:`0.2`)
        walks_per_node (int, optional): The number of walks per node, same as
            :class:`~torch_geometric.nn.models.Node2Vec`. (default: :obj:`1`)
        walk_length (int, optional): The walk length, same as
            :class:`~torch_geometric.nn.models.Node2Vec`. (default: :obj:`3`)
        num_nodes (int, optional): The number of nodes, *i.e.*
            :obj:`max_val + 1` of :attr:`edge_index`. (default: :obj:`None`)
        is_sorted (bool, optional): If set to :obj:`True`, will expect
            :obj:`edge_index` to be already sorted row-wise.
            (default: :obj:`False`)
        training (bool, optional): If set to :obj:`False`, this operation is a
            no-op. (default: :obj:`True`)

    :rtype: (:class:`LongTensor`, :class:`BoolTensor`)

    Example:

        >>> edge_index = torch.tensor([[0, 1, 1, 2, 2, 3],
        ...                            [1, 0, 2, 1, 3, 2]])
        >>> edge_index, edge_mask = dropout_path(edge_index)
        >>> edge_index
        tensor([[1, 2],
                [2, 3]])
        >>> edge_mask # masks indicating which edges are retained
        tensor([False, False,  True, False,  True, False])
    """

    if p < 0. or p > 1.:
        raise ValueError(f'Sample probability has to be between 0 and 1 '
                         f'(got {p}')

    num_edges = edge_index.size(1)
    edge_mask = edge_index.new_ones(num_edges, dtype=torch.bool)
    if not training or p == 0.0:
        return edge_index, edge_mask

    if random_walk is None:
        raise ImportError('`dropout_path` requires `torch-cluster`.')

    num_nodes = maybe_num_nodes(edge_index, num_nodes)
    # print("edge_index: ", edge_index.shape, edge_index)
    # print("num_nodes: ", num_nodes)
    edge_orders = None
    ori_edge_index = edge_index
    if not is_sorted:
        edge_orders = torch.arange(num_edges, device=edge_index.device)
        edge_index, edge_orders = sort_edge_index(edge_index, edge_orders,
                                                  num_nodes=num_nodes)

    row, col = edge_index
    sample_mask = torch.rand(row.size(0), device=edge_index.device) <= p
    start = row[sample_mask].repeat(walks_per_node)

    deg = degree(row, num_nodes=num_nodes)
    rowptr = row.new_zeros(num_nodes + 1)
    torch.cumsum(deg, 0, out=rowptr[1:])

    n_id, e_id = random_walk(rowptr, col, start, walk_length, 1.0, 1.0)

    e_id = e_id[e_id != -1].view(-1)  # filter illegal edges
    if edge_orders is not None:
        e_id = edge_orders[e_id]
    edge_mask[e_id] = False
    edge_index = ori_edge_index[:, edge_mask]
    if return_subgraph:
        subgraph_mask=n_id_list_to_edge_index(n_id,num_nodes)

    else:
        subgraph_mask=torch.ones((num_nodes,num_nodes))
    return edge_index, edge_mask,subgraph_mask
    
def n_id_list_to_edge_index(n_id_list, num_node): 
    edge_index = torch.zeros((num_node, num_node)).to(n_id_list.device) 
    for n_id in n_id_list: 
        unique_ids = torch.unique(n_id) 
        mask = (unique_ids.view(-1, 1) != unique_ids.view(1, -1)).to(n_id_list.device) 
        edge_index[unique_ids.view(-1, 1), unique_ids.view(1, -1)] += mask 
    edge_index[edge_index!=0]=1
    return edge_index

def compute_E_high(adj_matrix, feat_matrix):
    adj_tensor = torch.tensor(adj_matrix, dtype=torch.float32)
    feat_tensor = feat_matrix.clone().detach().to(dtype=torch.float32)

    deg_tensor = torch.sum(adj_tensor, dim=1)
    deg_matrix = torch.diag(deg_tensor)

    laplacian_tensor = deg_matrix - adj_tensor
    numerator = torch.matmul(torch.matmul(feat_tensor.T, laplacian_tensor), feat_tensor)
    denominator = torch.matmul(feat_tensor.T, feat_tensor)
    
    S_high = torch.sum(numerator) / torch.sum(denominator)

    return S_high.item()

def compute_G_ano(adj_matrix, feat_matrix):
    a_high = compute_E_high(adj_matrix, feat_matrix)
    deg_matrix = torch.diag(torch.sum(torch.tensor(adj_matrix, dtype=torch.float32), dim=1))
    s_high = compute_E_high(adj_matrix, deg_matrix)

    return a_high, s_high


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
    def __init__(self, nfeat, nhid, drop_prob, aggr='add'):
        super().__init__()

        self.gc1 = GCNConv(nhid, nhid, aggr=aggr)
        self.gc2 = GCNConv(nhid, nfeat, aggr=aggr)
        self.drop_prob = drop_prob

    def forward(self, x, edge_index):
        x = F.relu(self.gc1(x, edge_index))
        x = F.dropout(x, self.drop_prob, training=self.training)
        x = F.relu(self.gc2(x, edge_index))

        return x

class Structure_Decoder(nn.Module):
    def __init__(self, nhid, drop_prob, aggr='add'):
        super().__init__()

        self.gc1 = GCNConv(nhid, nhid, aggr)
        self.drop_prob = drop_prob

    def forward(self, x, edge_index):
        x = F.relu(self.gc1(x, edge_index))
        x = F.dropout(x, self.drop_prob, training=self.training)
        x = x @ x.T

        return x
    
class PreModel(nn.Module):
    def __init__(self, feat_size, hidden_size, drop_prob):
        super().__init__()
    
        self.shared_encoder = Encoder(feat_size, hidden_size, drop_prob)
        self.attr_decoder = Attribute_Decoder(feat_size, hidden_size, drop_prob)
        self.struct_decoder = Structure_Decoder(hidden_size, drop_prob)

    def embed(self, x, edge_index):
        return self.shared_encoder(x, edge_index)
    
    def attr_recon(self, x, edge_index):
        return self.attr_decoder(x, edge_index)
    
    def struct_recon(self, x, edge_index):
        return self.struct_decoder(x, edge_index)

    def forward(self, x, edge_index):
        # encode
        x = self.shared_encoder(x, edge_index)
        # decode feature matrix
        X_hat = self.attr_decoder(x, edge_index)
        # decode adjacency matrix
        A_hat = self.struct_decoder(x, edge_index)
        # return reconstructed matrices
        return A_hat, X_hat

class ADAGAD(nn.Module):
    def __init__(self, feat_size, hidden_size, 
                 drop_prob=0.3,
                 lr=5e-3,
                 epoch=20,
                 device='cuda',
                 mask_rate1=0.01,
                 select_gano_num=30,
                 replace_rate=0.,
                 drop_edge_rate=0.01,
                 drop_path_rate=0.01,
                 predict_all_edge1=0,
                 predict_all_edge2=0,
                 predict_all_edge3=0,
                 drop_path_length=3,
                 walks_per_node=3,
                 alpha=None,
                 loss_weight=0.,
                 aggr_f='add'):
        super().__init__()
        # initialize args list
        self.lr = lr
        self.epoch = epoch
        self.device = device
        self.select_gano_num = select_gano_num
        self.mask_rate1 = mask_rate1
        self._replace_rate = replace_rate
        self.drop_edge_rate = drop_edge_rate
        self.drop_path_rate = drop_path_rate
        self.predict_all_edge1 = predict_all_edge1
        self.predict_all_edge2 = predict_all_edge2
        self.predict_all_edge3 = predict_all_edge3
        self.drop_path_length = drop_path_length
        self.walks_per_node = walks_per_node

        self.alpha = alpha
        self.loss_weight = loss_weight

        # premodels
        self.node_premodel = PreModel(feat_size, hidden_size, drop_prob)
        self.edge_premodel = PreModel(feat_size, hidden_size, drop_prob)
        self.subgraph_premodel = PreModel(feat_size, hidden_size, drop_prob)

        # remodel
        self.remodel = ReModel(hid_dim=hidden_size, drop_prob=drop_prob,alpha=self.alpha, lr=self.lr, epoch=self.epoch, loss_weight=self.loss_weight, device=self.device,
                               aggr_f=aggr_f)

    def node_denoise(self, x, mask_rate=0.3):
        num_nodes = x.shape[0]
        perm = torch.randperm(num_nodes, device=x.device)

        # random masking
        num_mask_nodes = int(mask_rate * num_nodes)
        mask_nodes = perm[: num_mask_nodes]

        keep_nodes = perm[num_mask_nodes: ]

        if self._replace_rate > 0 and int(self._replace_rate * num_mask_nodes)>0:
            num_noise_nodes = int(self._replace_rate * num_mask_nodes)
            perm_mask = torch.randperm(num_mask_nodes, device=x.device)
            token_nodes = mask_nodes[perm_mask[: int(self._mask_token_rate * num_mask_nodes)]]
            noise_nodes = mask_nodes[perm_mask[-int(self._replace_rate * num_mask_nodes):]]
            noise_to_be_chosen = torch.randperm(num_nodes, device=x.device)[:num_noise_nodes]

            out_x = x.clone()
            out_x[token_nodes] = 0.0
            out_x[noise_nodes] = x[noise_to_be_chosen]
        else:
            out_x = x.clone()
            token_nodes = mask_nodes
            out_x[mask_nodes] = 0.0

        # out_x[token_nodes] += self.enc_mask_token

        return out_x, (mask_nodes, keep_nodes)
    
    def intersection_edge(self,edge_index_1, edge_index_2,max_num_nodes): 
        s1=to_dense_adj(edge_index_1,max_num_nodes=max_num_nodes)[0]
        s2=to_dense_adj(edge_index_2,max_num_nodes=max_num_nodes)[0]
        intersection_s=torch.min(s1,s2)

        intersection_edge_index,_=dense_to_sparse(intersection_s)
        # print('intersection_edge_index',intersection_edge_index)

        return intersection_edge_index,intersection_s
    
    def mask_attr_prediction(self, model, x, edge_index, criterion, mask_rate=0., drop_edge_rate=0., drop_path_rate=0., predict_all_edge=0):
        num_nodes=x.size()[0]

        # mask edge to reduce struct uncertainty
        dence_edge_index=to_dense_adj(edge_index,max_num_nodes=num_nodes)[0]
        # use_e_high=False

        # Node-level denoising pretraining
        if self.select_gano_num:

            G_ano_init=float('inf')
            for j in range(self.select_gano_num):
                use_x, (mask_nodes, keep_nodes) = self.node_denoise(x, mask_rate)
                a_ano,s_ano = compute_G_ano(dence_edge_index,use_x)
                G_ano = a_ano + s_ano

                if G_ano < G_ano_init:
                    use_x_select = use_x
                    mask_nodes_select = mask_nodes
                    keep_nodes_select = keep_nodes
                    G_ano_init = G_ano
          #print('final G_ano',G_ano_init)
            use_x = use_x_select
            mask_nodes = mask_nodes_select
            keep_nodes = keep_nodes_select
        else:
            use_x, (mask_nodes, keep_nodes) = self.node_denoise(x, mask_rate)

        use_x=use_x.to(torch.float32)
        # print('use_x',use_x.size())

        # Edge-level denoising pretraining
        if drop_edge_rate > 0:
            # use_mask_edge_edge_index, masked_edge_edges = dropout_edge(edge_index, _drop_edge_rate)
            
            if self.select_gano_num:
                G_ano_init=float('inf')
                for j in range(self.select_gano_num):
                    use_mask_edge_edge_index, masked_edge_edges = dropout_edge(edge_index, drop_edge_rate)
                    # to_dense_adj(edge_index)[0]
                    a_ano,s_ano=compute_G_ano(to_dense_adj(use_mask_edge_edge_index,max_num_nodes=num_nodes)[0],use_x)
                    G_ano=a_ano + s_ano
                  #print('G_ano',G_ano)    
                    if G_ano<G_ano_init:
                        use_mask_edge_edge_index_select=use_mask_edge_edge_index
                        masked_edge_edges_select=masked_edge_edges
                        G_ano_init=G_ano
              #print('final G_ano',G_ano_init)
                use_mask_edge_edge_index=use_mask_edge_edge_index_select
                masked_edge_edges=masked_edge_edges_select
            else:
                use_mask_edge_edge_index, masked_edge_edges = dropout_edge(edge_index, drop_edge_rate)

            use_mask_edge_edge_index = add_self_loops(use_mask_edge_edge_index)[0]
        else:
            use_mask_edge_edge_index = edge_index

        # mask path for struct reconstruction
        if drop_path_rate > 0:
            if self.select_gano_num:
                G_ano_init=float('inf')
                for j in range(self.select_gano_num):
                    use_mask_path_edge_index, masked_path_edges,_= dropout_subgraph(edge_index, p=drop_path_rate,walk_length=self.drop_path_length,walks_per_node=self.walks_per_node,return_subgraph=False)
                    a_ano,s_ano=compute_G_ano(to_dense_adj(use_mask_path_edge_index,max_num_nodes=num_nodes)[0],use_x)
                    G_ano = a_ano + s_ano 

                    if G_ano<G_ano_init:
                        use_mask_path_edge_index_select=use_mask_path_edge_index
                        masked_path_edges_select=masked_path_edges
                        G_ano_init=G_ano

                use_mask_path_edge_index=use_mask_path_edge_index_select
                masked_path_edges=masked_path_edges_select
            else:
                use_mask_path_edge_index, masked_path_edges,_= dropout_subgraph(edge_index, p=drop_path_rate,walk_length=self.drop_path_length,walks_per_node=self.walks_per_node,return_subgraph=False)
            
            use_mask_path_edge_index = add_self_loops(use_mask_path_edge_index)[0]
        else:
            use_mask_path_edge_index = edge_index
            
        # mask edge and path
        use_edge_index,use_s=self.intersection_edge(use_mask_edge_edge_index, use_mask_path_edge_index, num_nodes)

        
        # ---- attribute and edge reconstruction ----
        rep = model.embed(use_x, use_edge_index)
        # rep = self.encoder_to_decoder(enc_rep)

        loss = 0
        # ---- attribute reconstruction ----
        if mask_rate > 0:
            attr_recon = model.attr_recon(rep, use_edge_index)
            x_init = x
            x_rec = attr_recon
            loss += criterion(x_rec, x_init)
        # ---- edge reconstruction ----
        if drop_edge_rate > 0 or drop_path_rate > 0 :
            struct_recon = model.struct_recon(rep, use_edge_index)

            s_init = to_dense_adj(edge_index, max_num_nodes=num_nodes)[0]

            if predict_all_edge: 
                if  self.neg_s==None:
                    neg_rate=edge_index.size()[1]/(s_init.size()[0]**2) * predict_all_edge
                    self.neg_s=torch.rand(s_init.size()) <neg_rate
                   
                s_rec = torch.where((((use_s==0) & (s_init==1))|(self.neg_s).to(use_s.device)),struct_recon,s_init)
            else:
                s_rec = torch.where((use_s==0) & (s_init==1),struct_recon,s_init)

            loss += criterion(s_rec, s_init)

        
        return loss  

    def pretrain_one(self, model, x, edge_index, mask_rate=0., drop_edge_rate=0., drop_path_rate=0., predict_all_edge=0):
        s = to_dense_adj(edge_index)[0]

        criterion = nn.MSELoss()

        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)

        for i in range(self.epoch):
            model.train()
            loss = self.mask_attr_prediction(model, x, edge_index, criterion,
                                             mask_rate=mask_rate, drop_edge_rate=drop_edge_rate, drop_path_rate=drop_path_rate,
                                             predict_all_edge=predict_all_edge)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    def pretrain(self, graph):
        device = self.device

        x = graph.x
        x = x.to(device)

        edge_index = graph.edge_index
        edge_index = edge_index.to(device)
        
        print("=====Start pretraining node_premodel=====")
        self.node_premodel = self.node_premodel.to(device)
        self.pretrain_one(self.node_premodel, x, edge_index, self.mask_rate1)
        self.node_premodel = self.node_premodel.cpu()
        
        print("=====Start pretraining edge_premodel=====")
        self.edge_premodel = self.edge_premodel.to(device)
        self.pretrain_one(self.edge_premodel, x, edge_index, drop_edge_rate=self.drop_edge_rate, predict_all_edge=self.predict_all_edge2)
        self.edge_premodel = self.edge_premodel.cpu()
        
        print("=====Start pretraining subgraph_premodel=====")
        self.subgraph_premodel = self.subgraph_premodel.to(device)
        self.pretrain_one(self.subgraph_premodel, x, edge_index, drop_path_rate=self.drop_path_rate, predict_all_edge=self.predict_all_edge3)
        self.subgraph_premodel = self.subgraph_premodel.cpu()
        

    def retrain(self, graph):
        self.node_premodel.eval()
        self.edge_premodel.eval()
        self.subgraph_premodel.eval()
        
        self.remodel.fit(graph, self.node_premodel.shared_encoder, self.edge_premodel.shared_encoder, self.subgraph_premodel.shared_encoder)
        
        scores = self.remodel.decision_function(graph)

        auc_score = roc_auc_score(graph.y.bool().cpu().numpy(), scores)
        return auc_score
    
    def forward(self, graph):
        self.pretrain(graph)
        auc_score = self.retrain(graph)
        return auc_score
    

class ReModel(nn.Module):
    def __init__(self,
                 hid_dim=64,
                 drop_prob=0.3,
                 alpha=None,
                 lr=5e-3,
                 epoch=5,
                 batch_size=0, # all 0
                 num_neigh=-1, # all -1
                 loss_weight=0., 
                 device='cpu',
                 aggr_f='add'):
        super().__init__()

        # model param
        self.hid_dim = hid_dim
        self.alpha = alpha

        # training param
        self.lr = lr
        self.epoch = epoch
        self.device = device
        self.batch_size = batch_size
        self.num_neigh = num_neigh

        self.drop_prob = drop_prob
        self.num_layers = 4
        self.T = 2
        self.loss_weight = loss_weight
        self.weight_decay = 2e-4

        self.aggr = aggr_f

    def fit(self, G, pretrain_node_encoder=None, pretrain_edge_encoder=None, pretrain_subgraph_encoder=None):
        """
        Fit detector with input data.

        Parameters
        ----------
        G : torch_geometric.data.Data
            The input data.
        y_true : numpy.ndarray, optional
            The optional outlier ground truth labels used to monitor
            the training progress. They are not used to optimize the
            unsupervised model. Default: ``None``.

        Returns
        -------
        self : object
            Fitted estimator.
        """
        G.node_idx = torch.arange(G.x.shape[0])
        G.s = to_dense_adj(G.edge_index)[0]


        if self.alpha is None:
            self.alpha = torch.std(G.s).detach() / (torch.std(G.x).detach() + torch.std(G.s).detach())

        if self.batch_size == 0:
            self.batch_size = G.x.shape[0]
   
        self.num_node=self.batch_size

        loader = NeighborLoader(G, [self.num_neigh] * self.num_layers, batch_size=self.batch_size)

        self.model = ReModel_Base(in_dim=G.x.shape[1],
                                   hid_dim=self.hid_dim,
                                   drop_prob=self.drop_prob,
                                   aggr=self.aggr).to(self.device)


        self.model.attr_encoder.load_state_dict(pretrain_node_encoder.state_dict())
        self.model.struct_encoder.load_state_dict(pretrain_edge_encoder.state_dict())
        self.model.topology_encoder.load_state_dict(pretrain_subgraph_encoder.state_dict())

        for k,v in self.model.named_parameters():
            if k.split('.')[0]=='attr_encoder' or k.split('.')[0]=='struct_encoder' or k.split('.')[0]=='topology_encoder':
                v.requires_grad=False

        optimizer = torch.optim.Adam(self.model.parameters(),
                                     lr=self.lr,
                                     weight_decay=self.weight_decay)

        self.model.train()
        decision_scores = np.zeros(G.x.shape[0])
        for epoch in range(self.epoch):
            epoch_loss = 0
            for sampled_data in loader:
                batch_size = sampled_data.batch_size
                node_idx = sampled_data.node_idx
                x, s, edge_index = self.process_graph(sampled_data)
                
                x_, s_,soft_attention= self.model(x, edge_index)

                rank_score = self.loss_func(x[:batch_size],
                                       x_[:batch_size],
                                       s[:batch_size, node_idx],
                                       s_[:batch_size])
                loss = torch.mean(rank_score)
                decision_scores[node_idx[:batch_size]] = rank_score.detach().cpu().numpy()

                # epoch_loss += loss.item() * batch_size

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        #     if self.verbose:
        #         print("Epoch {:04d}: Loss {:.4f}"
        #               .format(epoch, epoch_loss / G.x.shape[0]), end='')
        #         if y_true is not None:
        #             auc = eval_roc_auc(y_true, decision_scores)
        #             print(" | AUC {:.4f}".format(auc), end='')
        #         print()

        # self.decision_scores_ = decision_scores
        # self._process_decision_scores()
        return self
        
    def decision_function(self, G):
        """
        Predict raw anomaly score using the fitted detector. Outliers
        are assigned with larger anomaly scores.

        Parameters
        ----------
        G : PyTorch Geometric Data instance (torch_geometric.data.Data)
            The input data.

        Returns
        -------
        outlier_scores : numpy.ndarray
            The anomaly score of shape :math:`N`.
        """
        check_is_fitted(self, ['model'])
        G.node_idx = torch.arange(G.x.shape[0])
        G.s = to_dense_adj(G.edge_index)[0]

        loader = NeighborLoader(G,
                                [self.num_neigh] * self.num_layers,
                                batch_size=self.batch_size)

        self.model.eval()
        outlier_scores = np.zeros(G.x.shape[0])
        for sampled_data in loader:
            batch_size = sampled_data.batch_size
            node_idx = sampled_data.node_idx

            x, s, edge_index = self.process_graph(sampled_data)
            x_, s_,_ = self.model(x, edge_index)
            rank_score = self.loss_func(x[:batch_size],
                                   x_[:batch_size],
                                   s[:batch_size, node_idx],
                                   s_[:batch_size])
            # print("rank_score: ", rank_score)
            outlier_scores[node_idx[:batch_size]] = rank_score.detach().cpu().numpy()
        return outlier_scores
    def process_graph(self, G):
        s = G.s.to(self.device)
        edge_index = G.edge_index.to(self.device)
        x = G.x.to(self.device)

        return x, s, edge_index
    
    def loss_func(self, x, x_, s, s_):
        score=self.rec_loss(x,x_,s,s_)
        entropy_loss=self.log_t_entropy_loss(x,x_,s,s_,score)

        rank_score = score + self.loss_weight * entropy_loss
        return rank_score

    def log_t_entropy_loss(self,x,x_,s,s_,score):
        diag_s= torch.eye(s.size()[0]).to(s.device) + s
        all_score=score.repeat(score.size()[0],1).float()

        all_score=torch.where(diag_s.float()>0.1,all_score,torch.tensor(0.0, dtype=torch.float).to(s.device))+1e-6
        log_all_score=torch.log(all_score) / self.T

        all_score=F.softmax(log_all_score,dim =1)

        all_log_score=-torch.log(all_score) * all_score
        all_log_score=torch.sum(all_log_score,1)
        
        return all_log_score

    def rec_loss(self, x, x_, s, s_):

        diff_attribute = torch.pow(x_ - x, 2)
        attribute_errors = torch.sqrt(torch.sum(diff_attribute, 1))

        diff_structure = torch.pow(s_ - s, 2)
        structure_errors = torch.sqrt(torch.sum(diff_structure, 1))

        score = self.alpha * attribute_errors + (1 - self.alpha) * structure_errors
        return score

class ReModel_Base(nn.Module):
    def __init__(self,
                 in_dim,
                 hid_dim,
                 drop_prob,
                 aggr='add'
                 ):
        super().__init__()
        self.attr_encoder = Encoder(in_dim, hid_dim, drop_prob)
        self.struct_encoder = Encoder(in_dim, hid_dim, drop_prob)
        self.topology_encoder = Encoder(in_dim, hid_dim, drop_prob)

        self.attention_layer1 =torch.nn.Linear(hid_dim*3, hid_dim*3)
        self.attention_layer2=torch.nn.Softmax(dim=2)

        decoder_in_dim=hid_dim

        # self.attr_decoder = Attribute_Decoder(in_dim, decoder_in_dim, drop_prob, aggr=aggr)
        # self.struct_decoder = Structure_Decoder(decoder_in_dim, drop_prob, aggr=aggr)

        self.attr_decoder = GCN(
                in_channels=int(decoder_in_dim),
                hidden_channels=int(hid_dim),
                num_layers=1,
                out_channels=int(in_dim),
                dropout=drop_prob,
                aggr=aggr)
        
        self.struct_decoder = GCN(
                in_channels=int(hid_dim),
                hidden_channels=int(hid_dim),
                num_layers=1,
                out_channels=int(in_dim),
                dropout=drop_prob,
                aggr=aggr)
  
    def forward(self, x, edge_index):

        h_attr=self.attr_encoder(x, edge_index)
        h_struct=self.struct_encoder(x,edge_index)
        h_topology=self.topology_encoder(x,edge_index)

        # attention agg
        self.attention = self.attention_layer1(torch.cat([h_attr,h_struct,h_topology],dim=1))
        self.attention = self.attention_layer2(torch.reshape(self.attention,(-1,h_attr.size()[-1],3)))
        h = h_attr * self.attention[:,:,0] + h_struct * self.attention[:,:,1] + h_topology * self.attention[:,:,2]
        h=h.to(torch.float32)

        x_ = self.attr_decoder(h, edge_index)
        h_ = self.struct_decoder(h, edge_index)
        s_ = h_ @ h_.T

        return x_, s_, self.attention

