import torch


def train_conad(dataset, cuda=True, epoch1=100, epoch2=50, lr=1e-3, margin=0.5):
    # data = load_data('inj_flickr') # in PyG format
    # print(data)
    # input attributed network G
    adj, attrs, label, _ = load_anomaly_detection_dataset(dataset)
    # create graph and attribute object, as anchor point
    graph1 = dgl.from_scipy(scipy.sparse.coo_matrix(adj)).add_self_loop()
    attrs1 = torch.FloatTensor(attrs)
    num_attr = attrs.shape[1]
    print("graph1: ", graph1)
    print("attrs1: ", attrs1)

    # hidden dimension, output dimension
    hidden_dim, out_dim = 128, 64
    hidden_num = 2
    model = GRL(num_attr, hidden_dim, out_dim, hidden_num)

    criterion = lambda z, z_hat, l: torch.square(z - z_hat) * (l==0) - l * torch.square(z - z_hat) + margin
    
    cuda_device = torch.device('cuda') if cuda else torch.device('cpu')
    cpu_device = torch.device('cpu')
    model = model.to(cuda_device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    t = datetime.strftime(datetime.now(), '%y_%m_%d_%H_%M')
    sw = SummaryWriter('logs/siamese_%s_%s' % (dataset, t))
    
    model.train()
    
    # anomaly injection
    adj_aug, attrs_aug, label_aug = make_anomalies(adj, attrs, rate=.2, clique_size=20, sourround=50)
    graph2 = dgl.from_scipy(scipy.sparse.coo_matrix(adj_aug)).add_self_loop()
    attrs2 = torch.FloatTensor(attrs_aug)
    
    # train encoder with supervised contrastive learning
    for i in tqdm(range(epoch1)):
        # augmented labels introduced by injection
        labels = torch.FloatTensor(label_aug).unsqueeze(-1)
        if cuda:
            graph1 = graph1.to(cuda_device)
            attrs1 = attrs1.to(cuda_device)
            graph2 = graph2.to(cuda_device)
            attrs2 = attrs2.to(cuda_device)
            labels = labels.to(cuda_device)
        
        # train siamese loss
        orig = model.embed(graph1, attrs1)
        aug = model.embed(graph2, attrs2)
        margin_loss = criterion(orig, aug, labels)
        margin_loss = margin_loss.mean()
        sw.add_scalar('train/margin_loss', margin_loss, i)
        optimizer.zero_grad()
        margin_loss.backward()
        optimizer.step()
        
        # train reconstruction
        A_hat, X_hat = model(graph1, attrs1)
        a = graph1.adjacency_matrix().to_dense()
        recon_loss, struct_loss, feat_loss = loss_func(a.cuda() if cuda else a, A_hat, attrs1, X_hat, weight1=1, weight2=1, alpha=.7, mask=1)
        recon_loss = recon_loss.mean()
        # loss = bce_loss + recon_loss
        optimizer.zero_grad()
        recon_loss.backward()
        optimizer.step()
        sw.add_scalar('train/rec_loss', recon_loss, i)
        sw.add_scalar('train/struct_loss', struct_loss, i)
        sw.add_scalar('train/feat_loss', feat_loss, i)
    
    # evaluate
    model.eval()
    with torch.no_grad():
        A_hat, X_hat = model(graph1, attrs1)
        A_hat, X_hat = A_hat.cpu(), X_hat.cpu()
        a = graph1.adjacency_matrix().to_dense().cpu()
        recon_loss, struct_loss, feat_loss = loss_func(a, A_hat, attrs1.cpu(), X_hat, weight1=1, weight2=1, alpha=.3)
        score = recon_loss.detach().numpy()
        print('AUC: %.4f' % roc_auc_score(label, score))
        for k in [50, 100, 200, 300]:
            print('Precision@%d: %.4f' % (k, precision_at_k(label, score, k)))