def train_adagad(args):
    """
        Process the raw PyG data object into a tuple of sub data
        objects needed for the model.

        Parameters
        ----------
        G : PyTorch Geometric Data instance (torch_geometric.data.Data)
            The input data.

        Returns
        -------
        x : torch.Tensor
            Attribute (feature) of nodes.
        s : torch.Tensor
            Adjacency matrix of the graph.
        edge_index : torch.Tensor
            Edge list of the graph.
    """
    graph=load_data(dataset_name)
    graph.edge_index=add_remaining_self_loops(graph.edge_index)[0]
    graph.s=to_dense_adj(graph.edge_index)[0]
    num_features=graph.x.size()[1]
    num_classes=4
    print("graph: ", graph)
    args.num_features = num_features

    auc_score_list = []

    attr_mask,struct_mask=None,None
    pretrain_auc_score_list=[]
    for i, seed in enumerate(seeds):
        print(f"####### Run {i} for seed {seed}")
        seed=int(seed)
        set_random_seed(seed)

        if logs:
            logger = TBLogger(name=f"{dataset_name}_loss_{loss_fn}_rpr_{replace_rate}_nh_{num_hidden}_nl_{num_layers}_lr_{lr}_mp_{max_epoch}_mpf_{max_epoch_f}_wd_{weight_decay}_wdf_{weight_decay_f}_{encoder_type}_{attr_decoder_type}_{struct_decoder_type}")
        else:
            logger = None

        attr_model,struct_model,topology_model = build_model(args)
        attr_model.to(device)
        struct_model.to(device)
        topology_model.to(device)

        if args.use_ssl:
            attr_remask=None
            struct_remask=None
            print('======== train attr encoder ========')
            if args.use_encoder_num>=1:

                optimizer = create_optimizer(optim_type, attr_model, lr, weight_decay)

                if use_scheduler:
                    logging.info("Use schedular")
                    scheduler = lambda epoch :( 1 + np.cos((epoch) * np.pi / max_epoch) ) * 0.5
                    # scheduler = lambda epoch: epoch / warmup_steps if epoch < warmup_steps \
                            # else ( 1 + np.cos((epoch - warmup_steps) * np.pi / (max_epoch - warmup_steps))) * 0.5
                    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=scheduler)
                else:
                    scheduler = None
                    
                x = graph.x
                if not load_model:
                    attr_model,attr_outlier_list,_= pretrain(attr_model, graph, x, optimizer, max_epoch, device, scheduler, model_name=model_name,aggr_f=aggr_f,lr_f=lr_f, max_epoch_f=max_epoch_f, alpha_f=alpha_f,dropout_f=dropout_f,loss_f=loss_f,loss_weight_f=loss_weight_f,T_f=T_f, num_hidden=args.num_hidden,logger=logger,use_ssl=args.use_ssl)
                    attr_model = attr_model.cpu()

                if load_model:
                    logging.info("Loading Model ... ")
                    attr_model.load_state_dict(torch.load("checkpoint.pt"))
                if save_model:
                    logging.info("Saveing Model ...")
                    torch.save(attr_model.state_dict(), "checkpoint.pt")
                
                attr_model = attr_model.to(device)
                attr_model.eval()

            print('======== train struct encoder ========')
            if args.use_encoder_num>=2:

                optimizer = create_optimizer(optim_type, struct_model, lr, weight_decay)

                if use_scheduler:
                    logging.info("Use schedular")
                    scheduler = lambda epoch :( 1 + np.cos((epoch) * np.pi / max_epoch) ) * 0.5
                    # scheduler = lambda epoch: epoch / warmup_steps if epoch < warmup_steps \
                            # else ( 1 + np.cos((epoch - warmup_steps) * np.pi / (max_epoch - warmup_steps))) * 0.5
                    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=scheduler)
                else:
                    scheduler = None
                    
                x = graph.x
                if not load_model:
                    struct_model,struct_node_outlier_list,struct_outlier_list= pretrain(struct_model, graph, x, optimizer, max_epoch, device, scheduler,  model_name=model_name,aggr_f=aggr_f,lr_f=lr_f, max_epoch_f=max_epoch_f, alpha_f=alpha_f,dropout_f=dropout_f,loss_f=loss_f,loss_weight_f=loss_weight_f,T_f=T_f, num_hidden=args.num_hidden,logger=logger,use_ssl=args.use_ssl)
                    struct_model = struct_model.cpu()

                if load_model:
                    logging.info("Loading Model ... ")
                    struct_model.load_state_dict(torch.load("checkpoint.pt"))
                if save_model:
                    logging.info("Saveing Model ...")
                    torch.save(struct_model.state_dict(), "checkpoint.pt")
                
                struct_model = struct_model.to(device)
                struct_model.eval()

            print('======== train topology encoder ========')
            if args.use_encoder_num>=3:

                optimizer = create_optimizer(optim_type, topology_model, lr, weight_decay)

                if use_scheduler:
                    logging.info("Use schedular")
                    scheduler = lambda epoch :( 1 + np.cos((epoch) * np.pi / max_epoch) ) * 0.5
                    # scheduler = lambda epoch: epoch / warmup_steps if epoch < warmup_steps \
                            # else ( 1 + np.cos((epoch - warmup_steps) * np.pi / (max_epoch - warmup_steps))) * 0.5
                    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=scheduler)
                else:
                    scheduler = None
                    
                x = graph.x
                if not load_model:
                    topology_model,topology_node_outlier_list,topology_outlier_list= pretrain(topology_model, graph, x, optimizer, max_epoch, device, scheduler,  model_name=model_name,aggr_f=aggr_f,lr_f=lr_f, max_epoch_f=max_epoch_f, alpha_f=alpha_f,dropout_f=dropout_f,loss_f=loss_f,loss_weight_f=loss_weight_f,T_f=T_f, num_hidden=args.num_hidden,logger=logger,use_ssl=args.use_ssl)
                    topology_model = topology_model.cpu()

                if load_model:
                    logging.info("Loading Model ... ")
                    topology_model.load_state_dict(torch.load("checkpoint.pt"))
                if save_model:
                    logging.info("Saveing Model ...")
                    torch.save(topology_model.state_dict(), "checkpoint.pt")
                
                topology_model = topology_model.to(device)
                struct_model.eval()

        print('finish one train!')
        # god_evaluation is Retrain?
        auc_score,ap_score,ndcg_score,pk_score,rk_score,final_outlier,_= god_evaluation(dataset_name,model_name,attr_encoder_name,struct_encoder_name,topology_encoder_name,attr_decoder_name,struct_decoder_name,attr_model,struct_model,topology_model,graph, graph.x, aggr_f,lr_f, max_epoch, alpha_f,dropout_f,loss_f,loss_weight_f,T_f,args.num_hidden,node_encoder_num_layers,edge_encoder_num_layers,subgraph_encoder_num_layers,attr_decoder_num_layers,struct_decoder_num_layers,use_ssl=args.use_ssl,use_encoder_num=args.use_encoder_num,attention=args.attention,sparse_attention_weight=args.sparse_attention_weight,theta=args.theta,eta=args.eta)
        auc_score_list.append(auc_score)

        if logger is not None:
            logger.finish()

    final_auc, final_auc_std = np.mean(auc_score_list), np.std(auc_score_list)



    print(f"# final_auc: {final_auc*100:.2f}±{final_auc_std*100:.2f}")