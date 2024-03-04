# %%

import torch
import torch_sparse
import scipy.sparse as sp

from models.allset import *

from train import train_eval_loop, eval
from hgraph.utils import load_hgraph
from torch_geometric.data import Data

from easydict import EasyDict
from pathlib import Path
from datetime import datetime
import pandas as pd
import json

# %%
        

def get_args_AllDeepSets():

    args = EasyDict(
        method = "AllDeepSets",
        add_self_loop = False,
        exclude_self = False,
        normtype = "all_one",
        runs = 1,
        train_prop = 0.8,
        valid_prop = 0.1,
        LearnMask = False,
        All_num_layers = 3,
        dropout = 0.0,
        aggregate = "add",
        normalization = "ln",
        deepset_input_norm = False,
        GPR = False,
        MLP_hidden = 16,
        MLP_num_layers = 2,
        heads = 1,
        PMA = False, # this will get set to False for AllDeepSets anyway
        alpha_softmax = None, # not used since self.PMA = False
        Classifier_hidden = 16,
        Classifier_num_layers = 2,
        cuda = 0,
        lr = 0.001,
        epochs = 500,
        display_step = 50,
        wd = 0.0,
    )

    return args     



def get_args_AllSetTransformer():

    args = EasyDict(
        method = "AllSetTransformer",
        add_self_loop = False,
        exclude_self = False,
        normtype = "all_one",
        runs = 1,
        train_prop = 0.8,
        valid_prop = 0.1,
        LearnMask = False,
        All_num_layers = 3,
        dropout = 0.0,
        aggregate = "add",
        normalization = "ln",
        deepset_input_norm = False,
        GPR = False,
        MLP_hidden = 16,
        MLP_num_layers = 2,
        heads = 1,
        PMA = True,
        alpha_softmax = False, # do not compute softmax over attentions, effectively collapsing to a mean
        Classifier_hidden = 16,
        Classifier_num_layers = 2,
        cuda = 0,
        lr = 0.001,
        epochs = 500,
        display_step = 50,
        wd = 0.0,
    )

    return args



def get_args_HGNN():
    
    args = EasyDict(
        method = "HGNN",
        add_self_loop = False,
        runs = 1,
        train_prop = 0.8,
        valid_prop = 0.1,
        All_num_layers = 3,
        dropout = 0.0,
        MLP_hidden = 16,
        HCHA_symdegnorm = None,
        alpha_softmax = None, # not needed since running HCHA with use_attention = False
        cuda = 0,
        lr = 0.001,
        epochs = 2000,
        display_step = 100,
        wd = 0.0,
    )

    return args



def get_args_HCHA():

    args = EasyDict(
        method = "HCHA",
        add_self_loop = False,
        runs = 1,
        train_prop = 0.8,
        valid_prop = 0.1,
        All_num_layers = 3,
        dropout = 0.0,
        MLP_hidden = 16,
        HCHA_symdegnorm = None,
        alpha_softmax = False,
        cuda = 0,
        lr = 0.001,
        epochs = 2000,
        display_step = 100,
        wd = 0.0,
    )

    return args



def get_args(method):

    if method == 'AllDeepSets':
        return get_args_AllDeepSets()
    
    if method == "AllSetTransformer":
        return get_args_AllSetTransformer()

    if method == "HGNN":
        return get_args_HGNN()
    
    if method == "HCHA":
        return get_args_HCHA()
    
    raise NotImplementedError



def further_process_data(data, args):
    """
    Refactored from Allset
    https://github.com/jianhao2016/AllSet/blob/main/src/train.py
    """


    if not hasattr(data, 'n_x'):
        data.n_x = torch.tensor([data.x.shape[0]])
    if not hasattr(data, 'num_hyperedges'):
        # note that we assume the he_id is consecutive.
        data.num_hyperedges = torch.tensor(
            [data.edge_index[0].max()-data.n_x[0]+1])


    if args.method in ['AllSetTransformer', 'AllDeepSets']:
        data = ExtractV2E(data)
        if args.add_self_loop:
            data = Add_Self_Loops(data)
        if args.exclude_self:
            data = expand_edge_index(data)

        #     Compute deg normalization: option in ['all_one','deg_half_sym'] (use args.normtype)
        # data.norm = torch.ones_like(data.edge_index[0])
        data = norm_contruction(data, option=args.normtype)
    elif args.method in ['CEGCN', 'CEGAT']:
        data = ExtractV2E(data)
        data = ConstructV2V(data)
        data = norm_contruction(data, TYPE='V2V')

    elif args.method in ['HyperGCN']:
        data = ExtractV2E(data)

    elif args.method in ['HNHN']:
        data = ExtractV2E(data)
        if args.add_self_loop:
            data = Add_Self_Loops(data)
        H = ConstructH_HNHN(data)
        data = generate_norm_HNHN(H, data, args)
        data.edge_index[1] -= data.edge_index[1].min()

    elif args.method in ['HCHA', 'HGNN']:
        data = ExtractV2E(data)
        if args.add_self_loop:
            data = Add_Self_Loops(data)
    #    Make the first he_id to be 0
        data.edge_index[1] -= data.edge_index[1].min()
        
    elif args.method in ['UniGCNII']:
        data = ExtractV2E(data)
        if args.add_self_loop:
            data = Add_Self_Loops(data)
        data = ConstructH(data)
        data.edge_index = sp.csr_matrix(data.edge_index)
        # Compute degV and degE
        if args.cuda in [0,1]:
            device = torch.device('cuda:'+str(args.cuda) if torch.cuda.is_available() else 'cpu')
        else:
            device = torch.device('cpu')
        (row, col), value = torch_sparse.from_scipy(data.edge_index)
        V, E = row, col
        V, E = V.to(device), E.to(device)

        degV = torch.from_numpy(data.edge_index.sum(1)).view(-1, 1).float().to(device)
        from torch_scatter import scatter
        degE = scatter(degV[V], E, dim=0, reduce='mean')
        degE = degE.pow(-0.5)
        degV = degV.pow(-0.5)
        degV[torch.isinf(degV)] = 1
        args.UniGNN_degV = degV
        args.UniGNN_degE = degE

        V, E = V.cpu(), E.cpu()
        del V
        del E

    return data



def load_data(
        args,
        load_allset_data = False,
        path_data = "data/standard/ones1/hgraph1.pickle",
    ):

    
    if load_allset_data:
        """
        Refactored from Allset
        https://github.com/jianhao2016/AllSet/blob/main/src/train.py
        """
        
        assert path_data is None, "Igoring data path, loading Allset's coauthor_cora"
        
        args.dname = "coauthor_cora"
        args.p2raw = "data/AllSet_all_raw_data/coauthorship/"


        dataset_allset = dataset_Hypergraph(
            name=args.dname,
            root = 'data/pyg_data/hypergraph_dataset_updated/',
            p2raw=args.p2raw)

        data = dataset_allset.data

        args.num_features = dataset_allset.num_features
        args.num_classes = dataset_allset.num_classes


    else:

        # load your own hgraph as data


        def get_weird_edge_index(edge_index, num_nodes):

            weird_edge_index = edge_index.clone()
            weird_edge_index[1,:] = weird_edge_index[1,:] + num_nodes

            weird_edge_index = torch.hstack((weird_edge_index, torch.roll(weird_edge_index, shifts=1, dims=0)))

            return weird_edge_index


        # Load example synthetic data hgraph
        hgraph = load_hgraph(path_data)

        args.num_features = hgraph.x.shape[1]
        args.num_classes = hgraph.num_classes

        data = Data(
            x=hgraph.x, 
            edge_index=get_weird_edge_index(hgraph.edge_index, hgraph.number_of_nodes()),
            y=hgraph.y,
            n_x=hgraph.number_of_nodes(),
            num_hyperedges=hgraph.number_of_edges(),
            # additionally add your deterministic masks. these are binary masks of shape (num_nodes,)
            train_mask=hgraph.train_mask,
            val_mask=hgraph.val_mask,
            test_mask=hgraph.test_mask,
        )


    data = further_process_data(data, args)

    print(data)

    return data



def parse_method(args, data):
    """
    Taken from Allset
    https://github.com/jianhao2016/AllSet/blob/main/src/train.py
    """

    #     Currently we don't set hyperparameters w.r.t. different dataset
    if args.method == 'AllSetTransformer':
        if args.LearnMask:
            model = SetGNN(args, data.norm)
        else:
            model = SetGNN(args)
    
    elif args.method == 'AllDeepSets':
        args.PMA = False
        args.aggregate = 'add'
        if args.LearnMask:
            model = SetGNN(args,data.norm)
        else:
            model = SetGNN(args)

#     elif args.method == 'SetGPRGNN':
#         model = SetGPRGNN(args)

    elif args.method == 'CEGCN':
        model = CEGCN(in_dim=args.num_features,
                      hid_dim=args.MLP_hidden,  # Use args.enc_hidden to control the number of hidden layers
                      out_dim=args.num_classes,
                      num_layers=args.All_num_layers,
                      dropout=args.dropout,
                      Normalization=args.normalization)

    elif args.method == 'CEGAT':
        model = CEGAT(in_dim=args.num_features,
                      hid_dim=args.MLP_hidden,  # Use args.enc_hidden to control the number of hidden layers
                      out_dim=args.num_classes,
                      num_layers=args.All_num_layers,
                      heads=args.heads,
                      output_heads=args.output_heads,
                      dropout=args.dropout,
                      Normalization=args.normalization)

    elif args.method == 'HyperGCN':
        #         ipdb.set_trace()
        He_dict = get_HyperGCN_He_dict(data)
        model = HyperGCN(V=data.x.shape[0],
                         E=He_dict,
                         X=data.x,
                         num_features=args.num_features,
                         num_layers=args.All_num_layers,
                         num_classses=args.num_classes,
                         args=args
                         )

    elif args.method == 'HGNN':
        # model = HGNN(in_ch=args.num_features,
        #              n_class=args.num_classes,
        #              n_hid=args.MLP_hidden,
        #              dropout=args.dropout)
        model = HCHA(args, use_attention=False)

    elif args.method == 'HNHN':
        model = HNHN(args)

    elif args.method == 'HCHA':
        model = HCHA(args, use_attention=True)

    elif args.method == 'MLP':
        model = MLP_model(args)
    elif args.method == 'UniGCNII':
            if args.cuda in [0,1]:
                device = torch.device('cuda:'+str(args.cuda) if torch.cuda.is_available() else 'cpu')
            else:
                device = torch.device('cpu')
            (row, col), value = torch_sparse.from_scipy(data.edge_index)
            V, E = row, col
            V, E = V.to(device), E.to(device)
            model = UniGCNII(args, nfeat=args.num_features, nhid=args.MLP_hidden, nclass=args.num_classes, nlayer=args.All_num_layers, nhead=args.heads,
                             V=V, E=E)
            
            args.UniGNN_degV = args.UniGNN_degV.to(device)
            args.UniGNN_degE = args.UniGNN_degE.to(device)
    #     Below we can add different model, such as HyperGCN and so on
    return model



def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)



def save_stuff(cfg, train_stats, hgraph, model, best_model):

    path = Path(cfg.save_dir)
    if cfg.save_datestamp:
        path = path / f"_{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}"
    path.mkdir(exist_ok=True, parents=True)

    # save cfg
    dict_cfg = cfg
    with open(path / "cfg.json", "w") as f:
        json.dump(dict_cfg, f, indent=4)

    # save train_stats
    train_stats = pd.DataFrame.from_dict(train_stats)
    train_stats = train_stats.set_index("epoch")
    train_stats.to_csv(path / "train_stats.csv")

    # save hgraph in torch Data format
    torch.save(hgraph, path / "hgraph.pt")
    
    # save model
    torch.save(model.state_dict(), path / "model")
    if best_model is not None:
        torch.save(best_model.state_dict(), path / "best_model")



# %%

def main():

    method = "AllSetTransformer"
    path_data = "data/standard/ones1/hgraph1.pickle"
    save_dir = "train_results/allsettransformer"

    # Gets args, data, device
    # --------------------------------------
    
    print(f"Running {method}...")

    args = get_args(method)
    args.save_datestamp = False
    args.save_dir = save_dir

    data = load_data(args, path_data=path_data)

    device = torch.device('cuda:'+str(args.cuda) if torch.cuda.is_available() else 'cpu')


    # Get splits
    # --------------------------------------

    use_own_splits = True

    if use_own_splits:

        # Loaded with own data
        train_mask = data.train_mask
        val_mask = data.val_mask
        test_mask = data.test_mask

    else:

        """
        Taken from Allset
        https://github.com/jianhao2016/AllSet/blob/main/src/train.py
        """
        split_idx_lst = []
        for run in range(args.runs):
            split_idx = rand_train_test_idx(
                data.y, train_prop=args.train_prop, valid_prop=args.valid_prop)
            split_idx_lst.append(split_idx)

        # Just train with run 0 of the split
        run = 0
        split_idx = split_idx_lst[run]
        train_mask = split_idx['train'].to(device)
        val_mask = split_idx['valid'].to(device)
        test_mask = split_idx['test'].to(device)


    # Load model
    # --------------------------------------
    
    model = parse_method(args, data)
    model.reset_parameters()

    num_params = count_parameters(model)
    print(f"{num_params=}")
    print(model)


    # Train
    # --------------------------------------

    model, data = model.to(device), data.to(device)

    train_stats, best_model = train_eval_loop(
        model=model,
        hgraph=data,
        train_mask=train_mask,
        val_mask=val_mask,
        test_mask=test_mask,
        lr=args.lr,
        num_epochs=args.epochs,
        contr_lambda=0.0,
        printevery=args.display_step,
        save_best=True,
    )


    # Print train outcomes
    # --------------------------------------

    print()
    print("Final Model")
    print(f"train: {eval(data, model, train_mask):.3f} | val {eval(data, model, val_mask):.3f} | test {eval(data, model, test_mask):.3f} ")

    print()
    print("Best Model")
    print(f"train: {eval(data, best_model, train_mask):.3f} | val {eval(data, best_model, val_mask):.3f} | test {eval(data, best_model, test_mask):.3f} ")


    # Save stuff
    # --------------------------------------
    
    save_stuff(args, train_stats, data, model, best_model)



if __name__ == '__main__':
    main()

# %%
