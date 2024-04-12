# %%

import torch
import torch_sparse
import scipy.sparse as sp

from models.allset import *

from train import train_eval_loop, eval, set_seed
from hgraph.utils import load_hgraph
from torch_geometric.data import Data

import hydra
from omegaconf import DictConfig
from easydict import EasyDict
from pathlib import Path
from datetime import datetime
import pandas as pd
import json



ALLSET_EXISTING_DATASETS = [
    '20newsW100',
    'ModelNet40',
    'zoo',
    'NTU2012',
    'Mushroom',
    'coauthor_cora',
    'coauthor_dblp',
    'yelp',
    'amazon-reviews',
    'walmart-trips',
    'house-committees',
    'walmart-trips-100',
    'house-committees-100',
    'cora',
    'citeseer',
    'pubmed',
]

# %% 

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



def load_data(args):
    
    if args.path_data in ALLSET_EXISTING_DATASETS:
        """
        Refactored from Allset
        https://github.com/jianhao2016/AllSet/blob/main/src/train.py
        """
                
        dname = args.path_data

        if dname in ['cora', 'citeseer','pubmed']:
            p2raw = 'data/AllSet_all_raw_data/cocitation/'
        elif dname in ['coauthor_cora', 'coauthor_dblp']:
            p2raw = 'data/AllSet_all_raw_data/coauthorship/'
        elif dname in ['yelp']:
            p2raw = 'data/AllSet_all_raw_data/yelp/'
        else:
            p2raw = 'data/AllSet_all_raw_data/'


        dataset_allset = dataset_Hypergraph(
            name=dname,
            root='data/pyg_data/hypergraph_dataset_updated/',
            p2raw=p2raw,
        )

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
        hgraph = load_hgraph(args.path_data)

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

@hydra.main(version_base=None, config_path="configs/train_allset", config_name="ast")
def main(cfg : DictConfig) -> None:

    # Gets args, data, device
    # --------------------------------------
    
    cfg = EasyDict(cfg)
    print(cfg)

    data = load_data(cfg)

    device = torch.device('cuda:'+str(cfg.cuda) if torch.cuda.is_available() else 'cpu')


    # Set seed
    # --------------------------------------
    set_seed(cfg.seed) # not effective


    # Get splits
    # --------------------------------------

    use_own_splits = not cfg.path_data in ALLSET_EXISTING_DATASETS

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
        for run in range(cfg.runs):
            split_idx = rand_train_test_idx(
                data.y, train_prop=cfg.train_prop, valid_prop=cfg.valid_prop)
            split_idx_lst.append(split_idx)

        # Just train with run 0 of the split
        run = 0
        split_idx = split_idx_lst[run]
        train_mask = split_idx['train'].to(device)
        val_mask = split_idx['valid'].to(device)
        test_mask = split_idx['test'].to(device)

        # load the splits into the hgraph data
        data.train_mask = torch.zeros(data.n_x, dtype=bool)
        data.val_mask = torch.zeros(data.n_x, dtype=bool)
        data.test_mask = torch.zeros(data.n_x, dtype=bool)
        data.train_mask[train_mask] = True
        data.val_mask[val_mask] = True
        data.test_mask[test_mask] = True


    # Load model
    # --------------------------------------
    
    model = parse_method(cfg, data)
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
        lr=cfg.lr,
        num_epochs=cfg.epochs,
        contr_lambda=0.0,
        printevery=cfg.display_step,
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
    
    print(f"Saving to {cfg.save_dir}")
    save_stuff(cfg, train_stats, data, model, best_model)



if __name__ == '__main__':
    main()

# %%
