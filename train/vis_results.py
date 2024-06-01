import matplotlib.pyplot as plt
import numpy as np
import torch
import pandas as pd
import seaborn as sns
from collections import Counter

import hypernetx as hnx
from hgraph import load_hgraph, put_hgraph_attributes_on_device
from models import get_model_class
from train.train_utils import eval

from omegaconf import OmegaConf
import json




def data_to_hnxhypergraph(data):
    """
    convention: nodes numbered from 0 corresponding to index, hedge numbered
    """

    from torch_geometric.data import Data
    assert isinstance(data, Data)

    num_nodes = data.n_x

    assert data.x.size(0) == num_nodes
    assert data.y.size(0) == num_nodes
    assert torch.max(data.edge_index[0]).item() == num_nodes-1
    if hasattr(data, 'totedges'):
        assert torch.max(data.edge_index[1]).item() == data.totedges-1 # num_hyperedges \neq totedges if preprocessing included self-loops etc.
    else:
        assert torch.max(data.edge_index[1]).item() == data.num_hyperedges-1

    incidence_dict = {}
    for i in range(data.edge_index.size(1)):
        node = data.edge_index[0,i].item()
        hedge = data.edge_index[1,i].item()
        node_name = node
        edge_name = f"e{hedge:05}"
        if edge_name not in incidence_dict:
            incidence_dict[edge_name] = []
        incidence_dict[edge_name].append(node_name)

    hgraph = hnx.Hypergraph(incidence_dict)

    for attr in data:
        attr_name, attr_val = attr
        setattr(hgraph, attr_name, attr_val)

    return hgraph



def get_single_run(path, device=None, load_best=False):

    with open(path / "cfg.json", "r") as f:
        cfg = json.load(f)
    cfg = OmegaConf.create(cfg)

    train_stats = pd.read_csv(path / "train_stats.csv", index_col=0)

    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    try:
        # hgraphs saved as attributed hnx.Hypergraph

        hgraph = load_hgraph(path / 'hgraph.pickle')
        put_hgraph_attributes_on_device(hgraph, device)

        model_args = dict(cfg.model.model_params)
        model_args["input_dim"] = hgraph.x.shape[1]
        model_args["output_dim"] = hgraph.num_classes

        model = get_model_class(cfg.model.model)(**model_args)

    except:
        # hgraphs saved as torch_geometric.data.Data objects

        data = torch.load(path / 'hgraph.pt')
        data.to(device)

        from train_allset import parse_method
        model = parse_method(cfg, data)

        hgraph = data_to_hnxhypergraph(data) 

    if load_best:
        model.load_state_dict(torch.load(path / "best_model"))
    else:
        model.load_state_dict(torch.load(path / "model"))
    model.eval()
    model.to(device)
    
    return cfg, train_stats, hgraph, model



def show_single_run(path):

    with open(path / "cfg.json", "r") as f:
        cfg = json.load(f)
    print(OmegaConf.to_yaml(cfg))


    train_stats = pd.read_csv(path / "train_stats.csv", index_col=0)

    print('end of training')
    print(train_stats.tail())

    print()
    print('best during training')
    print(train_stats.sort_values("train_acc").tail())

    plt.figure(figsize=(5,5))
    plt.plot(train_stats["train_acc"], label="train_acc")
    plt.plot(train_stats["val_acc"], label="val_acc")
    plt.xlabel('epoch')
    plt.legend()



def get_result_set(settings, runs=5, make_plots=False):


    stuff = {k: {} for k in settings}


    for setting in settings:

        if make_plots:
            fig_train, ax_train = plt.subplots(figsize=(5,5))
            fig_val, ax_val = plt.subplots(figsize=(5,5))

        stuff[setting]['train_acc_best'] = []

        for run in range(runs):

            path = settings[setting] / f'run{run}'

            train_stats = pd.read_csv(path / "train_stats.csv", index_col=0)

            if make_plots:
                ax_train.plot(train_stats['train_acc'], label=run)
                ax_val.plot(train_stats['val_acc'], label=run)

            for col in train_stats:
                if col not in stuff[setting]:
                    stuff[setting][col] = []
                stuff[setting][col].append(train_stats.tail(1)[col].item())

            stuff[setting]['train_acc_best'].append(train_stats.sort_values('train_acc').tail(1)['train_acc'].item())
        
        if make_plots:

            ax_train.legend()
            ax_train.set_title(setting)
            ax_train.set_xlabel('epoch')
            ax_train.set_ylabel('train_acc')
            plt.close()

            ax_val.legend()
            ax_val.set_title(setting)
            ax_val.set_xlabel('epoch')
            ax_val.set_ylabel('val_acc')
            plt.close()

            stuff[setting]['fig_train_acc'] = fig_train
            stuff[setting]['fig_val_acc'] = fig_val
            del fig_train, ax_train, fig_val, ax_val


    return stuff



def summarise_result_set(settings, stuff=None):

    if stuff is None:
        stuff = get_result_set(settings)

    summary = pd.DataFrame(columns=['train_acc', 'val_acc', 'train_loss', 'train_loss_xent', 'train_loss_contr'], index=settings.keys(), dtype=pd.StringDtype())

    for setting in settings:
        for col in summary.columns:
            if col not in stuff[setting]: continue
            summary.loc[setting, col] = f"{np.mean(stuff[setting][col]) : .2f} ± {np.std(stuff[setting][col]) : .2f}"

    return summary



def show_label_distribution(hgraph, num_classes, show_bar_chart=False):

    labels = hgraph.y
    labels_train = hgraph.y[hgraph.train_mask]
    labels_val = hgraph.y[hgraph.val_mask]

    labels_cnt = Counter(labels.tolist())
    labels_train_cnt = Counter(labels_train.tolist())
    labels_val_cnt = Counter(labels_val.tolist())

    print(len(labels), labels_cnt)
    print(len(labels_train), labels_train_cnt)
    print(len(labels_val), labels_val_cnt)

    assert len(labels_cnt) == len(labels_train_cnt) == len(labels_val_cnt) == num_classes

    if show_bar_chart:

        labels_unique = sorted(labels_cnt.keys())

        df = pd.DataFrame({
            'class': labels_unique,
            'all': [labels_cnt[k] for k in labels_unique],
            'train': [labels_train_cnt[k] for k in labels_unique],
            'val': [labels_val_cnt[k] for k in labels_unique],
        })
        tidy = df.melt(id_vars='class').rename(columns=str.title)
        fig, ax = plt.subplots(figsize=(5, 5))
        sns.barplot(x='Class', y='Value', hue='Variable', data=tidy, ax=ax)
        ax.set_ylabel('Count')
        ax.set_title('Count of classes across splits')
        plt.show()



def show_intraclass_accs(hgraph, model, num_classes):

    print('train acc', eval(hgraph, model, hgraph.train_mask))
    print('val acc', eval(hgraph, model, hgraph.val_mask))


    for cl in range(num_classes):
        mask = (hgraph.y == cl) & hgraph.train_mask
        print(f'class {cl} train acc \t {eval(hgraph, model, mask):.2f} \t freq {sum(mask)/len(mask):.2f}')
        mask = (hgraph.y == cl) & hgraph.val_mask
        print(f'class {cl} val acc \t {eval(hgraph, model, mask):.2f} \t freq {sum(mask)/len(mask):.2f}')