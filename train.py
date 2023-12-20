import os
import pickle
import json
from pathlib import Path
import pandas as pd
from datetime import datetime
import hydra
from omegaconf import DictConfig, OmegaConf

import torch
import hypernetx as hnx

from load_coraca import get_coraca_hypergraph
from hgraph import generate_random_hypergraph, generate_random_uniform_hypergraph, attach_houses_to_incidence_dict, incidence_matrix_to_edge_index, add_random_edges_to_incidence_dict

from models import MyHyperGCN, HyperGCN, HyperResidGCN

from train_utils import get_train_val_test_mask, train_eval_loop, train_eval_loop_many



def make_random_house(cfg):

    if cfg.name == "random_house" or cfg.name == "random_house_perturbed":
        h_base = generate_random_hypergraph(cfg.num_nodes, cfg.num_edges)
    elif cfg.name == "random_unif_house":
        h_base = generate_random_uniform_hypergraph(cfg.num_nodes, cfg.num_edges, cfg.k)
    else:
        raise NotImplementedError

    # anchor_nodes = torch.randint(low=0, high=h_base.number_of_nodes(), size=(num_houses,)).tolist()
    anchor_nodes = range(cfg.num_houses)
    incdict_deco, labels_deco = attach_houses_to_incidence_dict(anchor_nodes, h_base.incidence_dict, h_base.number_of_nodes(), h_base.number_of_edges())

    num_nodes_deco = len(set([node for lst in incdict_deco.values() for node in lst]))
    num_edges_deco = len(incdict_deco)
    assert num_nodes_deco == h_base.number_of_nodes() + 4 * cfg.num_houses
    assert num_edges_deco == h_base.number_of_edges() + 3 * cfg.num_houses

    incdict_deco = add_random_edges_to_incidence_dict(cfg.num_random_edges, incdict_deco, num_nodes_deco, num_edges_deco, cfg.deg_random_edges)

    h_deco = hnx.Hypergraph(incdict_deco)
    print(f"{h_base.number_of_nodes()=}, {h_base.number_of_edges()=}")
    print(f"{h_deco.number_of_nodes()=}, {h_deco.number_of_edges()=}")

    hgraph = h_deco

    train_mask, val_mask, test_mask = get_train_val_test_mask(n=hgraph.number_of_nodes(), split=[0.8, 0.2, 0.0], seed=3)
    hgraph.train_mask = train_mask
    hgraph.val_mask = val_mask
    hgraph.test_mask = test_mask

    if cfg.features == "zeros":
        hgraph.x = torch.zeros((hgraph.number_of_nodes(), 1), dtype=torch.float32)
    elif cfg.features == "ones":
        hgraph.x = torch.ones((hgraph.number_of_nodes(), 1), dtype=torch.float32)
    elif cfg.features == "randn":
        hgraph.x = torch.randn((hgraph.number_of_nodes(), 1), dtype=torch.float32)
    else:
        raise NotImplementedError

    hgraph.y = labels_deco
    hgraph.H = torch.tensor(hgraph.incidence_matrix().toarray())
    hgraph.edge_index = incidence_matrix_to_edge_index(hgraph.H)

    return hgraph



def make_hgraph(cfg):

    if cfg.name == "coraca":
        hgraph = get_coraca_hypergraph(split=[0.5, 0.25, 0.25], split_seed=cfg.split_seed)
    elif cfg.name == "random_house":
        hgraph = make_random_house(cfg)
    elif cfg.name == "random_house_perturbed":
        hgraph = make_random_house(cfg)
    elif cfg.name == "random_unif_house":
        hgraph = make_random_house(cfg)
    else:
        raise NotImplementedError
    
    return hgraph



def put_hgraph_attributes_on_device(hgraph, device) -> None:
    hgraph.train_mask = hgraph.train_mask.to(device)
    hgraph.val_mask = hgraph.val_mask.to(device)
    hgraph.test_mask = hgraph.test_mask.to(device)
    hgraph.x = hgraph.x.to(device)
    hgraph.y = hgraph.y.to(device)
    hgraph.H = hgraph.H.to(device)
    hgraph.edge_index = hgraph.edge_index.to(device)



def get_model_class(model):

    if model == "MyHyperGCN":
        return MyHyperGCN
    elif model == "HyperResidGCN":
        return HyperResidGCN
    else:
        raise NotImplementedError



def save_model(model, cfg):
    fname = os.path.join(cfg.save_dir, cfg.save_name + "_model")
    if cfg.save_datestamp:
        fname += f"_{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}"
    torch.save(model.state_dict(), fname)



def save_stats(train_stats, cfg):
    fname = os.path.join(cfg.save_dir, cfg.save_name)
    if cfg.save_datestamp:
        fname += f"_{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}"
    with open(fname, 'wb') as f:
        pickle.dump(train_stats, f)



def hgraph_to_dict(hgraph):
    
    dict_hgraph = dict(
        incidence_dict = hgraph.incidence_dict,
        train_mask = hgraph.train_mask,
        val_mask = hgraph.val_mask,
        test_mask = hgraph.test_mask,
        x = hgraph.x,
        y = hgraph.y,
        H = hgraph.H,
        edge_index = hgraph.edge_index,
    )

    return dict_hgraph



def save_stuff(cfg, train_stats, hgraph, model):

    path = Path(cfg.save_dir)
    if cfg.save_datestamp:
        path = path / f"_{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}"
    path.mkdir(exist_ok=True, parents=True)

    # save cfg
    dict_cfg = OmegaConf.to_container(cfg)
    with open(path / "cfg.json", "w") as f:
        json.dump(dict_cfg, f, indent=4)

    # save train_stats
    train_stats = pd.DataFrame.from_dict(train_stats)
    train_stats = train_stats.set_index("epoch")
    train_stats.to_csv(path / "train_stats.csv")

    # save hgraph
    dict_hgraph = hgraph_to_dict(hgraph)
    with open(path / "hgraph.pickle", "wb") as f:
        pickle.dump(dict_hgraph, f)
    
    # save model
    torch.save(model.state_dict(), path / "model")



@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg : DictConfig) -> None:

    print(OmegaConf.to_yaml(cfg))


    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


    hgraph = make_hgraph(cfg.hgraph)
    put_hgraph_attributes_on_device(hgraph, device)


    model_args = dict(cfg.model.model_params)
    model_args["input_dim"] = hgraph.x.shape[1]
    model_args["output_dim"] = cfg.hgraph.num_classes
    model = get_model_class(cfg.model.model)(**model_args)
    model.to(device)
    print(model)


    train_stats = train_eval_loop(
        model=model,
        hgraph=hgraph,
        train_mask=hgraph.train_mask,
        val_mask=hgraph.val_mask,
        test_mask=hgraph.test_mask,
        lr=cfg.train.lr,
        num_epochs=cfg.train.num_epochs,
        printevery=cfg.train.printevery,
        verbose=cfg.train.verbose,
    )

    # train_stats = train_eval_loop_many(
    #     nruns=10,
    #     model_class=get_model_class(cfg.model.model),
    #     model_args=model_args,
    #     hgraph=hgraph,
    #     train_mask=hgraph.train_mask,
    #     val_mask=hgraph.val_mask,
    #     test_mask=hgraph.test_mask,
    #     lr=cfg.train.lr,
    #     num_epochs=cfg.train.num_epochs,
    #     verbose=cfg.train.verbose,
    #     device=device,
    # )


    # train_stats["config"] = cfg
    # add_hgraph_to_dict(train_stats, hgraph)


    # save_model(model, cfg)
    # save_stats(train_stats, cfg)
    save_stuff(cfg, train_stats, hgraph, model)




if __name__ == "__main__":
    main()
