# %%

import numpy as np
import torch
import hypernetx as hnx
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from functools import partial
from pathlib import Path
import hydra
from omegaconf import DictConfig, OmegaConf
from easydict import EasyDict
import wandb
import json
from tqdm import tqdm

from train import get_single_run, eval, set_seed
from explain import get_local_hypergraph, transfer_features, explainer_loss, get_human_motif, hgnn_explain_sparse, get_learnt_subgraph
import models.allset

# %%


def load_stuff(cfg):

    path = Path(cfg.load_fn)
    cfg, train_stats, hgraph, model = get_single_run(path, torch.device('cpu'), cfg.load_best)

    print(f"train acc {eval(hgraph, model, hgraph.train_mask):.3f} | val acc {eval(hgraph, model, hgraph.val_mask):.3f}")

    return cfg, hgraph, model



def run_experiment(cfg, cfg_model, hgraph, model):

    set_seed(cfg.seed)

    hgraph_local = get_local_hypergraph(idx=cfg.node_idx, hgraph=hgraph, num_expansions=cfg.num_expansions, is_hedge_concept=False)
    transfer_features(hgraph, hgraph_local, cfg_model)


    if cfg.log_wandb:
        node_class = hgraph.y[cfg.node_idx].item()
        wandb.init(
            project=cfg.wandb.project,
            entity=cfg.wandb.entity,
            config=cfg,
            group=f"{cfg.wandb.experiment_name}-size-{cfg.coeffs.size}-ent-{cfg.coeffs.ent}-class-{node_class}",
        )
        wandb.run.name = f"{cfg.wandb.experiment_name}-size-{cfg.coeffs.size}-ent-{cfg.coeffs.ent}-class-{node_class}-node-{cfg.node_idx}"


    hgnn_explain_sparse(
        cfg.node_idx, 
        hgraph_local, 
        model, 
        init_strategy=cfg.init_strategy, 
        num_epochs=cfg.num_epochs, 
        lr=cfg.lr, 
        loss_pred_type=cfg.loss_pred_type,
        sample_with=cfg.sample_with,
        tau=cfg.tau,
        hgraph_full=hgraph,
        coeffs=cfg.coeffs,
        # scheduler_fn=partial(torch.optim.lr_scheduler.CosineAnnealingLR, T_max=50),
        # scheduler_fn=partial(torch.optim.lr_scheduler.CosineAnnealingWarmRestarts, T_0=100, T_mult=1),
        # scheduler_fn=partial(torch.optim.lr_scheduler.ExponentialLR, gamma=0.99),
        verbose=False,
        wandb_config=cfg.wandb if cfg.log_wandb else None,
    )


    if isinstance(model, models.allset.models.SetGNN): 
        
        H_learnt = torch.zeros(hgraph_local.shape)
        H_learnt = np.where(hgraph_local.H_unmasked == 1.0, H_learnt, np.nan)
        for i in range(hgraph_local.edge_index.size(1)):
            ind1 = hgraph_local.edge_index[0,i]
            ind2 = hgraph_local.edge_index[1,i]
            assert not np.isnan(H_learnt[ind1,ind2])
            H_learnt[ind1,ind2] = hgraph_local.norm[i]

        # populate this into hgraph_local.H
        hgraph_local.H = torch.nan_to_num(torch.tensor(H_learnt), nan=0.0)

    else:

        H_learnt = hgraph_local.H.detach().cpu()
        H_learnt = np.where(hgraph_local.H_unmasked == 1.0, H_learnt, np.nan)

    
    # explanation subgraph
    hgraph_expl = get_learnt_subgraph(hgraph, hgraph_local, thresh=0.5, cfg=cfg_model, node_idx=cfg.node_idx, component_only=True)


    if cfg.log_wandb:
        assert cfg.compute_complement is None
        summary = print_summary(cfg, cfg_model, hgraph, hgraph_local, hgraph_expl, model)
        wandb.finish()
    
    else:
        
        if cfg.compute_complement:

            # summary for explanation subgraph
            summary_expl = get_summary(cfg, cfg_model, hgraph, hgraph_local, hgraph_expl, model)

            # summary for complement subgraph
            # (this corrupts hgraph_local and does not restore it)
            hgraph_local.H = torch.nan_to_num(1. - torch.tensor(H_learnt), nan=0.0)
            hgraph_local.norm = 1. - hgraph_local.norm
            hgraph_compl = get_learnt_subgraph(hgraph, hgraph_local, thresh=0.5, cfg=cfg_model, node_idx=cfg.node_idx, component_only=True)
            summary_compl = get_summary(cfg, cfg_model, hgraph, hgraph_local, hgraph_compl, model)

            summary = {'explanation': summary_expl, 'complement': summary_compl}
        
        else:

            # summary for explanation subgraph
            summary = get_summary(cfg, cfg_model, hgraph, hgraph_local, hgraph_expl, model)
        
    return summary


@torch.no_grad()
def get_summary(cfg, cfg_model, hgraph, hgraph_local, hgraph_expl, model):

    node_idx = cfg.node_idx
    node_class = hgraph.y[node_idx].item()

    SUMMARY = {}
    SUMMARY.update({
        'gt_class': node_class,
    })

    # -------------------------------------------------
    # original graph
    logits_target = model(hgraph)[node_idx]
    pred_target = logits_target.softmax(dim=-1)
    pred_class = pred_target.argmax().item()

    SUMMARY.update({
        'pred_class': pred_class,
    })

    # -------------------------------------------------
    # human-selected graph

    if cfg.motif is not None:
        hgraph_selected = get_human_motif(node_idx, hgraph, cfg_model, cfg.motif)
        logits_selected = model(hgraph_selected)[hgraph_selected.node_to_ind[node_idx]]
        pred_selected = logits_selected.softmax(dim=-1)

        loss, loss_pred, loss_size, loss_mask_ent = explainer_loss(
            hgraph_selected.norm,
            pred_selected,
            pred_target,
            pred_target.argmax().item(),
            loss_pred_type=cfg.loss_pred_type,
            coeffs=cfg.coeffs,
        )

        SUMMARY.update({
            'loss/human': loss.item(),
            'loss_pred/human': loss_pred.item(),
            'loss_size/human': loss_size.item(),
            'loss_mask_ent/human': loss_mask_ent.item(),
            'classprob/human': pred_selected.tolist(),
        })


    # -------------------------------------------------
    # local computational graph, fractionally-relaxed

    logits_actual = model(hgraph_local)[hgraph_local.node_to_ind[node_idx]]
    pred_actual = logits_actual.softmax(dim=-1)


    loss, loss_pred, loss_size, loss_mask_ent = explainer_loss(
        hgraph_local.norm,
        pred_actual,
        pred_target,
        pred_target.argmax().item(),
        loss_pred_type=cfg.loss_pred_type,
        coeffs=cfg.coeffs,
    )

    SUMMARY.update({
        'loss/raw': loss.item(),
        'loss_pred/raw': loss_pred.item(),
        'loss_size/raw': loss_size.item(),
        'loss_mask_ent/raw': loss_mask_ent.item(),
        'classprob/raw': pred_actual.tolist(),
    })

    # -------------------------------------------------
    # learnt explanation subgraph, post-processed (sharpened with thresh=0.5 and taking the connected component)

    assert node_idx in hgraph_expl.node_to_ind

    if node_idx in hgraph_expl.node_to_ind:
        logits_expl = model(hgraph_expl)[hgraph_expl.node_to_ind[node_idx]]
    else:
        logits_expl = torch.ones_like(logits_actual) * np.nan
    pred_expl = logits_expl.softmax(dim=-1)
    
    assert torch.allclose(hgraph_expl.norm, torch.ones_like(hgraph_expl.norm)) or torch.allclose(hgraph_expl.norm, torch.tensor([0])) 


    loss, loss_pred, loss_size, loss_mask_ent = explainer_loss(
        hgraph_expl.norm,
        pred_expl,
        pred_target,
        pred_target.argmax().item(),
        loss_pred_type=cfg.loss_pred_type,
        coeffs=cfg.coeffs,
    )

    if torch.allclose(hgraph_expl.norm, torch.tensor([0])):
        incidence_dict = {}
    else:
        incidence_dict = hgraph_expl.incidence_dict

    SUMMARY.update({
        'loss/post': loss.item(),
        'loss_pred/post': loss_pred.item(),
        'loss_size/post': loss_size.item(),
        'loss_mask_ent/post': loss_mask_ent.item(),
        'classprob/post': pred_expl.tolist(),
        'activ_node/post': model.activ_node[hgraph_expl.node_to_ind[node_idx]].detach().cpu().tolist(),
        'incidence_dict/post': incidence_dict,
    })
    

    return SUMMARY



@torch.no_grad()
def print_summary(config, cfg_model, hgraph, hgraph_local, hgraph_expl, model):

    node_idx = config.node_idx
    node_class = hgraph.y[node_idx].item()
    coeffs = config.coeffs
    wandb_config = config.wandb if config.log_wandb else None


    print(f"explaining... Node {node_idx} | G.T. Class {node_class}")
    print()

    # -------------------------------------------------
    print("original graph")
    logits_target = model(hgraph)[node_idx]
    pred_target = logits_target.softmax(dim=-1)
    print("class probs", torch.round(pred_target.detach(), decimals=3))
    print()


    # -------------------------------------------------
    print("human-selected graph")
    hgraph_selected = get_human_motif(node_idx, hgraph, cfg_model, config.motif)
    logits_selected = model(hgraph_selected)[hgraph_selected.node_to_ind[node_idx]]
    pred_selected = logits_selected.softmax(dim=-1)
    print("class probs", torch.round(pred_selected.detach(), decimals=3))
    
    loss, loss_pred, loss_size, loss_mask_ent = explainer_loss(
        hgraph_selected.norm,
        pred_selected,
        pred_target,
        pred_target.argmax().item(),
        loss_pred_type=config.loss_pred_type,
        coeffs=coeffs,
    )
    print(f"{loss=:.3f} {loss_pred=:.3f} {loss_size=:.3f} {loss_mask_ent=:.3f}")
    print()

    if wandb_config is not None:
        wandb.log({
            'loss/human': loss,
            'loss_pred/human': loss_pred,
            'loss_size/human': loss_size,
            'loss_mask_ent/human': loss_mask_ent,
            'classprob/human': pred_selected[node_class],
        })
    
    loss_human = loss
    loss_size_human = loss_size
    loss_mask_ent_human = loss_mask_ent

    # -------------------------------------------------
    print("local computational graph, fractionally-relaxed")
    logits_actual = model(hgraph_local)[hgraph_local.node_to_ind[node_idx]]
    pred_actual = logits_actual.softmax(dim=-1)
    print("class probs", torch.round(pred_actual, decimals=3))

    loss, loss_pred, loss_size, loss_mask_ent = explainer_loss(
        hgraph_local.norm,
        pred_actual,
        pred_target,
        pred_target.argmax().item(),
        loss_pred_type=config.loss_pred_type,
        coeffs=coeffs,
    )
    print(f"{loss=:.3f} {loss_pred=:.3f} {loss_size=:.3f} {loss_mask_ent=:.3f}")
    print()

    if wandb_config is not None:
        wandb.log({
            'loss/relaxed': loss / loss_human,
            'loss_pred/relaxed': loss_pred,
            'loss_size/relaxed': loss_size / loss_size_human,
            'loss_mask_ent/relaxed': loss_mask_ent / loss_mask_ent_human,
            'classprob/relaxed': pred_actual[node_class],
        })

    # -------------------------------------------------
    print("learnt explanation subgraph, sharpened with thresh=0.5")
    if node_idx in hgraph_expl.node_to_ind:
        logits_expl = model(hgraph_expl)[hgraph_expl.node_to_ind[node_idx]]
        pred_expl = logits_expl.softmax(dim=-1)
    else:
        pred_expl = torch.ones_like(pred_actual) * np.nan
    print("class probs", torch.round(pred_expl, decimals=3))
    

    assert torch.allclose(hgraph_expl.norm, torch.ones_like(hgraph_expl.norm)) # since only kept the 1s

    loss, loss_pred, loss_size, loss_mask_ent = explainer_loss(
        hgraph_expl.norm,
        pred_expl,
        pred_target,
        pred_target.argmax().item(),
        loss_pred_type=config.loss_pred_type,
        coeffs=coeffs,
    )
    print(f"{loss=:.3f} {loss_pred=:.3f} {loss_size=:.3f} {loss_mask_ent=:.3f}")
    print()

    if wandb_config is not None:
        wandb.log({
            'loss/binarised': loss / loss_human,
            'loss_pred/binarised': loss_pred,
            'loss_size/binarised': loss_size / loss_size_human,
            'loss_mask_ent/binarised': loss_mask_ent / loss_mask_ent_human,
            'classprob/binarised': pred_expl[node_class],
        })
        

    if wandb_config is not None:

        fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(12,4))
        fig.suptitle(f"Node {node_idx} | G.T. Class {node_class}")
            
        hnx.draw(hgraph_local, with_node_labels=True, ax=ax[0])
        ax[0].set_title("local computational graph")
        hnx.draw(hgraph_selected, with_node_labels=True, ax=ax[1])
        ax[1].set_title("human-selected graph")
        hnx.draw(hgraph_expl, with_node_labels=True, ax=ax[2])
        ax[2].set_title("learnt explanation graph" + " (LOST NODE)" if torch.any(torch.isnan(pred_expl)).item() else "learnt explanation graph")

        # plt.show()

        wandb.log({"hgraph": wandb.Image(plt)})
    
    return None



@hydra.main(version_base=None, config_path="configs", config_name="learn_mask")
def main(config : DictConfig) -> None:

    print(OmegaConf.to_yaml(config))

    node_idxs = config.node_idx
    coeffs_size = config.coeffs.size
    coeffs_ent = config.coeffs.ent

    print(config)

    cfg_model, hgraph, model = load_stuff(config)

    for node_idx in node_idxs:
        for coeff_size in coeffs_size:
            for coeff_ent in coeffs_ent:

                cfg = EasyDict(config.copy())
                cfg.node_idx = node_idx
                cfg.coeffs.size = coeff_size
                cfg.coeffs.ent = coeff_ent

                print(cfg)
                
                run_experiment(cfg, cfg_model, hgraph, model)


@hydra.main(version_base=None, config_path="configs", config_name="learn_mask_randhouse")
def main2(config : DictConfig) -> None:

        cfg = EasyDict(config)
        cfg.coeffs = EasyDict(cfg.coeffs) # hack for JSON serialisation

        print(cfg)

        with open(cfg.save_fn, 'w') as f: 
            json.dump({"config": cfg, "summary": None}, f, indent=4)


        # -------------------------------------------------
        # load stuff
        cfg_model, hgraph, model = load_stuff(cfg)


        # -------------------------------------------------
        # get nodes to be explained

        if cfg.node_samples is None:
            node_idxs = hgraph.nodes()
        else:
            node_idxs = np.random.choice(
                hgraph.nodes(), size=cfg.node_samples, replace=False)
            node_idxs = sorted(node_idxs)
        if cfg.node_idxs is not None:
            node_idxs = cfg.node_idxs
        
        # -------------------------------------------------
        # run experiment for each of the nodes

        SUMMARY = {}

        pbar = tqdm(
            total=hgraph.number_of_nodes(),
            desc=f"Learning masks...",
            disable=False,
        )

        for node_idx in node_idxs:

            cfg.node_idx = node_idx
            summary = run_experiment(cfg, cfg_model, hgraph, model)
            SUMMARY[node_idx] = summary

            pbar.update(1)

            if node_idx % cfg.save_every == 0:
                with open(cfg.save_fn, 'w') as f: 
                    json.dump({"config": cfg, "summary": SUMMARY}, f, indent=4)
        
        # -------------------------------------------------
        # save outputs

        cfg.node_idx = None
        
        with open(cfg.save_fn, 'w') as f: 
            json.dump({"config": cfg, "summary": SUMMARY}, f, indent=4)
# %%

if __name__ == "__main__":
    # main()
    main2()

# %%
