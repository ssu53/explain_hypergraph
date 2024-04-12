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
from explain import get_local_hypergraph, transfer_features, explainer_loss, get_human_motif, hgnn_explain_sparse, show_learnt_subgraph
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
        init_strategy="const", 
        num_epochs=cfg.num_epochs, 
        lr=cfg.lr, 
        loss_pred_type=cfg.loss_pred_type,
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
    hgraph_expl = show_learnt_subgraph(hgraph_local, thresh=0.5, node_to_include=None, cfg=cfg_model)
    transfer_features(hgraph, hgraph_expl, cfg_model)
    summary_expl = get_summary_nogt(cfg, hgraph, hgraph_local, hgraph_expl, model)

    # complement subgraph
    hgraph_local.H = torch.nan_to_num(1. - torch.tensor(H_learnt), nan=0.0) # get the complementary
    hgraph_local.norm = 1. - hgraph_local.norm
    hgraph_compl = show_learnt_subgraph(hgraph_local, thresh=0.5, node_to_include=None, cfg=cfg_model)
    transfer_features(hgraph, hgraph_compl, cfg_model)
    summary_compl = get_summary_nogt(cfg, hgraph, hgraph_local, hgraph_compl, model)

    summary = {'explanation': summary_expl, 'complement': summary_compl}

    # summary = print_summary(cfg, cfg_model, hgraph, hgraph_local, hgraph_expl, model)
    # summary = get_summary(cfg, cfg_model, hgraph, hgraph_local, hgraph_expl, model)


    if cfg.log_wandb:
        wandb.finish()
        
    return summary


@torch.no_grad()
def get_summary_nogt(config, hgraph, hgraph_local, hgraph_expl, model):

    node_idx = config.node_idx
    node_class = hgraph.y[node_idx].item()

    SUMMARY = {}

    # -------------------------------------------------
    # original graph
    logits_target = model(hgraph)[node_idx]
    pred_target = logits_target.softmax(dim=-1)


    # -------------------------------------------------
    # local computational graph, fractionally-relaxed

    logits_actual = model(hgraph_local)[hgraph_local.node_to_ind[node_idx]]
    pred_actual = logits_actual.softmax(dim=-1)


    loss, loss_pred, loss_size, loss_mask_ent = explainer_loss(
        hgraph_local.norm,
        pred_actual,
        pred_target,
        pred_target.argmax().item(),
        loss_pred_type=config.loss_pred_type,
        coeffs=config.coeffs,
    )

    SUMMARY.update({
        'loss/relaxed': loss.item(),
        'loss_pred/relaxed': loss_pred.item(),
        'loss_size/relaxed': loss_size.item(),
        'loss_mask_ent/relaxed': loss_mask_ent.item(),
        'classprob/relaxed': pred_actual.tolist(),
    })

    # -------------------------------------------------
    # learnt explanation subgraph, sharpened with thresh=0.5

    if node_idx in hgraph_expl.node_to_ind:
        logits_expl = model(hgraph_expl)[hgraph_expl.node_to_ind[node_idx]]
        pred_expl = logits_expl.softmax(dim=-1)
    else:
        pred_expl = torch.ones_like(pred_actual) * np.nan
    
    assert torch.allclose(hgraph_expl.norm, torch.ones_like(hgraph_expl.norm)) # since only kept the 1s

    loss, loss_pred, loss_size, loss_mask_ent = explainer_loss(
        hgraph_expl.norm * (1.0 - 1e-6), # to compute non-nan loss_mask_ent
        pred_expl,
        pred_target,
        pred_target.argmax().item(),
        loss_pred_type=config.loss_pred_type,
        coeffs=config.coeffs,
    )

    SUMMARY.update({
        'loss/binarised': loss.item(),
        'loss_pred/binarised': loss_pred.item(),
        'loss_size/binarised': loss_size.item(),
        'loss_mask_ent/binarised': loss_mask_ent.item(),
        'classprob/binarised': pred_expl.tolist(),
    })

    return SUMMARY


@torch.no_grad()
def get_summary(config, cfg_model, hgraph, hgraph_local, hgraph_expl, model):

    node_idx = config.node_idx
    node_class = hgraph.y[node_idx].item()

    SUMMARY = {}

    # -------------------------------------------------
    # original graph
    logits_target = model(hgraph)[node_idx]
    pred_target = logits_target.softmax(dim=-1)

    # -------------------------------------------------
    # human-selected graph

    hgraph_selected = get_human_motif(node_idx, hgraph, cfg_model, config.motif)
    logits_selected = model(hgraph_selected)[hgraph_selected.node_to_ind[node_idx]]
    pred_selected = logits_selected.softmax(dim=-1)

    loss, loss_pred, loss_size, loss_mask_ent = explainer_loss(
        hgraph_selected.norm * (1.0 - 1e-6), # to compute non-nan loss_mask_ent
        pred_selected,
        pred_target,
        pred_target.argmax().item(),
        loss_pred_type=config.loss_pred_type,
        coeffs=config.coeffs,
    )

    SUMMARY.update({
        'loss/human': loss.item(),
        'loss_pred/human': loss_pred.item(),
        'loss_size/human': loss_size.item(),
        'loss_mask_ent/human': loss_mask_ent.item(),
        'classprob/human': pred_selected[node_class].item(),
        'classprob/0/human': pred_selected[0].item(),
        'classprob/1/human': pred_selected[1].item(),
        'classprob/2/human': pred_selected[2].item(),
        'classprob/3/human': pred_selected[3].item(),
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
        loss_pred_type=config.loss_pred_type,
        coeffs=config.coeffs,
    )

    SUMMARY.update({
        'loss/relaxed': loss.item(),
        'loss_pred/relaxed': loss_pred.item(),
        'loss_size/relaxed': loss_size.item(),
        'loss_mask_ent/relaxed': loss_mask_ent.item(),
        'classprob/relaxed': pred_actual[node_class].item(),
        'classprob/0/relaxed': pred_actual[0].item(),
        'classprob/1/relaxed': pred_actual[1].item(),
        'classprob/2/relaxed': pred_actual[2].item(),
        'classprob/3/relaxed': pred_actual[3].item(),
    })

    # -------------------------------------------------
    # learnt explanation subgraph, sharpened with thresh=0.5

    if node_idx in hgraph_expl.node_to_ind:
        logits_expl = model(hgraph_expl)[hgraph_expl.node_to_ind[node_idx]]
        pred_expl = logits_expl.softmax(dim=-1)
    else:
        pred_expl = torch.ones_like(pred_actual) * np.nan
    
    assert torch.allclose(hgraph_expl.norm, torch.ones_like(hgraph_expl.norm)) # since only kept the 1s

    loss, loss_pred, loss_size, loss_mask_ent = explainer_loss(
        hgraph_expl.norm * (1.0 - 1e-6), # to compute non-nan loss_mask_ent
        pred_expl,
        pred_target,
        pred_target.argmax().item(),
        loss_pred_type=config.loss_pred_type,
        coeffs=config.coeffs,
    )

    SUMMARY.update({
        'loss/binarised': loss.item(),
        'loss_pred/binarised': loss_pred.item(),
        'loss_size/binarised': loss_size.item(),
        'loss_mask_ent/binarised': loss_mask_ent.item(),
        'classprob/binarised': pred_expl[node_class].item(),
        'classprob/0/binarised': pred_expl[0].item(),
        'classprob/1/binarised': pred_expl[1].item(),
        'classprob/2/binarised': pred_expl[2].item(),
        'classprob/3/binarised': pred_expl[3].item(),
    })

    return SUMMARY



@torch.no_grad()
def print_summary(config, cfg_model, hgraph, hgraph_local, hgraph_expl, model):

    node_idx = config.node_idx
    node_class = hgraph.y[node_idx].item()
    coeffs = config.coeffs
    wandb_config = config.wandb if config.log_wandb else None


    print(f"explaining {node_idx}")
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
        hgraph_selected.norm * (1.0 - 1e-6), # to compute non-nan loss_mask_ent
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
    loss_pred_human = loss_pred
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
            'loss_pred/relaxed': loss_pred / loss_pred_human,
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
        hgraph_expl.norm * (1.0 - 1e-6), # to compute non-nan loss_mask_ent
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
            'loss_pred/binarised': loss_pred / loss_pred_human,
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

    # config = EasyDict(
    #     load_fn = 'train_results/alldeepsets/unperturbed_v3/hgraph0_rerun1',
    #     node_idx = [580],
    #     coeffs = EasyDict(
    #         size = [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0],
    #         ent = [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0],
    #     ),
    #     lr=0.01,
    #     num_epochs=400,
    #     scheduler = None,
    #     seed = 42,
    #     log_wandb = True,
    #     wandb = EasyDict(
    #         entity = "ssu53",
    #         project = "hgraph_mask_coeffs_class3",
    #         experiment_name = "",
    #     ),
    # )

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


# @hydra.main(version_base=None, config_path="configs", config_name="learn_mask_zoo")
# def learn_mask_for_all_nodes(config : DictConfig) -> None:
def learn_mask_for_all_nodes():
        
        config = EasyDict(
            load_fn = 'train_results/zoo/allsettransformer/run1',
            load_best = False,
            coeffs = EasyDict(size=0.0005, ent= 0.01),
            num_expansions = 1,
            lr = 0.01,
            num_epochs = 800,
            loss_pred_type = 'kl_div',
            scheduler = None,
            seed = 42,
            log_wandb = False,
        )

        print(config)

        SUMMARY = {}

        cfg_model, hgraph, model = load_stuff(config)

        pbar = tqdm(
            total=hgraph.number_of_nodes(),
            desc=f"Learning masks...",
            disable=False,
        )

        for node_idx in hgraph.nodes():

            config.node_idx = node_idx
            summary = run_experiment(config, cfg_model, hgraph, model)
            SUMMARY[node_idx] = summary

            pbar.update(1)

            if node_idx % 25 == 0:
                with open('explanation.json', 'w') as f: 
                    json.dump({"config": config, "summary": SUMMARY}, f, indent=4)
        
        config.node_idx = None

        with open('explanation.json', 'w') as f: 
            json.dump({"config": config, "summary": SUMMARY}, f, indent=4)
# %%

if __name__ == "__main__":
    # main()
    learn_mask_for_all_nodes()

# %%
