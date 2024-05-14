# %%

import numpy as np
import torch
from pathlib import Path
from easydict import EasyDict
from train import get_single_run, eval
from learn_mask import run_experiment

# %%


motif_type = None
path = Path('train_results/randhouse_v3/allsettransformer/hgraph0/run0')
# path = Path('train_results/coauthor_cora/allsettransformer/run1')
load_best = True

cfg_model, train_stats, hgraph, model = get_single_run(path, device=torch.device("cpu"), load_best=load_best)

print(cfg_model)

print(f"train acc {eval(hgraph, model, hgraph.train_mask):.3f} | val acc {eval(hgraph, model, hgraph.val_mask):.3f}")

# %%

# import sys
# del sys.modules['learn_mask'], sys.modules['explain'], sys.modules['explain.learn_mask']
# from learn_mask import run_experiment


cfg = EasyDict(
    seed = 0,
    node_idx = 606,
    num_expansions = cfg_model.All_num_layers,
    method = "learn_mask",
    log_wandb = False,
    thresh_num = 10,
    compute_complement = True,
    motif = None,
    loss_pred_type = "kl_div",
    coeffs = {'size': 0.005, 'ent': 0.01},

    init_strategy = "const",
    num_epochs = 400,
    lr = 0.01,
    sample_with = "gumbel_softmax",
    tau = 1.0,
)

summary = run_experiment(cfg, cfg_model, hgraph, model)


# %%

print("G.T.", summary['explanation']['gt_class'])
print("Pred", summary['explanation']['pred_class'])

print("Expl pred", np.argmax(summary['explanation']['classprob/post']))
print(f"{summary['explanation']['loss_pred/post']:.4f}")
print(f"{summary['explanation']['loss_size/post']:.4f}")
print(summary['explanation']['incidence_dict/post'])

print("Compl pred", np.argmax(summary['complement']['classprob/post']))
print(f"{summary['complement']['loss_pred/post']:.4f}")
print(f"{summary['complement']['loss_size/post']:.4f}")
print(summary['complement']['incidence_dict/post'])

# %%