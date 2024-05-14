# %%

import numpy as np
import torch
from pathlib import Path
import hypernetx as hnx

import pandas as pd
import matplotlib.pyplot as plt 
import seaborn as sns
import json
from easydict import EasyDict

import models.allset
from train import get_single_run, eval, set_seed
from explain import plot_concepts, ActivationClassifier, plot_samples, get_local_hypergraph, get_hyperedge_labels, transfer_features, hgnn_explain, hgnn_explain_sparse, get_learnt_subgraph, explainer_loss, get_human_motif


# %%


# # HyperResidGCN
# path = Path('train_results/hyperresidgcn/standard/ones1/hgraph0')
# path = Path('train_results/hyperresidgcn/standard_v2/hgraph0')
# path = Path('train_results/hyperresidgcn/standard_v3/hgraph0')

# HyperResidGCN with LEN
# path = Path('train_results/hyperresidgcn_len/standard/ones1/hgraph0')
# path = Path('train_results/hyperresidgcn_len/standard_v2/hgraph0')
# path = Path('train_results/hyperresidgcn_len/standard_v3/hgraph0')

# HyperResidGCN
# path = Path('train_results/hyperresidgcn_sas/standard/ones1/hgraph0')
# path = Path('train_results/hyperresidgcn_sas/standard_v2/hgraph0')
# path = Path('train_results/hyperresidgcn_sas/standard_v3/hgraph0')

# allsettransformer
# path = Path('train_results/allsettransformer/standard/ones1/hgraph0')
# path = Path('train_results/allsettransformer/standard_v2/hgraph0')
# path = Path('train_results/allsettransformer/standard_v3/hgraph0')
# path = Path('train_results/allsettransformer/unperturbed_v3/hgraph0')
# motif_type = 'house'
# load_best = True

# alldeepsets
# path = Path('train_results/alldeepsets/unperturbed_v3/hgraph0_rerun1')

# randhouse
# motif_type = 'house'
# path = Path('train_results/randhouse_v3/allsettransformer/hgraph0/run3')
# load_best = True

# communityhouse
# motif_type = 'house'
# path = Path('train_results/commhouse_v1/allsettransformer/hgraph0/run0')
# load_best = True

# treecycle
# motif_type = 'cycle'
# path = Path('train_results/treecycle_v3/allsettransformer/hgraph0/run0')
# load_best = True

# treegrid
motif_type = 'grid'
path = Path('train_results/treegrid_v3/allsettransformer/hgraph0/run0')
load_best = True

# zoo
# motif_type = None
# path = Path('train_results/zoo/allsettransformer/run1')
# load_best = False

# coauthor-cora
# motif_type = None
# path = Path('train_results/coauthor_cora/allsettransformer/run1')
# load_best = True


# rashomon
# motif_type = 'house'
# path = Path('train_results/rashomon/randhouse_v0/allsettransformer/hgraph0/run7')
# # path2 = Path('train_results/randhouse_v3/allsettransformer/hgraph0/run0')
# load_best = False

cfg, train_stats, hgraph, model = get_single_run(path, device=torch.device("cpu"), load_best=load_best)
# _, _, hgraph, _ = get_single_run(path2, device=torch.device("cpu"), load_best=load_best)

print(cfg)
print(f"num nodes {hgraph.n_x} | num edges originally {hgraph.num_hyperedges} | total edges {hgraph.totedges if hasattr(hgraph, 'totedges') else None}")


print(f"train acc {eval(hgraph, model, hgraph.train_mask):.3f} | val acc {eval(hgraph, model, hgraph.val_mask):.3f}")

# %%

# Fetch concepts

with torch.no_grad():
    _ = model(hgraph)
    pred_target = model(hgraph).argmax(dim=-1)

activ_node = model.activ_node.detach().cpu()
activ_hedge = model.activ_hedge.detach().cpu()
activ_node_agg = torch.tensor(hgraph.incidence_matrix().T @ activ_node)

hyperedge_labels = get_hyperedge_labels(hgraph)


class_label = hgraph.y
class_label_name = class_label

kmeans_model_node, _ = plot_concepts(activ_node, labels=class_label_name, num_clusters=7, cluster_by_binarise=False, fig_title="Nodes")

ac = ActivationClassifier(
    activ_node, kmeans_model_node, "decision_tree",
    hgraph.x.cpu().reshape(-1,1), class_label, 
    hgraph.train_mask.cpu(), hgraph.val_mask.cpu())
print(f"Concept completeness on classes: {ac.get_classifier_accuracy():.3f}")


print(f"train acc {eval(hgraph, model, hgraph.train_mask):.3f} | val acc {eval(hgraph, model, hgraph.val_mask):.3f}")


# %%

with open('explanation_treecycle_v0.json', 'r') as f:
    summary = json.load(f)
    summary = EasyDict(summary)

assert summary.config.load_fn == str(path)
assert summary.config.load_best == load_best
df = pd.DataFrame.from_dict(summary.summary, orient="index")
df = df.reindex(sorted(df.columns), axis=1)
df.index = [int(i)for i in df.index]

col = 'loss_pred/post'

import sys
del sys.modules['explain'], sys.modules['explain.vis_utils']
from explain import plot_concepts

kmeans_model_node, pca_model_node = plot_concepts(activ_node, labels=df[col].tolist(), categorical_label=False, num_clusters=7, cluster_by_binarise=False, fig_title=f"Nodes by {col}")

plt.figure(figsize=(5,5))
plt.hist(df[col], bins=20)
plt.title(f"distribution of {col}")
plt.show()


# %%

node_idx = 117
incidence_dict = df['incidence_dict/post'][node_idx]
assert len(incidence_dict) > 0 # otherwise it is an isolated node, deal with
hexpl = hnx.Hypergraph(incidence_dict)
transfer_features(hgraph, hexpl, cfg, isolated_node=len(incidence_dict)==0)

hgraph_local = get_local_hypergraph(idx=node_idx, hgraph=hgraph, num_expansions=cfg.All_num_layers, is_hedge_concept=False)
transfer_features(hgraph, hgraph_local, cfg)

fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(8,4))
fig.suptitle(f"Node {node_idx} | G.T. Class {hgraph.y[node_idx].item()}")
    
hnx.draw(hgraph_local, with_node_labels=True, ax=ax[0])
ax[0].set_title("local computational graph")
hnx.draw(hexpl, with_node_labels=True, ax=ax[1])
ax[1].set_title("learnt explanation graph")
plt.show()

with torch.no_grad():
    _ = model(hexpl)[hexpl.node_to_ind[node_idx]]

assert torch.allclose(torch.tensor(df['activ_node/post'][node_idx]), model.activ_node[hexpl.node_to_ind[node_idx]])

# %%


activ_node_expl = [df['activ_node/post'][i] for i in range(len(df))]
activ_node_expl = torch.tensor(activ_node_expl)

activ_node_pca = pca_model_node.transform(activ_node)
activ_node_expl_pca = pca_model_node.transform(activ_node_expl)


plt.figure(figsize=(7,5))
plt.scatter(activ_node_pca[:,0], activ_node_pca[:,1], alpha=0.3, s=10)
plt.scatter(activ_node_expl_pca[:,0], activ_node_expl_pca[:,1], alpha=0.3, s=10)
plt.show()

dists = torch.norm(activ_node_expl - activ_node, p=2, dim=-1)
dists_pca = torch.norm(torch.tensor(activ_node_expl_pca) - torch.tensor(activ_node_pca), p=2, dim=-1)

print(f"{dists.shape=}")
print(f"orig: {dists.mean():.3f} +/- {dists.std():.3f}")
print(f"PCA'd: {dists_pca.mean():.3f} +/- {dists_pca.std():.3f}")

plt.figure(figsize=(3,3))
plt.hist(dists, bins=20)
plt.show()

# print(dists[[117, 235, 236, 533]])
# print(dists[[ 60, 256, 534]])
# print(dists[[590, 442,  12, 443, 182, 260, 531]])

# %%


activ_node_expl = [df['activ_node/post'][i] for i in range(len(df))]
activ_node_expl = torch.tensor(activ_node_expl)

condition_mask = (hgraph.y == 1)
condition_mask = hgraph.y
activ_node_expl_sub = activ_node_expl[condition_mask]
activ_node_sub = activ_node[condition_mask]

activ_node_pca = pca_model_node.transform(activ_node_sub)
activ_node_expl_pca = pca_model_node.transform(activ_node_expl_sub)


plt.figure(figsize=(7,5))
plt.scatter(activ_node_pca[:,0], activ_node_pca[:,1], alpha=0.3, s=10)
plt.scatter(activ_node_expl_pca[:,0], activ_node_expl_pca[:,1], alpha=0.3, s=10)
plt.show()


assert activ_node_expl.shape == activ_node.shape
assert activ_node_expl_sub.shape == activ_node_sub.shape

dists = torch.norm(activ_node_expl_sub - activ_node_sub, p=2, dim=-1)

plt.figure(figsize=(3,3))
plt.hist(dists, bins=20)
plt.show()

# %%

def get_subclass_labels(hgraph):
    # fine grained labels - this is only a hack when there are no edge perturbations

    subclass_label_name = [None for _ in range(hgraph.number_of_nodes())]
    
    for node_idx in range(hgraph.number_of_nodes()):

        node_class = hgraph.y[node_idx].item()

        if node_class == House.Base.value:
            neighb_classes = set([hgraph.y[neighb].item() for neighb in hgraph.neighbors(node_idx)])
            if House.Middle.value in neighb_classes:
                subclass_label_name[node_idx] = HouseGranular.Base_Anchor.name
            else:
                subclass_label_name[node_idx] = HouseGranular.Base_Other.name
        elif node_class == House.Top.value:
            subclass_label_name[node_idx] = HouseGranular.Top.name
        elif node_class == House.Middle.value:
            if len(hgraph.neighbors(node_idx)) == 4:
                subclass_label_name[node_idx] = HouseGranular.Middle_Unanchored.name
            elif len(hgraph.neighbors(node_idx)) == 5:
                subclass_label_name[node_idx] = HouseGranular.Middle_Anchored.name
            else:
                raise ValueError
        elif node_class == House.Bottom.value:
            subclass_label_name[node_idx] = HouseGranular.Bottom.name
        else:
            raise ValueError
    
    return subclass_label_name
    

if motif_type == 'house':
    from hgraph import House, HouseGranular
    class_label = hgraph.y
    # class_label_name = [House(c.item()).name for c in class_label]
    # subclass_label_name = get_subclass_labels(hgraph)
    # subclass_label = torch.tensor([getattr(HouseGranular, c).value for c in subclass_label_name])
    class_label_name = class_label

if motif_type == 'cycle':
    from hgraph import Cycle
    class_label = hgraph.y
    class_label_name = [Cycle(c.item()).name for c in class_label]

if motif_type == 'grid':
    from hgraph import Grid
    class_label = hgraph.y
    class_label_name = [Grid(c.item()).name for c in class_label]

if motif_type == None:
    class_label = hgraph.y
    class_label_name = class_label

# %%

# Plot node concepts by cluster

# kmeans_model_node, _ = plot_concepts(activ_node, labels=subclass_label_name, num_clusters=7, cluster_by_binarise=False, fig_title="Nodes (Subclassed)")

kmeans_model_node, _ = plot_concepts(activ_node, labels=class_label_name, num_clusters=10, cluster_by_binarise=False, fig_title="Nodes")

# kmeans_model_hedge, _ = plot_concepts(activ_hedge, labels=hyperedge_labels, num_clusters=7, cluster_by_binarise=False, fig_title="Hyperedges")

# kmeans_model_node_agg, _ = plot_concepts(activ_node_agg, labels=hyperedge_labels, num_clusters=7, cluster_by_binarise=False, fig_title="Nodes Agg. onto Hyperedges")

# %%

# Compute concept completeness

ac = ActivationClassifier(
    activ_node, kmeans_model_node, "decision_tree",
    hgraph.x.cpu().reshape(-1,1), class_label, 
    hgraph.train_mask.cpu(), hgraph.val_mask.cpu())
print(f"Concept completeness on classes: {ac.get_classifier_accuracy():.3f}")


# ac_subclass = ActivationClassifier(
#     activ_node, kmeans_model_node, "decision_tree",
#     hgraph.x.cpu().reshape(-1,1), subclass_label, 
#     hgraph.train_mask.cpu(), hgraph.val_mask.cpu())
# print(f"Concept completeness on subclasses: {ac_subclass.get_classifier_accuracy():.3f}")


print(f"train acc {eval(hgraph, model, hgraph.train_mask):.3f} | val acc {eval(hgraph, model, hgraph.val_mask):.3f}")


# %%

mispredictions_train = torch.argwhere((pred_target != class_label) & hgraph.train_mask).flatten()
mispredictions_val = torch.argwhere((pred_target != class_label) & hgraph.val_mask).flatten()

print(len(mispredictions_train) / hgraph.train_mask.sum().item())
print(len(mispredictions_val) / hgraph.val_mask.sum().item())

print(mispredictions_train)
print(mispredictions_val)

# %%

# Plot node concepts by binarise
if 'model' not in cfg or (not cfg.model.model_params.softmax_and_scale):
    print("Binarised concepts do not apply here.")
else:
    _, _ = plot_concepts(hgraph, model, num_clusters=None, cluster_by_binarise=True)


# %%
_, _ = plot_samples(activ_node, kmeans_model_node, hgraph.y, hgraph, num_expansions=min(cfg.All_num_layers, 2), num_nodes_view=3, is_hedge_concept=False)

# %%
# _, _ = plot_samples(activ_hedge, kmeans_model_hedge, hyperedge_labels, hgraph, num_expansions=2, num_nodes_view=3, is_hedge_concept=True)

# %%
# _, _ = plot_samples(activ_node_agg, kmeans_model_node_agg, hyperedge_labels, hgraph, num_expansions=2, num_nodes_view=3, is_hedge_concept=True)

# %%

# where would there be a misprediction if you only take the node itself?


from explain import get_edges_of_nodes

with torch.no_grad():
    pred_target = model(hgraph).argmax(dim=-1)

preds_node_only = torch.ones((hgraph.number_of_nodes(),), dtype=int) * -1

for node_idx in hgraph.nodes():
    hgraph_local = hnx.Hypergraph({
        get_edges_of_nodes(hgraph, [node_idx]).pop(): [node_idx],
    })
    transfer_features(hgraph, hgraph_local, cfg)
    with torch.no_grad():
        preds_node_only[node_idx] = model(hgraph_local).argmax(dim=-1)

    if node_idx > 150: break


print(node_idx)
print(torch.argwhere((preds_node_only != pred_target) & (preds_node_only != -1)))

# %%

# Select node to explain

node_idx = 591

set_seed(42)

hgraph_local = get_local_hypergraph(idx=node_idx, hgraph=hgraph, num_expansions=cfg.All_num_layers, is_hedge_concept=False)
transfer_features(hgraph, hgraph_local, cfg)

# hnx.draw(hgraph_neighb.collapse_nodes(), with_node_counts=True, with_node_labels=True)
hnx.draw(hgraph_local, with_node_labels=True)


# %%

import sys
del sys.modules['explain'], sys.modules['explain.sparsemax'], sys.modules['explain.learn_mask']
from explain import hgnn_explain_sparse

# may need to tune these dynamically depending on... the size of hgraph_local?
coeffs = {'size': 0.005, 'ent': 0.01} # for house, may change size of 0.0005 to get perfect pred

if isinstance(model, models.allset.models.SetGNN): 
    losses = hgnn_explain_sparse(
        node_idx, 
        hgraph_local, 
        model, 
        init_strategy="const",
        num_epochs=400,
        lr=0.01, 
        loss_pred_type="kl_div",
        print_every=25,
        hgraph_full=hgraph,
        coeffs=coeffs,
        # sample_with="sigmoid",
        # tau=None,
        sample_with="gumbel_softmax",
        tau=1.0,
        )
    plt.plot(losses)
else:
    hgnn_explain(
        node_idx, 
        hgraph_local, 
        model, 
        init_strategy="const", 
        num_epochs=200, 
        lr=0.01, 
        loss_pred_type="kl_div",
        print_every=25,
        hgraph_full=hgraph,
        coeffs=coeffs,
        )


# %%

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

# %%

df = pd.DataFrame(
    H_learnt,
    index=hgraph_local.ind_to_node.values(),
    columns=hgraph_local.ind_to_edge.values(),
    )

plt.figure(figsize=(5,3))
sns.heatmap(df, annot=True, cmap='viridis', fmt='.2f', cbar=False, annot_kws={"fontsize":8})

# %%

del sys.modules['explain'], sys.modules['explain.learn_mask']
from explain import get_learnt_subgraph

hgraph_expl = get_learnt_subgraph(hgraph, hgraph_local, thresh=0.5, cfg=cfg, node_idx=node_idx, component_only=True)

# %%

import sys
del sys.modules['explain'], sys.modules['explain.learn_mask']
from explain import get_human_motif

with torch.no_grad():

    print(f"explaining {node_idx}")
    print()

    # -------------------------------------------------
    print("original graph")
    logits_target = model(hgraph)[node_idx]
    pred_target = logits_target.softmax(dim=-1)
    print("class probs", torch.round(pred_target, decimals=3))
    print()


    # -------------------------------------------------
    # print("human-selected graph")

    # hgraph_selected = get_human_motif(node_idx, hgraph, cfg, motif_type)
    # logits_selected = model(hgraph_selected)[hgraph_selected.node_to_ind[node_idx]]
    # pred_selected = logits_selected.softmax(dim=-1)
    # print("class probs", torch.round(pred_selected, decimals=3))
    
    # loss, loss_pred, loss_size, loss_mask_ent = explainer_loss(
    #     hgraph_selected.norm,
    #     pred_selected,
    #     pred_target,
    #     pred_target.argmax().item(),
    #     loss_pred_type="kl_div",
    #     coeffs=coeffs,
    # )
    # print(f"{loss=:.3f} {loss_pred=:.3f} {loss_size=:.3f} {loss_mask_ent=:.3f}")
    # print()


    # -------------------------------------------------
    print("local computational graph, raw")
    logits_actual = model(hgraph_local)[hgraph_local.node_to_ind[node_idx]]
    pred_actual = logits_actual.softmax(dim=-1)
    print("class probs", torch.round(pred_actual, decimals=3))

    loss, loss_pred, loss_size, loss_mask_ent = explainer_loss(
        hgraph_local.norm,
        pred_actual,
        pred_target,
        pred_target.argmax().item(),
        loss_pred_type="kl_div",
        coeffs=coeffs,
    )
    print(f"{loss=:.3f} {loss_pred=:.3f} {loss_size=:.3f} {loss_mask_ent=:.3f}")
    print()


    # -------------------------------------------------
    print("learnt explanation subgraph, post-processed")
    if node_idx in hgraph_expl.node_to_ind:
        logits_expl = model(hgraph_expl)[hgraph_expl.node_to_ind[node_idx]]
        pred_expl = logits_expl.softmax(dim=-1)
    else:
        pred_expl = torch.ones_like(pred_actual) * np.nan
    print("class probs", torch.round(pred_expl, decimals=3))
    

    assert torch.allclose(hgraph_expl.norm, torch.ones_like(hgraph_expl.norm)) or torch.allclose(hgraph_expl.norm, torch.tensor([0])) 

    loss, loss_pred, loss_size, loss_mask_ent = explainer_loss(
        hgraph_expl.norm,
        pred_expl,
        pred_target,
        pred_target.argmax().item(),
        loss_pred_type="kl_div",
        coeffs=coeffs,
    )
    print(f"{loss=:.3f} {loss_pred=:.3f} {loss_size=:.3f} {loss_mask_ent=:.3f}")
    print()
    


fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(12,4))
fig.suptitle(f"Node {node_idx} | G.T. Class {hgraph.y[node_idx].item()}")
    
hnx.draw(hgraph_local, with_node_labels=True, ax=ax[0])
ax[0].set_title("local computational graph")
# hnx.draw(hgraph_selected, with_node_labels=True, ax=ax[1])
# ax[1].set_title("human-selected graph")
hnx.draw(hgraph_expl, with_node_labels=True, ax=ax[2])
ax[2].set_title("learnt explanation graph" + " (LOST NODE)" if torch.any(torch.isnan(pred_expl)).item() else "learnt explanation graph")

plt.show()
# %%


with torch.no_grad():

    print("local computational graph, fractionally-relaxed")
    logits_actual = model(hgraph_local)[hgraph_local.node_to_ind[node_idx]]
    pred_actual = logits_actual.softmax(dim=-1)
    print("class probs", torch.round(pred_actual, decimals=3))

    loss, loss_pred, loss_size, loss_mask_ent = explainer_loss(
        hgraph_local.norm,
        pred_actual,
        pred_target,
        pred_target.argmax().item(),
        loss_pred_type="kl_div",
        coeffs=coeffs,
    )
    print(f"{loss=:.3f} {loss_pred=:.3f} {loss_size=:.3f} {loss_mask_ent=:.3f}")
    print()


    print("local computational graph, binarised")
    hgraph_local_norm = hgraph_local.norm.clone()
    hgraph_local.norm = torch.round(hgraph_local_norm, decimals=0)
    logits_actual = model(hgraph_local)[hgraph_local.node_to_ind[node_idx]]
    pred_actual = logits_actual.softmax(dim=-1)
    print("class probs", torch.round(pred_actual, decimals=3))

    loss, loss_pred, loss_size, loss_mask_ent = explainer_loss(
        hgraph_local.norm,
        pred_actual,
        pred_target,
        pred_target.argmax().item(),
        loss_pred_type="kl_div",
        coeffs=coeffs,
    )
    print(f"{loss=:.3f} {loss_pred=:.3f} {loss_size=:.3f} {loss_mask_ent=:.3f}")
    print()
# %%
