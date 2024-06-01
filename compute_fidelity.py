# %%

import torch
from train import get_single_run
import json
from easydict import EasyDict
from pathlib import Path

import hypernetx as hnx
import matplotlib.pyplot as plt


def get_explanation_inds(explanation):
    return [int(i) for i in explanation.summary.keys()]

def get_explanation_statistic(explanation, key: str, inds, is_complement: bool = False):
    subhgraph = 'complement' if is_complement else 'explanation'
    stat = [explanation.summary[str(i)][subhgraph][key] for i in inds]
    if 'incidence_dict' in key: return stat
    stat = torch.tensor(stat, dtype=float)
    return stat

def fidelity(p, p_ref, dissimilarity_func):
    if p.ndim == 1: p = p.unsqueeze(1)
    if p_ref.ndim == 1: p = p.unsqueeze(1)
    return dissimilarity_func(p, p_ref)

def dissimilar_by_accuracy(p, p_ref):
    assert p.ndim == p_ref.ndim == 2
    assert p.shape == p_ref.shape
    return (1 - (p.argmax(dim=-1) == p_ref.argmax(dim=-1)).to(float)).mean().item()

def dissimilar_by_kl(p, p_ref):
    assert p.ndim == p_ref.ndim == 2
    assert p.shape == p_ref.shape
    return (p_ref * (p_ref / p).log()).sum(dim=1).mean(dim=0).item()

def dissimilar_by_kl_norm(p, p_ref):
    assert p.ndim == p_ref.ndim == 2
    assert p.shape == p_ref.shape
    return (1 - (- p_ref * (p_ref / p).log()).exp()).sum(dim=1).mean(dim=0).item()

def dissimilar_by_tv(p, p_ref):
    assert p.ndim == p_ref.ndim == 2
    assert p.shape == p_ref.shape
    return 0.5 * (p - p_ref).abs().sum(dim=1).mean(dim=0).item()

def dissimilar_by_xent(p, p_ref):
    assert p.ndim == p_ref.ndim == 2
    assert p.shape == p_ref.shape
    return -(p_ref * p.log()).sum(dim=1).mean(dim=0).item()

def dissimilar_by_xent_norm(p, p_ref):
    assert p.ndim == p_ref.ndim == 2
    assert p.shape == p_ref.shape
    return (1 - (p_ref * p.log()).exp()).sum(dim=1).mean(dim=0).item()


# %%

path = Path('train_results/FINAL/cora')
load_best = True

cfg_model, _, hgraph, model = get_single_run(path, device=torch.device("cpu"), load_best=load_best)

with torch.no_grad():
    logits = model(hgraph)
    probs = logits.softmax(dim=-1)
    y_pred = logits.argmax(dim=-1)
    class_label = hgraph.y

# %%

explanation_fn = "explanation_learn_normal.json"

with open(path / explanation_fn) as f:
    explanation = EasyDict(json.load(f))

assert explanation.config.load_fn == str(path)
assert explanation.config.load_best == load_best


print(path, explanation_fn)

inds = get_explanation_inds(explanation)
print(f"Number of instances: {len(inds)}")
probs_target = probs[inds]
probs_expl = get_explanation_statistic(explanation, key='classprob/post', inds=inds, is_complement=False)
probs_compl = get_explanation_statistic(explanation, key='classprob/post', inds=inds, is_complement=True)
loss_expl = get_explanation_statistic(explanation, key='loss/post', inds=inds, is_complement=False)
loss_compl = get_explanation_statistic(explanation, key='loss/post', inds=inds, is_complement=True)
loss_pred_expl = get_explanation_statistic(explanation, key='loss_pred/post', inds=inds, is_complement=False)
loss_pred_compl = get_explanation_statistic(explanation, key='loss_pred/post', inds=inds, is_complement=True)


print()

print("fidelity | - | +")
print(f"acc | expl {fidelity(p=probs_expl, p_ref=probs_target, dissimilarity_func=dissimilar_by_accuracy):.4f} | compl {fidelity(p=probs_compl, p_ref=probs_target, dissimilarity_func=dissimilar_by_accuracy):.4f}")
print(f"kl | expl {fidelity(p=probs_expl, p_ref=probs_target, dissimilarity_func=dissimilar_by_kl):.4f} | compl {fidelity(p=probs_compl, p_ref=probs_target, dissimilarity_func=dissimilar_by_kl):.4f}")
print(f"tvd | expl {fidelity(p=probs_expl, p_ref=probs_target, dissimilarity_func=dissimilar_by_tv):.4f} | compl {fidelity(p=probs_compl, p_ref=probs_target, dissimilarity_func=dissimilar_by_tv):.4f}")
print(f"xent | expl {fidelity(p=probs_expl, p_ref=probs_target, dissimilarity_func=dissimilar_by_xent):.4f} | compl {fidelity(p=probs_compl, p_ref=probs_target, dissimilarity_func=dissimilar_by_xent):.4f}")
print(f"loss | expl {loss_expl.mean().item():.4f} | compl {loss_compl.mean().item():.4f}")
print(f"losspred | expl {loss_pred_expl.mean().item():.4f} | compl {loss_pred_compl.mean().item():.4f}")

size_pre = get_explanation_statistic(explanation, key='size/pre', inds=inds, is_complement=False)
size_expl = get_explanation_statistic(explanation, key='size/post', inds=inds, is_complement=False)
size_compl = get_explanation_statistic(explanation, key='size/post', inds=inds, is_complement=True)

print()

print(f"size pre {size_pre.mean()}")
print(f"size | expl {size_expl.mean():.1f} | compl {size_compl.mean():.1f}")
print(f"size frac | expl {(size_expl / size_pre).mean():.3f} | compl {(size_compl / size_pre).mean():.3f}")
print(f"size median| expl {size_expl.median():.1f} | compl {size_compl.median():.1f}")
print(f"size frac median | expl {(size_expl / size_pre).median():.3f} | compl {(size_compl / size_pre).median():.3f}")


# %%


def dataset_name_to_latex(dataset_name):

    if dataset_name == "randhouse":
        return r"\textsc{H-RandHouse}"
    
    if dataset_name == "commhouse":
        return r"\textsc{H-CommHouse}"
    
    if dataset_name == "treecycle":
        return r"\textsc{H-TreeCycle}"
    
    if dataset_name == "treegrid":
        return r"\textsc{H-TreeGrid}"
    
    if dataset_name == "cora":
        return r"\textsc{Cora}"
    
    if dataset_name == "coauthor_cora":
        return r"\textsc{CoauthorCora}"
    
    if dataset_name == "coauthor_dblp":
        return r"\textsc{CoauthorDBLP}"
    
    if dataset_name == "zoo":
        return r"\textsc{Zoo}"
    
    raise NotImplementedError



def make_latex_table(datasets, methods, show_loss=False, is_complement=False):
    """
    dataset & 
    method &
    &
    loss &
    & 
    fid-acc & 
    fid-kl & 
    fid-tv &
    & 
    size & 
    density
    """

    LATEX_STR = r""

    for dataset in datasets:

        path = Path(f"train_results/FINAL/{dataset}")
        load_best = True

        cfg_model, _, hgraph, model = get_single_run(path, device=torch.device("cpu"), load_best=load_best)

        with torch.no_grad():
            probs = model(hgraph).softmax(dim=-1)

        LATEX_STR += dataset_name_to_latex(dataset)
        LATEX_STR += "\n"

        for method in methods:

            explanation_fn = f"explanation_{method}.json"

            with open(path / explanation_fn) as f:
                explanation = EasyDict(json.load(f))

            assert explanation.config.load_fn == str(path)
            assert explanation.config.load_best == load_best


            LATEX_STR += " & "
            LATEX_STR += method
            LATEX_STR += " && "


            inds = get_explanation_inds(explanation)
            # if dataset == "randhouse":
            #     inds = list(range(311,inds[-1]))
            print(f"{dataset} | {method} | averaging {len(inds)} instances")

            probs_target = probs[inds]
            probs_sub = get_explanation_statistic(explanation, 'classprob/post', inds=inds, is_complement=is_complement)

            size_pre = get_explanation_statistic(explanation, 'size/pre', inds=inds, is_complement=False)
            size_sub = get_explanation_statistic(explanation, 'size/post', inds=inds, is_complement=is_complement)

            if show_loss:
                loss_sub = get_explanation_statistic(explanation, key='loss/post', inds=inds, is_complement=is_complement)

            fid_acc = fidelity(p=probs_sub, p_ref=probs_target, dissimilarity_func=dissimilar_by_accuracy)
            fid_kl = fidelity(p=probs_sub, p_ref=probs_target, dissimilarity_func=dissimilar_by_kl)
            fid_tv = fidelity(p=probs_sub, p_ref=probs_target, dissimilarity_func=dissimilar_by_tv)

            if show_loss:
                LATEX_STR += f"{loss_sub.mean():.2f}"
                LATEX_STR += " && "

            LATEX_STR += f"{fid_acc:.2f}"
            LATEX_STR += " & "
            LATEX_STR += f"{fid_kl:.2f}"
            LATEX_STR += " & "
            LATEX_STR += f"{fid_tv:.2f}"

            LATEX_STR += " && "

            LATEX_STR += f"{size_sub.mean():.1f}"
            LATEX_STR += " & "
            LATEX_STR += f"{(size_sub / size_pre).mean():.2f}"

            LATEX_STR += r" \\ "
            LATEX_STR += "\n"
        
        LATEX_STR += r"\midrule"
        LATEX_STR += "\n"

    return LATEX_STR

# %%

latex_table = make_latex_table(
    datasets=[
        # "randhouse",
        # "commhouse",
        # "treecycle",
        "treegrid",
        # "cora",
        # "coauthor_cora",
        # "coauthor_dblp",
        # "zoo",
        ],
    methods=[
        # "random", 
        # "gradient",
        # "gradient_thresh20",
        "attention", 
        "attention_thresh20",
        # "learn",
        # "learn_size0-01",
        # "learn_size0-02",
        "learn_size0-05",
        # "learn_size0-1",
        # "learn_size0-2",
        # "learn_size0-5",
        # "learn_size0-05_normal",
        ],
    show_loss=False,
    is_complement=False,
)

print(latex_table)

# %%

# def make_latex_table(datasets, methods):
#     """
#     dataset & 
#     method &
#     & 
#     loss explanation & 
#     fid- kl & 
#     density explanation &
#     & 
#     loss complement & 
#     fid+ kl &
#     density complement
#     """

#     LATEX_STR = r""

#     for dataset in datasets:

#         path = Path(f"train_results/FINAL/{dataset}")
#         load_best = True

#         cfg_model, _, hgraph, model = get_single_run(path, device=torch.device("cpu"), load_best=load_best)

#         with torch.no_grad():
#             probs = model(hgraph).softmax(dim=-1)

#         LATEX_STR += dataset_name_to_latex(dataset)
#         LATEX_STR += "\n"

#         for method in methods:

#             explanation_fn = f"explanation_{method}.json"

#             with open(path / explanation_fn) as f:
#                 explanation = EasyDict(json.load(f))

#             assert explanation.config.load_fn == str(path)
#             assert explanation.config.load_best == load_best


#             LATEX_STR += " & "
#             LATEX_STR += method
#             LATEX_STR += " && "


#             inds = get_explanation_inds(explanation)
#             print(f"{dataset} | {method} | averaging over {len(inds)} instances")

#             probs_target = probs[inds]
#             probs_expl = get_explanation_statistic(explanation, 'classprob/post', inds=inds, is_complement=False)
#             probs_compl = get_explanation_statistic(explanation, 'classprob/post', inds=inds, is_complement=True)

#             size_pre = get_explanation_statistic(explanation, 'size/pre', inds=inds, is_complement=False)
#             size_expl = get_explanation_statistic(explanation, 'size/post', inds=inds, is_complement=False)
#             size_compl = get_explanation_statistic(explanation, 'size/post', inds=inds, is_complement=True)

#             loss_expl = get_explanation_statistic(explanation, key='loss/post', inds=inds, is_complement=False)
#             loss_compl = get_explanation_statistic(explanation, key='loss/post', inds=inds, is_complement=True)

#             fid_expl_kl = fidelity(p=probs_expl, p_ref=probs_target, dissimilarity_func=dissimilar_by_kl)
#             fid_compl_kl = fidelity(p=probs_compl, p_ref=probs_target, dissimilarity_func=dissimilar_by_kl)

#             LATEX_STR += f"{loss_expl.mean():.2f}"
#             LATEX_STR += " & "

#             LATEX_STR += f"{fid_expl_kl:.2f}"
#             LATEX_STR += " & "

#             LATEX_STR += f"{(size_expl / size_pre).mean():.2f}"

#             LATEX_STR += " && "

#             LATEX_STR += f"{loss_compl.mean():.2f}"
#             LATEX_STR += " & "

#             LATEX_STR += f"{fid_compl_kl:.2f}"
#             LATEX_STR += " & "

#             LATEX_STR += f"{(size_compl / size_pre).mean():.2f}"

#             LATEX_STR += r" \\ "
#             LATEX_STR += "\n"
        

#         LATEX_STR += r"\midrule"
#         LATEX_STR += "\n"

#     return LATEX_STR

# %%

# sampler ablation

latex_table = make_latex_table(
    datasets=["randhouse", "zoo"],
    methods=["learn", "learn_sigmoid", "learn_sparsemax"],
    show_loss=True,
)

print(latex_table)


# %%

def get_subgraph_from_file(explanation_fn, node_idx, ax, path, load_best, with_title=True):

    with open(path / explanation_fn) as f:
        explanation = EasyDict(json.load(f))

    assert explanation.config.load_fn == str(path)
    assert explanation.config.load_best == load_best

    print(path, explanation_fn, explanation.summary[str(node_idx)].explanation['gt_class'], explanation.summary[str(node_idx)].explanation['pred_class'])
    # assert len(hgraph_local.norm) == get_explanation_statistic(explanation, key='size/pre', inds=[node_idx], is_complement=False).item()

    incdict = get_explanation_statistic(explanation, key='incidence_dict/post', inds=[node_idx], is_complement=False)[0]

    if len(incdict) > 0:
        hgraph_expl = hnx.Hypergraph(incdict)
        hnx.draw(hgraph_expl, ax=ax, node_radius={node_idx: 3})
    else:
        ax.text(x=0.5, y=0.5, s="Null hypergraph.", horizontalalignment="center")
        ax.axis('off')
    
    kl_loss = get_explanation_statistic(explanation, key='loss_pred/post', inds=[node_idx], is_complement=False).item()
    
    if with_title:
        ax.set_title(explanation_fn.replace("explanation_","").replace(".json","").replace("_size0-05","") + f" (KLDiv = {kl_loss:.2f})")


# %%

dataset = "treegrid"
path = Path(f"train_results/FINAL/{dataset}")
load_best = True
for node_idx in [473, 136, 361, 553, 15, 366, 482, 102, 484, 299]:
    print(node_idx)
    fig, ax = plt.subplots(1,1,figsize=(3,3))
    get_subgraph_from_file("explanation_learn_size0-05.json", node_idx, ax, path, load_best, with_title=False)
    plt.show()
    fig.savefig(f"example_ours_{dataset}_{node_idx}.svg", format="svg")


# %%

# get the subhyergraph to visualise

from explain import get_local_hypergraph, transfer_features

dataset = "randhouse"
num_expansions = 3
# dataset = "coauthor_cora"
# num_expansions = 1


path = Path(f"train_results/FINAL/{dataset}")
load_best = True

cfg_model, _, hgraph, model = get_single_run(path, device=torch.device("cpu"), load_best=load_best)

with torch.no_grad():
    logits = model(hgraph)
    probs = logits.softmax(dim=-1)
    y_pred = logits.argmax(dim=-1)
    class_label = hgraph.y


node_idx = 691

print(f"node {node_idx} | class {class_label[node_idx].item()}")

hgraph_local = get_local_hypergraph(idx=node_idx, hgraph=hgraph, num_expansions=num_expansions, is_hedge_concept=False)
transfer_features(hgraph, hgraph_local, cfg_model)


fig, ax = plt.subplots(5,1,figsize=(3,12))
hnx.draw(hgraph_local, ax=ax[0], node_radius={node_idx: 3})
ax[0].set_title("computational subhgraph")
get_subgraph_from_file("explanation_random.json", node_idx, ax[1], path, load_best)
get_subgraph_from_file("explanation_gradient.json", node_idx, ax[2], path, load_best)
get_subgraph_from_file("explanation_attention.json", node_idx, ax[3], path, load_best)
get_subgraph_from_file("explanation_learn_size0-05.json", node_idx, ax[4], path, load_best)
plt.show()
fig.savefig(f"example_{dataset}_{node_idx}.svg", format="svg")

# %%


# check that the KL thing is correct here

with open(path / "explanation_learn.json") as f:
    explanation = EasyDict(json.load(f))
incdict = get_explanation_statistic(explanation, key='incidence_dict/post', inds=[i], is_complement=False)[0]
incdict = get_explanation_statistic(explanation, key='incidence_dict/post', inds=[i], is_complement=False)[0]
incdict = {'e0' + k.replace('e',''):v for k,v in incdict.items()}
hgraph_expl = hnx.Hypergraph(incdict)
plt.figure(figsize=(3,3))
hnx.draw(hgraph_expl)
plt.show()

transfer_features(hgraph, hgraph_expl, cfg_model)

with torch.no_grad():
    p_expl = model(hgraph_expl).softmax(dim=-1)

dissimilar_by_kl(p_expl[hgraph_expl.node_to_ind[i]].unsqueeze(0), probs[i].unsqueeze(0))

# %%