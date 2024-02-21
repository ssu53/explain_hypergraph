# %%

import numpy as np
import matplotlib.pyplot as plt 
import torch
from pathlib import Path
import hypernetx as hnx
import networkx as nx 

from hgraph import incidence_matrix_to_edge_index
from train.vis_results import get_single_run
from explain import plot_concepts, ActivationClassifier, plot_samples, get_local_hypergraph

# %%


def enrich_features(hgraph, sub_hgraph):

    # ASSUME hgraph.incidence_matrix returns correctly sorted 
    node_idx = sorted(set([node for nodes in sub_hgraph.incidence_dict.values() for node in nodes]))
    ind_to_node = {k:v for k,v in zip(range(len(node_idx)), node_idx)}
    node_to_ind = {v:k for k,v in ind_to_node.items()}

    edge_idx = sorted(sub_hgraph.incidence_dict.keys())
    ind_to_edge = {k:v for k,v in zip(range(len(edge_idx)), edge_idx)}
    edge_to_ind = {v:k for k,v in ind_to_edge.items()}

    sub_hgraph.ind_to_node = ind_to_node
    sub_hgraph.node_to_ind = node_to_ind
    sub_hgraph.ind_to_edge = ind_to_edge
    sub_hgraph.edge_to_ind = edge_to_ind
    sub_hgraph.train_mask = hgraph.train_mask[node_idx]
    sub_hgraph.val_mask = hgraph.val_mask[node_idx]
    sub_hgraph.test_mask = hgraph.test_mask[node_idx]
    sub_hgraph.x = hgraph.x[node_idx]
    sub_hgraph.y = hgraph.y[node_idx]
    sub_hgraph.num_house_types = hgraph.num_house_types
    sub_hgraph.num_classes = hgraph.num_classes

    sub_hgraph.H = torch.tensor(sub_hgraph.incidence_matrix().toarray(), dtype=torch.float32)
    sub_hgraph.H_unmasked = torch.tensor(sub_hgraph.incidence_matrix().toarray(), dtype=torch.float32)
    sub_hgraph.edge_index = incidence_matrix_to_edge_index(sub_hgraph.H_unmasked)

    assert all(hgraph.y[list(sub_hgraph.ind_to_node.values())] == sub_hgraph.y)
    assert all(hgraph.train_mask[list(sub_hgraph.ind_to_node.values())] == sub_hgraph.train_mask)
    assert all(hgraph.val_mask[list(sub_hgraph.ind_to_node.values())] == sub_hgraph.val_mask)
    assert all(hgraph.test_mask[list(sub_hgraph.ind_to_node.values())] == sub_hgraph.test_mask)

    return sub_hgraph



def incidence_matrix_to_incidence_dict_named(H, node_names, edge_names):

    _, num_edges = H.shape

    incidence_dict = {}
    for edge_ind in range(num_edges):
        node_inds = torch.where(H[:,edge_ind] == 1)[0].tolist()
        print(edge_ind, node_inds)
        incidence_dict[edge_names[edge_ind]] = [node_names[ni] for ni in node_inds]
    return incidence_dict



def mask_density(masked_adj, adj):
    return torch.sum(masked_adj) / torch.sum(adj)


def mask_values_stats(masked_adj, adj):
    """
    masked_adj is the masked probabilities on the binary incidence matrix
    adj is the original binary incidence matrix
    """

    assert masked_adj.shape == adj.shape
    assert torch.all(masked_adj[adj == 0.0] == 0)
    assert torch.allclose(torch.where(adj == 0.0, 0.0, 1.0), adj)
    assert torch.max(masked_adj).item() <= 1.0
    assert torch.max(masked_adj).item() >= 0.0

    print(f"mask density: {mask_density(masked_adj, adj)}")
    values_being_learnt = masked_adj[adj != 0.0]
    values_being_learnt = values_being_learnt.flatten().detach().cpu().numpy()

    plt.figure(figsize=(3,3))
    plt.hist(values_being_learnt, bins=np.linspace(0.0,1.0,100))
    plt.title("Distribution of learnt mask entries")
    plt.show()


def hgnn_explain(
        node, 
        hgraph, 
        model, 
        init_strategy="normal", 
        num_epochs=200, 
        lr=0.01, 
        loss_pred_type="maximise_label",
        print_every=25,
        hgraph_full=None,
    ):
    """
    node: index of node to be explained
    hgraph: should be at least the node's computational subgraph, such that the model's prediction on it matches that on the full hgraph
    model: trained model
    
    TOOD: feature mask
    """


    model.eval()

    with torch.no_grad():
        logits_target = model(hgraph_full)[node]
        # assert logits_target == model(hgraph)[hgraph.node_to_ind[node]]
        pred_target = logits_target.softmax(dim=-1)
        label_target = pred_target.argmax().item()


    if init_strategy == "const":
        mask = torch.ones_like(hgraph.H_unmasked)
    elif init_strategy == "normal":
        mask = torch.empty_like(hgraph.H_unmasked)
        std = torch.nn.init.calculate_gain("relu") * (2.0 / (mask.shape[0] + mask.shape[1])) ** 0.5
        with torch.no_grad():
            mask.normal_(1.0, std)
    else:
        raise NotImplementedError


    mask.requires_grad = True

    for p in model.parameters():
        p.requires_grad = False

    optimiser = torch.optim.Adam([mask], lr=lr)

    coeffs = {
        'size': 0.05,
        'ent': 1.0,
    }

    print(f"Learning subgraph for node #{node} with model label {label_target} and g.t. label {hgraph.y[hgraph.node_to_ind[node]]}")

    for epoch in range(1,num_epochs+1):

        mask_prob = torch.sigmoid(mask)
        hgraph.H = mask_prob * hgraph.H_unmasked
        ind = hgraph.node_to_ind[node]
        logits_actual = model(hgraph)[ind]
        pred_actual = logits_actual.softmax(dim=0)

        if loss_pred_type == "mutual_info":
            """
            Mutual information i.e. conditional entropy loss
            Is optimised by placing all the predicted probability weight on any one class
            This appears in GNNExplainer paper
            """
            loss_pred = -torch.sum(pred_actual * torch.log(pred_actual))

        elif loss_pred_type == "maximise_label":
            """
            Maximise the logit corresponding to the model being explained
            This appears in GNNExplainer code
            """
            loss_pred = -torch.log(pred_actual[label_target])

        elif loss_pred_type == "kl_div":
            """
            KL divergence between actual and target prediction
            """
            loss_pred = torch.sum(pred_target * (torch.log(pred_target) - torch.log(pred_actual)))
        
        else:
            raise NotImplementedError

        loss_size = coeffs['size'] * torch.sum(mask_prob)
        mask_ent = -mask_prob * torch.log(mask_prob) - (1 - mask_prob) * torch.log(1 - mask_prob)
        loss_mask_ent = coeffs['ent'] * torch.mean(mask_ent)

        loss = loss_pred + loss_size + loss_mask_ent
        if epoch % print_every == 0:
            print(
                epoch, 
                f"{loss.item():.2f}", 
                f"{loss_pred.item():.2f}", 
                f"{loss_size.item():.2f}", 
                f"{loss_mask_ent.item():.2f}", 
                f"{mask_density(hgraph_local.H, hgraph_local.H_unmasked):.2f}",
                np.round(logits_actual.detach().cpu().numpy(),2),
                np.round(pred_actual.detach().cpu().numpy(),2),
            )
        loss.backward()
        optimiser.step()


    mask_values_stats(hgraph.H, hgraph.H_unmasked)


    # print(H_neighb.H_unmasked[H_neighb.node_to_ind[node],:])
    # print(H_neighb.H[H_neighb.node_to_ind[node],:])



def show_learnt_subgraph(hgraph_learn, thresh_num=None, thresh=None, node_to_include=None):

    if thresh_num is not None:
        thresh = sorted(hgraph_learn.H.detach().cpu().numpy().flatten())[-thresh_num]
        print(f"Masking threshold override to {thresh=}.")
    
    if thresh is None: 
        print("Provide thresh_num or thresh as arguments.")
    

    H_sparse = torch.where(hgraph_learn.H >= thresh, 1.0, 0.0)


    if node_to_include is not None:
        # Patch this node back into the learnt mask
        H_sparse[hgraph_learn.node_to_ind[node_to_include],:] = hgraph_learn.H_unmasked[hgraph_learn.node_to_ind[node_to_include],:]
        print(H_sparse)


    hgraph_sparse_incdict = incidence_matrix_to_incidence_dict_named(H_sparse.cpu(), hgraph_learn.ind_to_node, hgraph_learn.ind_to_edge)

    hnx.draw(hnx.Hypergraph(hgraph_sparse_incdict), layout=nx.spring_layout)


# %%
    

# Load training results, compute cobncept completeness, show samples

path = Path('train_results/s24_standard_v3/ones1/gcn_resid_len/run0')
cfg, train_stats, hgraph, model = get_single_run(path, device=torch.device("cpu"))

concepts, kmeans_model = plot_concepts(hgraph, model, num_clusters=7, binarise_concepts=False)


ac = ActivationClassifier(
    concepts, kmeans_model, "decision_tree",
    hgraph.x.cpu().reshape(-1,1), hgraph.y.cpu(), 
    hgraph.train_mask.cpu(), hgraph.val_mask.cpu())
print(f"Concept completeness: {ac.get_classifier_accuracy()}")


sample_graphs, sample_feats = plot_samples(concepts, kmeans_model, hgraph.y, hgraph, num_expansions=2, num_nodes_view=3)


# %%


# Select node to explain

node_idx = 546

hgraph_local = get_local_hypergraph(node_idx=node_idx, hgraph=hgraph, num_expansions=3)
enrich_features(hgraph, hgraph_local)

# hnx.draw(hgraph_neighb.collapse_nodes(), with_node_counts=True, with_node_labels=True)
hnx.draw(hgraph_local, with_node_labels=True)


# %%

hgnn_explain(
    node_idx, 
    hgraph_local, 
    model, 
    init_strategy="normal", 
    num_epochs=200, 
    lr=0.01, 
    loss_pred_type="kl_div",
    print_every=25,
    hgraph_full=hgraph,
    )


# %%

show_learnt_subgraph(hgraph_local, thresh_num=12, node_to_include=None)
# %%
