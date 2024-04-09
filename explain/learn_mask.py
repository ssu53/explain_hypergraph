import numpy as np
import matplotlib.pyplot as plt 
import torch
import networkx as nx 
import hypernetx as hnx
from collections import Counter
import wandb

from hgraph import incidence_matrix_to_edge_index
from models.allset import norm_contruction
from explain import get_edges_of_nodes



def transfer_features(hgraph, sub_hgraph, cfg):

    assert all(node in hgraph.nodes() for node in sub_hgraph.nodes())
    assert all(hedge in hgraph.edges() for hedge in sub_hgraph.edges())
    assert all(node in hgraph.incidence_dict[hedge] for hedge in sub_hgraph.edges() for node in sub_hgraph.incidence_dict[hedge])

    # ASSUME hgraph.incidence_matrix returns correctly sorted, and numbered from 0,...
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

    # this is named with ind, not the node
    sub_hgraph.edge_index = incidence_matrix_to_edge_index(sub_hgraph.H_unmasked)

    if 'normtype' in cfg:
        # populate sub_hgraph.norm for AllDeepSets (.norm is not used for AllSetTransformers)
        norm_contruction(sub_hgraph, option=cfg.normtype)

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
        # print(edge_ind, node_inds)
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



def mask_values_stats_sparse(mask):

    assert mask.ndim == 1
    values_being_learnt = mask.detach().cpu().numpy()

    plt.figure(figsize=(3,3))
    plt.hist(values_being_learnt, bins=np.linspace(0.0,1.0,100))
    plt.title("Distribution of learnt mask entries")
    plt.show()



def explainer_loss(mask_prob, pred_actual, pred_target, label_target, loss_pred_type, coeffs):

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

    return loss, loss_pred, loss_size, loss_mask_ent



def hgnn_explain_sparse(
        node, 
        hgraph, 
        model, 
        init_strategy="normal", 
        num_epochs=200, 
        lr=0.01, 
        loss_pred_type="maximise_label",
        print_every=25,
        hgraph_full=None,
        coeffs=None,
        scheduler_fn=None,
        verbose=True,
        wandb_config=None,
    ):
    """
    THIS ONE UPDATES .norm for the edge weights in the sparse representation
    node: index of node to be explained
    hgraph: should be at least the node's computational subgraph, such that the model's prediction on it matches that on the full hgraph
    model: trained model
    
    TOOD: feature mask
    """

    if verbose: 
        print(coeffs)
    # coeff_ent_max = coeffs['ent']


    model.eval()

    with torch.no_grad():
        logits_target = model(hgraph_full)[node]
        # assert logits_target == model(hgraph)[hgraph.node_to_ind[node]]
        pred_target = logits_target.softmax(dim=-1)
        label_target = pred_target.argmax().item()
        
        if verbose:
            print("Target logit distribution", np.round(pred_target.detach().cpu().numpy(),2))


    if init_strategy == "const":
        mask = torch.ones_like(hgraph.norm).to(torch.float32)
        mask = mask * 3
    elif init_strategy == "normal":
        mask = torch.empty_like(hgraph.norm).to(torch.float32)
        std = torch.nn.init.calculate_gain("relu") * (2.0 / mask.shape[0]) ** 0.5 # yolo'd the gain value here
        with torch.no_grad():
            mask.normal_(1.0, std)
    else:
        raise NotImplementedError


    mask.requires_grad = True

    for p in model.parameters():
        p.requires_grad = False

    optimiser = torch.optim.Adam([mask], lr=lr)
    if scheduler_fn is not None:
        scheduler = scheduler_fn(optimiser)

    if verbose:
        print(f"Learning subgraph for node #{node} with model label {label_target} and g.t. label {hgraph.y[hgraph.node_to_ind[node]]}")

    mask_prob = torch.sigmoid(mask)
    # hgraph.H = mask_prob * hgraph.H_unmasked
    hgraph.norm = mask_prob

    lrs = []

    for epoch in range(1,num_epochs+1):

        ind = hgraph.node_to_ind[node]
        logits_actual = model(hgraph)[ind]
        pred_actual = logits_actual.softmax(dim=0)

        # coeffs = {'size': coeffs['size'], 'ent': min(coeff_ent_max * epoch / num_epochs * 2, coeff_ent_max)}

        loss, loss_pred, loss_size, loss_mask_ent = explainer_loss(
            mask_prob,
            pred_actual,
            pred_target if loss_pred_type == "kl_div" else None,
            label_target if loss_pred_type == "maximise_label" else None,
            loss_pred_type,
            coeffs,
        )

        # update mask values into hgraph
        mask_prob = torch.sigmoid(mask)
        # hgraph.H = mask_prob * hgraph.H_unmasked
        hgraph.norm = mask_prob

        if wandb_config is not None:
            wandb.log({
                'train/loss': loss,
                'train/loss_pred': loss_pred,
                'train/loss_size': loss_size,
                'train/loss_mask_ent': loss_mask_ent,
                'train/mask_density': mask_density(hgraph.norm, torch.ones_like(hgraph.norm)),
            })

        if verbose and (epoch % print_every == 0):
            print(
                epoch, 
                f"{loss.item():.2f}", 
                f"{loss_pred.item():.2f}", 
                f"{loss_size.item():.2f}", 
                f"{loss_mask_ent.item():.2f}", 
                # f"{mask_density(hgraph_local.H, hgraph_local.H_unmasked):.2f}",
                f"{mask_density(hgraph.norm, torch.ones_like(hgraph.norm)):.2f}",
                np.round(logits_actual.detach().cpu().numpy(),2),
                np.round(pred_actual.detach().cpu().numpy(),2),
            )

        if scheduler_fn is not None: lrs.append(scheduler.get_last_lr().pop())
        loss.backward()
        optimiser.step()
        if scheduler_fn is not None: scheduler.step()
    
    if wandb_config is not None:
        wandb.log({
            'train/loss': loss,
            'train/loss_pred': loss_pred,
            'train/loss_size': loss_size,
            'train/loss_mask_ent': loss_mask_ent,
            'train/mask_density': mask_density(hgraph.norm, torch.ones_like(hgraph.norm)),
        })

    if verbose:
        # mask_values_stats(hgraph.H, hgraph.H_unmasked)
        mask_values_stats_sparse(mask_prob)

    if verbose and (scheduler_fn is not None):
        plt.figure(figsize=(3,3))
        plt.title("learning rate schedule")
        plt.plot(range(1, num_epochs+1), lrs)
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
        
        print("Target logit distribution", np.round(pred_target.detach().cpu().numpy(),2))


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
        'size': 0.01,
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
                f"{mask_density(hgraph.H, hgraph.H_unmasked):.2f}",
                np.round(logits_actual.detach().cpu().numpy(),2),
                np.round(pred_actual.detach().cpu().numpy(),2),
            )
        loss.backward()
        optimiser.step()


    mask_values_stats(hgraph.H, hgraph.H_unmasked)


    # print(H_neighb.H_unmasked[H_neighb.node_to_ind[node],:])
    # print(H_neighb.H[H_neighb.node_to_ind[node],:])



def show_learnt_subgraph(hgraph_learn, thresh_num=None, thresh=None, node_to_include=None, cfg=None):

    if thresh_num is not None:
        thresh = sorted(hgraph_learn.H.detach().cpu().numpy().flatten())[-thresh_num]
        print(f"Masking threshold override to {thresh=}.")
    
    if thresh is None: 
        print("Provide thresh_num or thresh as arguments.")
    

    H_sparse = torch.where(hgraph_learn.H >= thresh, 1.0, 0.0)


    if node_to_include is not None:
        # Patch this node back into the learnt mask
        H_sparse[hgraph_learn.node_to_ind[node_to_include],:] = hgraph_learn.H_unmasked[hgraph_learn.node_to_ind[node_to_include],:]
        # print(H_sparse)


    hgraph_sparse_incdict = incidence_matrix_to_incidence_dict_named(H_sparse.cpu(), hgraph_learn.ind_to_node, hgraph_learn.ind_to_edge)

    hgraph_sparse = hnx.Hypergraph(hgraph_sparse_incdict)

    hnx.draw(hgraph_sparse, layout=nx.spring_layout)

    return hgraph_sparse



def get_hyperedge_labels(hgraph):
    node_labels = hgraph.y
    hedge_labels = []
    for hedge in hgraph.incidence_dict:
        cnt = Counter(node_labels[node].item() for node in hgraph.incidence_dict[hedge]) # labels contained
        if cnt.total() == 3 and cnt.get(1, 0) == 1 and cnt.get(2, 0) == 2:
            hedge_labels.append(2) # roof hedge of house
        elif cnt.total() == 4 and cnt.get(2, 0) == 2 and cnt.get(3, 0 ) == 2:
            hedge_labels.append(3) # body hedge of house
        elif cnt.total() == 2 and cnt.get(0, 0) == 1 and cnt.get(2, 0) == 1:
            hedge_labels.append(1) # link connecting motif to base graph
        else:
            hedge_labels.append(0) # base graph hedge
    return torch.tensor(hedge_labels).long()



def subgraph_selector(hgraph, incidence_dict, cfg):
    hgraph_selected = hnx.Hypergraph(incidence_dict)
    transfer_features(hgraph, hgraph_selected, cfg)
    return hgraph_selected



def get_human_motif(node_idx, hgraph, cfg):
    """
    hacky
    node indexes must be ordered such that all the 0 nodes come first, then house motifs of form 1 2 2 3 3
    doesn't deal with added edge perturbations
    """

    # ground truth label
    label = hgraph.y[node_idx].item()

    if label == 0: 
        # choose the motif to be the single node and all of its containing hyperedges
        # this seems to tend to be classified as class 0
        edges = get_edges_of_nodes(hgraph, [node_idx])
        hgraph_selected = subgraph_selector(
            hgraph, 
            {edge: [node_idx] for edge in edges},
            cfg,
        )
        return hgraph_selected
    if label == 1:
        motif_nodes = [node_idx, node_idx+1, node_idx+2, node_idx+3, node_idx+4]
    if label == 2:
        if hgraph.y[node_idx-1].item() == 1:
            motif_nodes = [node_idx-1, node_idx, node_idx+1, node_idx+2, node_idx+3]
        else:
            motif_nodes = [node_idx-2, node_idx-1, node_idx, node_idx+1, node_idx+2]
    if label == 3:
        if hgraph.y[node_idx-1].item() == 2:
            motif_nodes = [node_idx-3, node_idx-2, node_idx-1, node_idx, node_idx+1]
        else:
            motif_nodes = [node_idx-4, node_idx-3, node_idx-2, node_idx-1, node_idx]
    
    # check that motif_nodes contains nodes of labels 1 2 2 3 3
    assert len(motif_nodes) == 5
    assert sum([hgraph.y[node_idx].item() == 1 for node_idx in motif_nodes]) == 1
    assert sum([hgraph.y[node_idx].item() == 2 for node_idx in motif_nodes]) == 2
    assert sum([hgraph.y[node_idx].item() == 3 for node_idx in motif_nodes]) == 2

    # get minimal hyperedges that contain it
    motif_edges = get_edges_of_nodes(hgraph, motif_nodes)

    # get minimal incidence dictionary containning those nodes and hyperedges
    select_incidence_dict = {edge: hgraph.incidence_dict[edge] for edge in motif_edges}

    # include one hyperedge containing only the anchor node
    # observed that this is necessary to achieve good logit accuracy for label 1 nodes
    anchor_node = [node_idx for node_idx in set(sum(select_incidence_dict.values(), [])) if hgraph.y[node_idx].item() == 0].pop()
    anchor_edge = [edge for edge in hgraph.incidence_dict if anchor_node in hgraph.incidence_dict[edge]]
    anchor_edge = [edge for edge in anchor_edge if edge not in select_incidence_dict]
    anchor_edge = anchor_edge.pop()
    select_incidence_dict[anchor_edge] = [anchor_node]

    # get the motif as subgraph hnx object
    hgraph_selected = subgraph_selector(
        hgraph, 
        select_incidence_dict,
        cfg,
    )
    # hnx.draw(hgraph_selected, with_node_labels=True)

    return hgraph_selected