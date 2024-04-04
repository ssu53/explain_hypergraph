import numpy as np
import random
from tqdm import tqdm
import torch
from models.graph_models import GCN
import copy


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False



def update_stats(training_stats, epoch_stats):
    """ Store metrics along the training
    Args:
      epoch_stats: dict containg metrics about one epoch
      training_stats: dict containing lists of metrics along training
    Returns:
      updated training_stats
    """
    if training_stats is None:
        training_stats = {}
        for key in epoch_stats.keys():
            training_stats[key] = []
    for key,val in epoch_stats.items():
        training_stats[key].append(val)
    return training_stats



def get_accuracy(logits, y):
    """
    Args:
        logts: logits predicted by model [num_nodes, num_classes]
        y: labels [num_nodes,]
    Returns:
        acc: prediction accuracy
    """

    num_nodes = len(y)
    y_hat = torch.argmax(logits, dim=-1)
    acc = (y == y_hat).sum().item() / num_nodes
    return acc



def train(graph, model, mask, optimiser):
    """
    Args:
        graph: graph/hypergraph object
        model: model
        mask: mask for subset of nodes to predict and evaluate
        optimiser: optimiser
    Return:
        loss: float of loss
    """
    model.train()
    y = graph.y[mask]
    optimiser.zero_grad()
    if isinstance(model, GCN):
        logits = model(graph.x, graph.edge_index)[mask]
    else:
        logits = model(graph)[mask]
    loss = torch.nn.functional.cross_entropy(logits, y)
    # loss.backward()
    # optimiser.step()
    del logits
    torch.cuda.empty_cache()
    return loss



def get_contrastive_samples(inds, class_label, batch_size, p_sim):
    """
    Samples pairs of indices for contrastive learning, with probability p_sim from same class.

    Args:
        inds: indices to sample from
    Returns
        ind1: indices of first sample in pair
        ind2: indices of second sample in pair
        label: +1 if pair is from same class, -1 otherwise
    """

    assert 0.0 <= p_sim <= 1.0

    sim_label = torch.where(torch.rand(batch_size) < p_sim, 1, -1).to(inds.device)
    ind1 = torch.zeros_like(sim_label)
    ind2 = torch.zeros_like(sim_label)

    for i in range(batch_size):
        ind = torch.randint(len(inds), (1,))
        i1 = inds[ind]
        # i1 = np.random.choice(inds)
        c1 = class_label[ind]

        if sim_label[i] == +1:
            ind2_choices = inds[(class_label == c1) & (inds != i1)]
        if sim_label[i] == -1:
            ind2_choices = inds[class_label != c1]
        
        # sample single element from ind2_choices
        i2 = ind2_choices[torch.randint(len(ind2_choices), (1,))]
        
        ind1[i] = i1
        ind2[i] = i2

    return ind1, ind2, sim_label



def train_contrastive(hgraph, final_embedding, train_mask, optimiser, contr_loss_fn, contr_lambda, contr_batch_size, contr_psim):

    optimiser.zero_grad()

    num_nodes = hgraph.number_of_nodes()
    assert final_embedding.shape[0] == num_nodes
    assert train_mask.shape[0] == num_nodes

    inds = torch.arange(num_nodes).to(train_mask.device)[train_mask]
    class_labels = hgraph.y[train_mask]
    ind1, ind2, labels = get_contrastive_samples(inds, class_labels, contr_batch_size, contr_psim)    
    loss = contr_loss_fn(final_embedding[ind1], final_embedding[ind2], labels) * contr_lambda

    # loss.backward()
    # optimiser.step()

    return loss



@torch.no_grad()
def eval(graph, model, mask):
    """
    Args:
        hgraph: hypergraph object
        model: model
        mask: mask for subset of nodes to evaluate
    Return:
        acc
    """
    if sum(mask).item() == 0: return torch.nan
    model.eval()
    y = graph.y[mask]
    if isinstance(model, GCN):
        logits = model(graph.x, graph.edge_index)[mask]
    else:
        logits = model(graph)[mask]
    acc = get_accuracy(logits, y)
    del logits
    torch.cuda.empty_cache()
    return acc



def train_eval_loop(model, hgraph, train_mask, val_mask, test_mask, lr, num_epochs, contr_lambda=0.1, contr_margin=0.25, contr_batch_size=256, contr_psim=0.0, printevery=10, save_best=False, verbose=True, **kwargs):
    
    optimiser = torch.optim.Adam(model.parameters(), lr=lr)
    train_stats = None

    if contr_lambda > 0:
        contr_loss_fn = torch.nn.CosineEmbeddingLoss(margin=contr_margin, reduction="mean")

    train_acc = eval(hgraph, model, train_mask)
    val_acc = eval(hgraph, model, val_mask)
    test_acc = eval(hgraph, model, test_mask)
    epoch_stats = {
        'train_acc': train_acc,
        'val_acc': val_acc,
        'test_acc': test_acc,
        'epoch': 0,
        'train_loss': torch.nan,
        'train_loss_xent': torch.nan,
        'train_loss_contr': torch.nan,
    }
    train_stats = update_stats(train_stats, epoch_stats)

    best_train_acc = train_acc
    best_model = None

    for epoch in range(1,num_epochs+1): # 1-index the epochs

        if contr_lambda > 0:
            activations = {}
            h = model.emb_layer.register_forward_hook(get_posthook("emb", activations))
        
        train_loss_xent = train(hgraph, model, train_mask, optimiser)
        
        if contr_lambda > 0:
            h.remove()
            train_loss_contr = train_contrastive(hgraph, activations['emb'], train_mask, optimiser, contr_loss_fn, contr_lambda, contr_batch_size, contr_psim)
            train_loss = train_loss_xent + train_loss_contr
        else:
            train_loss_contr = None
            train_loss = train_loss_xent

        train_loss.backward()
        optimiser.step()
        
        train_loss = train_loss.item()
        train_loss_xent = train_loss_xent.item()
        train_loss_contr = train_loss_contr.item() if train_loss_contr is not None else torch.nan

        train_acc = eval(hgraph, model, train_mask)
        val_acc = eval(hgraph, model, val_mask)
        test_acc = eval(hgraph, model, test_mask)
        
        if verbose and epoch % printevery == 0:
            print(f"Epoch {epoch} with train loss (xent, contr): {train_loss:.3f} ({train_loss_xent:.3f}, {train_loss_contr:.3f}) train acc: {train_acc:.3f} val acc: {val_acc:.3f}")
        
        epoch_stats = {
            'train_acc': train_acc,
            'val_acc': val_acc,
            'test_acc': test_acc,
            'train_loss': train_loss,
            'train_loss_xent': train_loss_xent,
            'train_loss_contr': train_loss_contr,
            'epoch': epoch,
        }
        train_stats = update_stats(train_stats, epoch_stats)

        if save_best and (train_acc > best_train_acc):
            best_train_acc = train_acc
            best_model = copy.deepcopy(model)

    if verbose:
        print(f"Final train acc: {train_acc:.3f} | val acc: {val_acc:.3f} | test acc: {test_acc:.3f} ")

    return train_stats, best_model



def train_eval_loop_many(
        nruns: int, 
        model_class,
        model_args,
        hgraph, 
        train_mask, val_mask, test_mask, 
        lr, num_epochs, 
        printevery=10, verbose=False,
        device=None,
        ):        

    train_stats_all = {}

    for nrun in tqdm(range(nruns)):

        model = model_class(**model_args)
        model.to(device)
        
        train_stats_all[nrun] = train_eval_loop(model, hgraph, train_mask, val_mask, test_mask, lr, num_epochs, printevery, verbose)

    return train_stats_all



def get_activations(key, dict):
    def hook(model, input, output):
        dict[key] = output.detach()
    return hook



def get_posthook(key, dict, detach=False):
    def hook(model, input, output):
        if detach:
            dict[key] = output.detach()
        else:
            dict[key] = output
    return hook
 


def get_prehook(key, dict, detach=False):
    def hook(model, input, output):
        input, _ = input # ignore the hypegraph edge index / incidence matrix
        if detach:
            dict[key] = input.detach()
        else:
            dict[key] = input
    return hook