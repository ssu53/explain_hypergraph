import pickle
import os
from datetime import datetime
from tqdm import tqdm

import torch
from models.graph_models import GCN



def get_train_val_test_mask(n, split, seed):

    split_rand_generator = torch.Generator().manual_seed(seed)
    node_index = range(n)
    train_inds, val_inds, test_inds = torch.utils.data.random_split(node_index, split, generator=split_rand_generator)

    train_mask = torch.zeros(n, dtype=bool)
    train_mask[train_inds] = True

    val_mask = torch.zeros(n, dtype=bool)
    val_mask[val_inds] = True

    test_mask = torch.zeros(n, dtype=bool)
    test_mask[test_inds] = True

    return train_mask, val_mask, test_mask



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
    loss.backward()
    optimiser.step()
    del logits
    torch.cuda.empty_cache()
    return loss.item()


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
    with torch.no_grad():
        if isinstance(model, GCN):
            logits = model(graph.x, graph.edge_index)[mask]
        else:
            logits = model(graph)[mask]
        acc = get_accuracy(logits, y)
    del logits
    torch.cuda.empty_cache()
    return acc



def train_eval_loop(model, hgraph, train_mask, val_mask, test_mask, lr, num_epochs, printevery=10, verbose=True):
    
    optimiser = torch.optim.Adam(model.parameters(), lr=lr)
    train_stats = None

    train_acc = eval(hgraph, model, train_mask)
    val_acc = eval(hgraph, model, val_mask)
    test_acc = eval(hgraph, model, test_mask)
    epoch_stats = {'train_acc': train_acc, 'val_acc': val_acc, 'test_acc': test_acc, 'epoch': 0, 'train_loss': torch.nan}
    train_stats = update_stats(train_stats, epoch_stats)

    for epoch in range(1,num_epochs+1): # 1-index the epochs
        train_loss = train(hgraph, model, train_mask, optimiser)
        train_acc = eval(hgraph, model, train_mask)
        val_acc = eval(hgraph, model, val_mask)
        test_acc = eval(hgraph, model, test_mask)
        
        if verbose and epoch % printevery == 0:
            print(f"Epoch {epoch} with train loss: {train_loss:.3f} train acc: {train_acc:.3f} val acc: {val_acc:.3f}")
        
        epoch_stats = {'train_acc': train_acc, 'val_acc': val_acc, 'test_acc': test_acc, 'epoch': epoch, 'train_loss': train_loss}
        train_stats = update_stats(train_stats, epoch_stats)

    if verbose:
        print(f"Final train acc: {train_acc:.3f} | val acc: {val_acc:.3f} | test acc: {test_acc:.3f} ")

    return train_stats



def train_eval_loop_many(
        nruns: int, 
        model_class,
        model_args,
        hgraph, 
        train_mask, val_mask, test_mask, 
        lr, num_epochs, 
        printevery=10, verbose=False,
        save_dir=None, save_name=None,
        device=None,
        ):        

    train_stats_all = {}

    for nrun in tqdm(range(nruns)):

        model = model_class(**model_args)
        model.to(device)
        
        train_stats_all[nrun] = train_eval_loop(model, hgraph, train_mask, val_mask, test_mask, lr, num_epochs, printevery, verbose)
    
    if save_dir is not None:
        if save_name is None:
            save_name = f"train_stats_{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}"
        with open(os.path.join(save_dir, save_name), 'wb') as f:
            pickle.dump(train_stats_all, f)

    return train_stats_all



def get_activations(feat_name, activations):
    def hook(model, input, output):
        activations[feat_name] = output.detach()
    return hook


