import torch
from models.graph_models import GCN


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
    model.eval()
    y = graph.y[mask]
    with torch.no_grad():
        if isinstance(model, GCN):
            logits = model(graph.x, graph.edge_index)[mask]
        else:
            logits = model(graph)[mask]
        acc = get_accuracy(logits, y)
    return acc



def train_eval_loop(model, hgraph, train_mask, val_mask, test_mask, lr, num_epochs, printevery=10):
    
    optimiser = torch.optim.Adam(model.parameters(), lr=lr)
    train_stats = None

    for epoch in range(num_epochs):
        
        train_loss = train(hgraph, model, train_mask, optimiser)
        train_acc = eval(hgraph, model, train_mask)
        val_acc = eval(hgraph, model, val_mask)
        
        if epoch % printevery == 0:
            print(f"Epoch {epoch} with train loss: {train_loss:.3f} train acc: {train_acc:.3f} val acc: {val_acc:.3f}")
        
        epoch_stats = {'train_acc': train_acc, 'val_acc': val_acc, 'epoch': epoch}
        train_stats = update_stats(train_stats, epoch_stats)

    test_acc = eval(hgraph, model, test_mask)
    print(f"Final test acc: {test_acc:.3f}")

    return train_stats



def get_activations(feat_name, activations):
    def hook(model, input, output):
        activations[feat_name] = output.detach()
    return hook


