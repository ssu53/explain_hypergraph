import numpy as np
import random
import torch

import hypernetx as hnx
from networkx.algorithms.bipartite import gnmk_random_graph


def generate_random_uniform_hypergraph(num_nodes, num_edges, k):
    """
    Generates k-uniform random hypergraphs, where each hyperedge contains exactly k nodes
    The hypergraph has maxinum num_nodes nodes (may have fewer as isolated nodes will be omitted)
    The hypergraph has maximum num_edges hyperedges (may have fewer as repeat edges will be omitted)
    """
    
    nodes = range(num_nodes)
    edges = []

    for _ in range(num_edges):
        edge = list(set(random.sample(nodes, k=k)))
        if edge not in edges:
          edges.append(edge)

    H = np.zeros((num_nodes,len(edges)), dtype=np.int32)
    for i,edge in enumerate(edges):
        for node in edge:
            H[node,i] = 1
    
    node_names = np.cumsum(np.any(H, axis=1)) - 1 # 0-index the nodes that will remain in Hypergraph (i.e. non-isolated nodes)
    edge_names = [f"e{edge:04}" for edge in range(len(edges))]
    
    hgraph = hnx.Hypergraph.from_incidence_matrix(H, node_names, edge_names)

    return hgraph



def generate_random_hypergraph(num_nodes, num_edges):
    """
    Generates random hypergraph from random bipartite graph
    """
    
    graph = gnmk_random_graph(num_edges, num_nodes, 3*num_edges)
    hgraph = hnx.Hypergraph.from_bipartite(graph) # first bipartite part are edges, second part are nodes

    # reindex the nodes and edges from 0
    H = hgraph.incidence_matrix().toarray()
    num_nodes, num_edges = H.shape
    node_names = range(num_nodes)
    edge_names = [f"e{edge:04}" for edge in range(num_edges)]
    hgraph = hnx.Hypergraph.from_incidence_matrix(H, node_names, edge_names)

    return hgraph



def get_trivial_features(labels, feature_type: str):

    assert len(labels.shape) == 1
    num_nodes = len(labels)

    if feature_type == "zeros":
        return torch.zeros((num_nodes, 1), dtype=torch.float32)
    elif feature_type == "ones":
        return torch.ones((num_nodes, 1), dtype=torch.float32)
    elif feature_type == "randn":
        return torch.randn((num_nodes, 1), dtype=torch.float32)
    else:
        raise NotImplementedError



def get_multiclass_normal_features(labels, num_classes: int, dim_feat: int = 16, w: float = 10.0, sigma: float = 1.0):

    assert w >= 0
    assert len(labels.shape) == 1
    assert torch.all(labels < num_classes)

    # standard uniform between -w and w
    mu_motif_type = torch.rand((num_classes, dim_feat)) * 2*w - w
    mu = mu_motif_type[labels]
    
    features = torch.normal(mean=mu, std=sigma).to(torch.float32)

    return features



if __name__ == "__main__":

    import matplotlib.pyplot as plt

    my_num_house_types = 3
    my_labels_type = torch.arange(my_num_house_types+1)
    my_labels_type = torch.tile(my_labels_type, (20,))

    my_features = get_multiclass_normal_features(my_labels_type, my_num_house_types+1, dim_feat=3, w=10.0, sigma=1.0)

    fig = plt.figure()
    ax = plt.axes(projection ='3d')
    x, y, z = my_features.split(split_size=(1,1,1), dim=1)
    ax.scatter(x, y, z, c=my_labels_type, cmap="Dark2")
    ax.set_title('features')
    plt.show()
