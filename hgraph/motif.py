import torch
from numpy.random import choice



def attach_house_to_incidence_dict(node_anchor: int, incidence_dict, num_nodes: int, num_edges: int):
    """
    Attaches a house motif to base graph given by incidence_dict, attached at node_anchor

    edge names must be by convention: e0001, e0002, ...
    node names must be by convention: 0, 1, 2, ...

    Returns
        incidence_dict: updated incidence dictionary with single house decoration
    """

    assert num_edges == len(incidence_dict)

    incidence_dict[f"e{num_edges  :04}"] = [node_anchor, num_nodes+0, num_nodes+1]
    incidence_dict[f"e{num_edges+1:04}"] = [node_anchor, num_nodes+1, num_nodes+2, num_nodes+3]
    incidence_dict[f"e{num_edges+2:04}"] = [node_anchor, num_nodes+0, num_nodes+1, num_nodes+2, num_nodes+3]

    return incidence_dict



def attach_houses_to_incidence_dict(node_anchors, incidence_dict, num_nodes: int, num_edges: int):
    """
    Attach house motifs to base graph given by incidence_dict, attached at node_anchors

    edge names must be by convention: e0001, e0002, ...
    node names must be by convention: 0, 1, 2, ...

    Returns
        incidence_dict: updated incidence dictionary with house decorations
        labels: node labels where 0 = base graph, 1 = top of house, 2 = middle of house, 3 = bottom of house
    """

    labels = torch.zeros((num_nodes + len(node_anchors) * 4), dtype=torch.int64)

    for node_anchor in node_anchors:
        
        incidence_dict = attach_house_to_incidence_dict(node_anchor, incidence_dict, num_nodes, num_edges)
        
        labels[node_anchor] = 2
        labels[num_nodes + 0] = 1
        labels[num_nodes + 1] = 2
        labels[num_nodes + 2] = 3
        labels[num_nodes + 3] = 3
        
        num_nodes = num_nodes + 4
        num_edges = num_edges + 3
    

    return incidence_dict, labels



def add_random_edges_to_incidence_dict(num_random_edges: int, incidence_dict, num_nodes: int, num_edges: int, k: int = 2):

    assert num_edges == len(incidence_dict)
    ind_edge = num_edges

    for _ in range(num_random_edges):
        incidence_dict[f"e{ind_edge:04}"] = choice(range(num_nodes), size=k, replace=False).tolist()
        ind_edge += 1
    
    return incidence_dict



if __name__ == "__main__":

    import networkx as nx
    import hypernetx as hnx
    from random_hgraph import generate_random_hypergraph

    h1 = generate_random_hypergraph(num_nodes=10, num_edges=6)

    h2 = hnx.Hypergraph(add_random_edges_to_incidence_dict(num_random_edges=3, incidence_dict=h1.incidence_dict, num_nodes=h1.number_of_nodes(), num_edges=h1.number_of_edges(), k=3))

    print(h1.shape)
    print(h2.shape)

    print(h1.incidence_dict)
    print(h2.incidence_dict)

    hnx.draw(h2, layout=nx.spring_layout)
