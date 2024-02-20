import torch
from numpy.random import choice
from numpy import floor


def attach_house_to_incidence_dict(node_anchor: int, incidence_dict, num_nodes: int, num_edges: int, with_outer_hedge: bool = True):
    """
    Attaches a house motif to base graph given by incidence_dict, connected with an order-2 hyperedge to node_anchor

    edge names must be by convention: e0001, e0002, ...
    node names must be by convention: 0, 1, 2, ...

    Returns
        incidence_dict: updated incidence dictionary with single house decoration
    """

    assert num_edges == len(incidence_dict)

    incidence_dict[f"e{num_edges  :04}"] = [node_anchor, num_nodes+1]                           # join to base graph
    incidence_dict[f"e{num_edges+1:04}"] = [num_nodes+0, num_nodes+1, num_nodes+2]              # top and middle of house
    incidence_dict[f"e{num_edges+2:04}"] = [num_nodes+1, num_nodes+2, num_nodes+3, num_nodes+4] # middle and botom of house
    if with_outer_hedge:
        incidence_dict[f"e{num_edges+3:04}"] = [num_nodes+0, num_nodes+1, num_nodes+2, num_nodes+3, num_nodes+4]

    return incidence_dict



def attach_houses_to_incidence_dict(node_anchors, incidence_dict, num_nodes: int, num_edges: int, num_house_types: int = 1, with_outer_hedge: bool = True):
    """
    Attach house motifs to base graph given by incidence_dict, attached at node_anchors

    edge names must be by convention: e0001, e0002, ...
    node names must be by convention: 0, 1, 2, ...

    Returns
        incidence_dict: updated incidence dictionary with house decorations
        labels: node labels where 0 = base graph, 1 = top of house, 2 = middle of house, 3 = bottom of house
    """

    num_nodes_final = (num_nodes + len(node_anchors) * 5)
    labels_structure = torch.zeros(num_nodes_final, dtype=torch.int64)
    labels_type = torch.zeros(num_nodes_final, dtype=torch.int64)

    for i,node_anchor in enumerate(node_anchors):
        
        incidence_dict = attach_house_to_incidence_dict(node_anchor, incidence_dict, num_nodes, num_edges, with_outer_hedge)
        
        labels_structure[num_nodes + 0] = 1
        labels_structure[num_nodes + 1] = 2
        labels_structure[num_nodes + 2] = 2
        labels_structure[num_nodes + 3] = 3
        labels_structure[num_nodes + 4] = 3

        house_type = floor(i / (len(node_anchors) / num_house_types)) + 1
        labels_type[num_nodes + 0] = house_type
        labels_type[num_nodes + 1] = house_type
        labels_type[num_nodes + 2] = house_type
        labels_type[num_nodes + 3] = house_type
        labels_type[num_nodes + 4] = house_type
        
        num_nodes = num_nodes + 5
        num_edges = num_edges + 4 if with_outer_hedge else num_edges + 3
        assert num_edges == len(incidence_dict)
    

    return incidence_dict, labels_structure, labels_type



def add_random_edges_to_incidence_dict(num_random_edges: int, incidence_dict, num_nodes: int, num_edges: int, k: int = 2):

    assert num_edges == len(incidence_dict)
    ind_edge = num_edges

    for _ in range(num_random_edges):
        incidence_dict[f"e{ind_edge:04}"] = choice(range(num_nodes), size=k, replace=False).tolist()
        ind_edge += 1
    
    return incidence_dict



def unify_labels(labels_structure, labels_type):
    assert labels_structure.shape == labels_type.shape
    labels = labels_structure + 3 * torch.clip(labels_type-1, min=0, max=None)
    return labels



def split_labels(labels):
    labels_type = (labels-1).div(3, rounding_mode="floor") + 1
    labels_structure = labels - (labels-1).div(3, rounding_mode="trunc") * 3
    return labels_structure, labels_type



if __name__ == "__main__":

    import networkx as nx
    import hypernetx as hnx
    from random_hgraph import generate_random_hypergraph

    h1 = generate_random_hypergraph(num_nodes=10, num_edges=6)

    h2_incdict, h2_labels_strc, h2_labels_type = attach_houses_to_incidence_dict([0,1,2,], h1.incidence_dict, h1.number_of_nodes(), h1.number_of_edges(), num_house_types=3)
    h2 = hnx.Hypergraph(h2_incdict)
    h3 = hnx.Hypergraph(add_random_edges_to_incidence_dict(num_random_edges=3, incidence_dict=h2.incidence_dict, num_nodes=h2.number_of_nodes(), num_edges=h2.number_of_edges(), k=3))

    print(h1.shape)
    print(h2.shape)
    print(h3.shape)

    print(h1.incidence_dict)
    print(h2.incidence_dict)
    print(h3.incidence_dict)

    hnx.draw(h1, layout=nx.spring_layout)
    hnx.draw(h2, layout=nx.spring_layout)
    hnx.draw(h3, layout=nx.spring_layout)

    h2_labels = unify_labels(h2_labels_strc, h2_labels_type)

    print(h2_labels_strc)
    print(h2_labels_type)
    print(h2_labels)

    h2_labels_strc_recon, h2_labels_type_recon = split_labels(h2_labels)
    assert torch.equal(h2_labels_strc, h2_labels_strc_recon)
    assert torch.equal(h2_labels_type, h2_labels_type_recon)

