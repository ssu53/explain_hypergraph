import torch
from numpy.random import choice
from numpy import floor
from enum import Enum

from typing import List

from .utils import EDGE_IDX2NAME



class House(Enum):
    Base = 0
    Top = 1
    Middle = 2
    Bottom = 3


class HouseGranular(Enum):
    Base_Anchor = 0
    Base_Other = 1
    Top = 2
    Middle_Unanchored = 3
    Middle_Anchored = 4
    Bottom = 5


class Cycle(Enum):
    Base = 0
    Cycle = 1


class CycleGranular(Enum):
    Base_Anchor = 0
    Base_Other = 1 # can be further divided into leaf, root, rest
    Cycle_Anchor = 2
    Cycle_1fromAnchor = 3
    Cycle_2fromAnchor = 4
    Cycle_OppAnchor = 5


class Grid(Enum):
    Base = 0
    Grid = 1


class GridGranular(Enum):
    Base_Anchor = 0
    Base_Other = 1 # can be further divided into leaf, root, rest
    Grid_Anchor = 2
    Grid_1fromAnchor = 3 # i.e. in the row or col of anchor
    Grid_2fromAnchor = 4 # i.e. the rest in grid




def attach_house_to_incidence_dict(node_anchor: int, incidence_dict, num_nodes: int, num_edges: int, with_outer_hedge: bool = True):
    """
    Attaches a house motif to base graph given by incidence_dict, connected with an order-2 hyperedge to node_anchor

    edge names must be by convention: e0001, e0002, ...
    node names must be by convention: 0, 1, 2, ...

    Returns
        incidence_dict: updated incidence dictionary with single house decoration
    """

    assert num_edges == len(incidence_dict)

    incidence_dict[EDGE_IDX2NAME(num_edges)] = [node_anchor, num_nodes+1]                           # join to base graph
    incidence_dict[EDGE_IDX2NAME(num_edges+1)] = [num_nodes+0, num_nodes+1, num_nodes+2]              # top and middle of house
    incidence_dict[EDGE_IDX2NAME(num_edges+2)] = [num_nodes+1, num_nodes+2, num_nodes+3, num_nodes+4] # middle and botom of house
    if with_outer_hedge:
        incidence_dict[EDGE_IDX2NAME(num_edges+3)] = [num_nodes+0, num_nodes+1, num_nodes+2, num_nodes+3, num_nodes+4]

    return incidence_dict



def attach_cycle_to_incidence_dict(incidence_dict, labels, node_anchor: int, num_nodes: int, num_edges: int):
    """
    this might be too hard
    """

    assert num_edges == len(incidence_dict)

    node0 = num_nodes       # connected to anchor
    node1 = num_nodes + 1  
    node2 = num_nodes + 2
    node3 = num_nodes + 3
    node4 = num_nodes + 4
    node5 = num_nodes + 5

    edge0 = num_edges       # connecting anchor and node0
    edge1 = num_edges + 1
    edge2 = num_edges + 2
    edge3 = num_edges + 3

    for node in [node0, node1, node2, node3, node4, node5]:
        labels[node] = Cycle.Cycle.value

    incidence_dict[EDGE_IDX2NAME(edge0)] = [node_anchor, node0]
    incidence_dict[EDGE_IDX2NAME(edge1)] = [node0, node1, node2]
    incidence_dict[EDGE_IDX2NAME(edge2)] = [node2, node3, node4]
    incidence_dict[EDGE_IDX2NAME(edge3)] = [node4, node5, node0]




def attach_hexblob_to_incidence_dict(incidence_dict, labels, node_anchor: int, num_nodes: int, num_edges: int) -> None:
    """
    """

    assert num_edges == len(incidence_dict)

    node0 = num_nodes       # connected to anchor
    node1 = num_nodes + 1   
    node2 = num_nodes + 2
    node3 = num_nodes + 3
    node4 = num_nodes + 4
    node5 = num_nodes + 5

    edge0 = num_edges       # connecting anchor and node0
    edge1 = num_edges + 1

    for node in [node0, node1, node2, node3, node4, node5]:
        labels[node] = Cycle.Cycle.value

    incidence_dict[EDGE_IDX2NAME(edge0)] = [node_anchor, node0]
    incidence_dict[EDGE_IDX2NAME(edge1)] = [node0, node1, node2, node3, node4, node5]



def attach_grid_to_incidence_dict(incidence_dict, labels, node_anchor: int, num_nodes: int, num_edges: int) -> None:
    """
    """

    assert num_edges == len(incidence_dict)

    node0 = num_nodes       # connected to anchor
    node1 = num_nodes + 1   
    node2 = num_nodes + 2
    node3 = num_nodes + 3
    node4 = num_nodes + 4
    node5 = num_nodes + 5
    node6 = num_nodes + 6
    node7 = num_nodes + 7
    node8 = num_nodes + 8

    edge0 = num_edges       # connecting anchor and node0
    edge1 = num_edges + 1
    edge2 = num_edges + 2
    edge3 = num_edges + 3
    edge4 = num_edges + 4
    edge5 = num_edges + 5
    edge6 = num_edges + 6

    for node in [node0, node1, node2, node3, node4, node5, node6, node7, node8]:
        labels[node] = Grid.Grid.value

    incidence_dict[EDGE_IDX2NAME(edge0)] = [node_anchor, node0]
    incidence_dict[EDGE_IDX2NAME(edge1)] = [node0, node1, node2]
    incidence_dict[EDGE_IDX2NAME(edge2)] = [node3, node4, node5]
    incidence_dict[EDGE_IDX2NAME(edge3)] = [node6, node7, node8]
    incidence_dict[EDGE_IDX2NAME(edge4)] = [node0, node3, node6]
    incidence_dict[EDGE_IDX2NAME(edge5)] = [node1, node4, node7]
    incidence_dict[EDGE_IDX2NAME(edge6)] = [node2, node5, node8]



def attach_motifs_to_incidence_dict(incidence_dict, labels, motif: str, node_anchors: List[int], num_nodes: int, num_edges: int) -> None:
    """
    motifs are uni-coloured
    incidence_dict modified in-place
    labels modified in-place
    """

    if motif == 'cycle':
        attach_motif_func = attach_cycle_to_incidence_dict
        motif_num_nodes = 6
        motif_num_edges = 4
    elif motif == 'grid':
        attach_motif_func = attach_grid_to_incidence_dict
        motif_num_nodes = 9
        motif_num_edges = 7
    elif motif == 'house':
        raise NotImplementedError
    else:
        raise NotImplementedError

    for node_anchor in node_anchors:
        attach_motif_func(incidence_dict, labels, node_anchor, num_nodes, num_edges)
        num_nodes += motif_num_nodes
        num_edges += motif_num_edges
    
    return


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
        incidence_dict[EDGE_IDX2NAME(ind_edge)] = choice(range(num_nodes), size=k, replace=False).tolist()
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

    import matplotlib.pyplot as plt
    import networkx as nx
    import hypernetx as hnx


    # --------------------------------------

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

    plt.figure()
    hnx.draw(h1, layout=nx.spring_layout)
    plt.show()
    plt.figure()
    hnx.draw(h2, layout=nx.spring_layout)
    plt.show()
    plt.figure()
    hnx.draw(h3, layout=nx.spring_layout)
    plt.show()

    h2_labels = unify_labels(h2_labels_strc, h2_labels_type)

    print(h2_labels_strc)
    print(h2_labels_type)
    print(h2_labels)

    h2_labels_strc_recon, h2_labels_type_recon = split_labels(h2_labels)
    assert torch.equal(h2_labels_strc, h2_labels_strc_recon)
    assert torch.equal(h2_labels_type, h2_labels_type_recon)

    # --------------------------------------

    from random_hgraph import generate_hypertrio_tree

    h = generate_hypertrio_tree(depth=3)
    print(h.incidence_dict)

    plt.figure()
    hnx.draw(h)
    plt.show()

    incidence_dict = attach_grid_to_incidence_dict(
        node_anchor=2,
        incidence_dict=h.incidence_dict,
        num_nodes=h.number_of_nodes(),
        num_edges=h.number_of_edges(),
    )
    
    print(incidence_dict)
    plt.figure()
    hnx.draw(hnx.Hypergraph(incidence_dict))
    plt.show()
