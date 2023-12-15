import torch
import hypernetx as hnx
import dhg

from train_utils import get_train_val_test_mask


def get_hgraph_from_edgelist(num_nodes: int, num_edges: int, edge_list, add_self_edges: bool = True) -> hnx.Hypergraph:
    """
    num_nodes: number of nodes
    num_edges: number of edges
    edge_list: list of lists of nodes joined by a hyperedge e.g.
            [[235, 355], [1133, 1666, 1888], [783, 785], ...]
    """

    assert len(edge_list) == num_edges

    edge_dict = {}

    for i, edge in enumerate(edge_list):
        edge_dict[f"e{i}"] = edge
    
    if add_self_edges:
        for i in range(num_nodes):
            edge_dict[f"self-e{i}"] = [i]
    else:
        raise NotImplementedError # need to deal with isolated nodes being lost by hnx.Hypergraph
    
    hgraph = hnx.Hypergraph(edge_dict)

    return hgraph



def incidence_matrix_to_edge_index(H):

    nodes, edges = torch.where(H > 0)
    edge_index = torch.vstack((nodes, edges))

    return edge_index



def incidence_matrix_to_incidence_dict(H):

    _, num_edges = H.shape

    incidence_dict = {}
    for edge in range(num_edges):
        inds = torch.where(H[:,edge] == 1)[0]
        incidence_dict[f"e{edge:04}"] = inds.tolist()
    return incidence_dict



def get_train_val_test_mask_standardsplit():
    # https://github.com/malllabiisc/HyperGCN
    pass



def get_coraca_hypergraph(split=[0.5, 0.25, 0.25], split_seed=3) -> hnx.Hypergraph:

    coraca_dhg = dhg.data.CoauthorshipCora(data_root='data')
    hgraph_coraca = get_hgraph_from_edgelist(coraca_dhg['num_vertices'], coraca_dhg['num_edges'], coraca_dhg['edge_list'])

    hgraph_coraca.x = coraca_dhg['features']
    hgraph_coraca.y = coraca_dhg['labels']

    # do not use the dhg's split for this dataset
    train_mask, val_mask, test_mask = get_train_val_test_mask(n=coraca_dhg['num_vertices'], split=split, seed=split_seed)
    hgraph_coraca.train_mask = train_mask
    hgraph_coraca.val_mask = val_mask
    hgraph_coraca.test_mask = test_mask

    hgraph_coraca.H = torch.tensor(hgraph_coraca.incidence_matrix().toarray())
    hgraph_coraca.edge_index = incidence_matrix_to_edge_index(hgraph_coraca.H)

    hgraph_coraca.name = 'Cora Co-Authorship'

    assert hgraph_coraca.H.sum().item() == hgraph_coraca.edge_index.shape[1]

    return hgraph_coraca