import torch
import pickle
import hypernetx as hnx



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



def hgraph_to_dict(hgraph):
    
    dict_hgraph = dict(
        incidence_dict = hgraph.incidence_dict,
        train_mask = hgraph.train_mask,
        val_mask = hgraph.val_mask,
        test_mask = hgraph.test_mask,
        x = hgraph.x,
        y = hgraph.y,
        H = hgraph.H,
        edge_index = hgraph.edge_index,
    )

    return dict_hgraph



def load_hgraph(path):
    with open(path, "rb") as f:
        hgraph_dict = pickle.load(f)
    hgraph = hnx.Hypergraph(hgraph_dict["incidence_dict"])
    hgraph.x = hgraph_dict['x']
    hgraph.y = hgraph_dict['y']
    hgraph.train_mask = hgraph_dict['train_mask']
    hgraph.val_mask = hgraph_dict['val_mask']
    hgraph.test_mask = hgraph_dict['test_mask']
    hgraph.H = hgraph_dict['H']
    hgraph.edge_index = hgraph_dict['edge_index']    
    return hgraph



def put_hgraph_attributes_on_device(hgraph, device) -> None:
    hgraph.train_mask = hgraph.train_mask.to(device)
    hgraph.val_mask = hgraph.val_mask.to(device)
    hgraph.test_mask = hgraph.test_mask.to(device)
    hgraph.x = hgraph.x.to(device)
    hgraph.y = hgraph.y.to(device)
    hgraph.H = hgraph.H.to(device)
    hgraph.edge_index = hgraph.edge_index.to(device)
