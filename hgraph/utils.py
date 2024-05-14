import torch
import pickle
import hypernetx as hnx
from sklearn.model_selection import train_test_split



def EDGE_IDX2NAME(edge_idx: int):
    return f"e{edge_idx:04}"



def EDGE_NAME2IDX(edge_name: str):
    return int(edge_name.replace('e',''))



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



def get_split_mask(n, stratify, split, seed):

    assert len(split) == 3 # train, val, test
    train_frac, val_frac, test_frac = split

    node_inds = range(n)

    train_inds, nontrain_inds = train_test_split(
        node_inds,
        train_size=train_frac,
        random_state=seed,
        stratify=stratify)

    if test_frac == 0:
        val_inds = nontrain_inds
        test_inds = []
    elif val_frac == 0:
        val_inds = []
        test_inds = nontrain_inds
    else:
        val_inds, test_inds = train_test_split(
            nontrain_inds,
            train_size=split[1]/(split[1]+split[2]),
            random_state=seed,
            stratify=stratify[nontrain_inds] if stratify is not None else None)
    
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
        num_house_types = hgraph.num_house_types if hasattr(hgraph, 'num_house_types') else None,
        num_classes = hgraph.num_classes,
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
    try:
        hgraph.num_house_types = hgraph_dict['num_house_types']
        hgraph.num_classes = hgraph_dict['num_classes']
    except:
        pass
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
