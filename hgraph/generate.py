import torch
import hypernetx as hnx

from .random_hgraph import get_trivial_features, get_multiclass_normal_features
from .load_coraca import get_coraca_hypergraph
from .random_hgraph import generate_random_hypergraph, generate_random_uniform_hypergraph
from .motif import unify_labels, attach_houses_to_incidence_dict, add_random_edges_to_incidence_dict
from .utils import incidence_matrix_to_edge_index, load_hgraph, get_split_mask



def make_random_base_graph(cfg):

    if cfg.base == "random":
        hgraph = generate_random_hypergraph(cfg.num_nodes, cfg.num_edges)
    elif cfg.base == "random_unif":
        hgraph = generate_random_uniform_hypergraph(cfg.num_nodes, cfg.num_edges, cfg.k)
    else:
        raise NotImplementedError
    
    return hgraph



def decorate_and_perturb(cfg, h_base):

    # anchor_nodes = torch.randint(low=0, high=h_base.number_of_nodes(), size=(num_houses,)).tolist()
    anchor_nodes = range(cfg.num_houses)
    incdict, labels_strc, labels_type = attach_houses_to_incidence_dict(
        anchor_nodes,
        h_base.incidence_dict,
        h_base.number_of_nodes(),
        h_base.number_of_edges(),
        cfg.num_house_types
    )

    num_nodes = len(set([node for lst in incdict.values() for node in lst]))
    num_edges = len(incdict)
    assert num_nodes == h_base.number_of_nodes() + 4 * cfg.num_houses
    assert num_edges == h_base.number_of_edges() + 3 * cfg.num_houses

    incdict = add_random_edges_to_incidence_dict(cfg.num_random_edges, incdict, num_nodes, num_edges, cfg.deg_random_edges)

    hgraph = hnx.Hypergraph(incdict)

    return hgraph, labels_strc, labels_type



def populate_attributes(cfg, hgraph, labels_strc, labels_type):

    if cfg.features in ["zeros", "ones", "randn"]:
        hgraph.x = get_trivial_features(labels_strc, feature_type=cfg.features)
    elif cfg.features == "multiclass_normal":
        hgraph.x = get_multiclass_normal_features(labels_type, cfg.num_house_types+1, cfg.dim_feat, cfg.w, cfg.sigma)
    else:
        raise NotImplementedError
    hgraph.y = unify_labels(labels_strc, labels_type)
    hgraph.H = torch.tensor(hgraph.incidence_matrix().toarray())
    hgraph.edge_index = incidence_matrix_to_edge_index(hgraph.H)

    train_mask, val_mask, test_mask = get_split_mask(n=hgraph.number_of_nodes(), stratify=hgraph.y, split=[0.8, 0.2, 0.0], seed=3)
    hgraph.train_mask = train_mask
    hgraph.val_mask = val_mask
    hgraph.test_mask = test_mask

    hgraph.num_house_types = 1
    hgraph.num_classes = hgraph.num_house_types * 3 + 1

    return hgraph



def make_random_house(cfg):

    h_base = make_random_base_graph(cfg)

    hgraph, labels_strc, labels_type = decorate_and_perturb(cfg, h_base)
    print(f"{h_base.number_of_nodes()=}, {h_base.number_of_edges()=}")
    print(f"{hgraph.number_of_nodes()=}, {hgraph.number_of_edges()=}")

    populate_attributes(cfg, hgraph, labels_strc, labels_type)

    return hgraph



def make_hgraph(cfg):

    if "path" in cfg:
        print(f"Loading pre-existing graph at {cfg.path}. Ignoring other hgraph configs.")
        hgraph = load_hgraph(cfg.path)
        return hgraph

    if cfg.base == "coraca":
        hgraph = get_coraca_hypergraph(split=[0.5, 0.25, 0.25], split_seed=cfg.split_seed)
    elif cfg.base == "random":
        hgraph = make_random_house(cfg)
    elif cfg.name == "random_unif":
        hgraph = make_random_house(cfg)
    else:
        raise NotImplementedError
    
    return hgraph
