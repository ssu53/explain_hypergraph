import numpy as np
import matplotlib.pyplot as plt
import hypernetx as hnx
from sklearn.cluster import KMeans

from typing import List



def get_index_of_node(hgraph: hnx.Hypergraph, node: str) -> int:
    return np.where(hgraph._state_dict['labels']['nodes'] == node)[0].item()



def get_index_of_edge(hgraph: hnx.Hypergraph, edge: str) -> int:
    return np.where(hgraph._state_dict['labels']['edges'] == edge)[0].item()



def get_edges_of_nodes(hgraph: hnx.Hypergraph, nodes: List[int]):
    
    im = hgraph.incidence_matrix().toarray()
    nodes = [get_index_of_node(hgraph, n) for n in nodes]
    im_nodes = im[nodes, :]
    edges = im_nodes.sum(axis=0).nonzero()[0]
    edges = {hgraph._state_dict['labels']['edges'][i] for i in edges}
    
    return edges



def get_nodes_of_edges(hgraph: hnx.Hypergraph, edges: List[str]):
    
    im = hgraph.incidence_matrix().toarray()
    edges = [get_index_of_edge(hgraph, e) for e in edges]
    im_edges = im[:, edges]
    nodes = im_edges.sum(axis=1).nonzero()[0]
    nodes = {hgraph._state_dict['labels']['nodes'][i] for i in nodes}
    
    return nodes



def get_local_hypergraph(idx, hgraph: hnx.Hypergraph, num_expansions: int, is_hedge_concept: bool, graph_data=None) -> hnx.Hypergraph:

    assert isinstance(hgraph, hnx.Hypergraph)
    if isinstance(idx, int) or isinstance(idx, np.integer): idx = [idx]

    if is_hedge_concept:

        neighb_nodes = set()
        neighb_edges = set(["e" + "{0:0>4}".format(i) for i in idx])

        for _ in range(num_expansions):

            neighb_edges_new = set()

            for edge in neighb_edges:
                neighb_edges_new.update(hgraph.edge_neighbors(edge))
            
            neighb_nodes.update(get_nodes_of_edges(hgraph, list(neighb_edges)))

            neighb_edges.update(neighb_edges_new)

    else:

        neighb_nodes = set(idx)
        neighb_edges = set()
        
        for _ in range(num_expansions):

            neighb_nodes_new = set()

            for node in neighb_nodes:
                neighb_nodes_new.update(hgraph.neighbors(node))

            # neighb_edges.update(hgraph.restrict_to_nodes(neighb_nodes).edges)
            neighb_edges.update(get_edges_of_nodes(hgraph, list(neighb_nodes)))

            neighb_nodes.update(neighb_nodes_new)


    # make hypergraph comprising neighb_nodes and neighb_edges
    neighb_dict = {}
    for edge in neighb_edges:
        # nodes_in_edge = hgraph.restrict_to_edges([edge]).nodes
        nodes_in_edge = get_nodes_of_edges(hgraph, [edge])
        neighb_dict[edge] = [node for node in nodes_in_edge if node in neighb_nodes]
    
    hgraph_local = hnx.Hypergraph(neighb_dict)

    if graph_data is not None: raise NotImplementedError

    return hgraph_local



def get_local_hypergraphs(idxs, y, hgraph, num_expansions, is_hedge_concept, graph_data=None):

    graphs = []
    color_maps = []
    labels = []
    node_labels = []
    
    for idx in idxs:
        
        neighb_hgraph = get_local_hypergraph(idx, hgraph, num_expansions, is_hedge_concept, graph_data)
        
        color_map = [] # how to plot node color into hypergraph?
        node_label = {}
        
        graphs.append(neighb_hgraph)
        color_maps.append(color_map)
        labels.append(y[idx])
        node_labels.append(node_label)

    return graphs, color_maps, labels, node_labels



def get_node_distances(kmeans_model, data):
    """
    Returns:
        [num_nodes, num_clusters] distance to each cluster
    """
    try:
        assert isinstance(kmeans_model, KMeans)
        res_sorted = kmeans_model.transform(data)
    except:
        raise NotImplementedError
    return res_sorted



def plot_samples(activ, kmeans_model, y, hgraph, num_expansions, num_nodes_view=2, path=None, is_hedge_concept=False, graph_data=None):

    if is_hedge_concept:
        assert activ.size(0) == hgraph.number_of_edges()
    else:
        assert activ.size(0) == hgraph.number_of_nodes()

    assert isinstance(kmeans_model, KMeans)
    num_clusters = kmeans_model.n_clusters

    res_sorted = get_node_distances(kmeans_model, activ)

    if isinstance(num_nodes_view, int): num_nodes_view = [num_nodes_view]
    col = sum([abs(number) for number in num_nodes_view])

    fig, axes = plt.subplots(num_clusters, col, figsize=(18, 3 * num_clusters + 2))
    fig.suptitle(f'Nearest Instances to Cluster Centroid for Activations of Layer', y=1.005)

    if graph_data is not None:
        fig2, axes2 = plt.subplots(num_clusters, col, figsize=(18, 3 * num_clusters + 2))
        fig2.suptitle(f'Nearest Instances to Cluster Centroid for Activations of Layer (by node index)', y=1.005)

    l = list(range(0, num_clusters))
    sample_graphs = []
    sample_feat = []

    for i, ax_list in zip(l, axes):
        
        distances = res_sorted[:, i]

        top_graphs, color_maps = [], []
        for view in num_nodes_view:
            if view < 0:
                top_indices = np.argsort(distances)[::][view:]
            else:
                top_indices = np.argsort(distances)[::][:view]

            tg, cm, labels, node_labels = get_local_hypergraphs(top_indices, y, hgraph, num_expansions, is_hedge_concept, graph_data)
            top_graphs = top_graphs + tg
            color_maps = color_maps + cm

        if graph_data is None:
            for ax, new_G, idx, g_label in zip(ax_list, top_graphs, top_indices, labels):
                hnx.draw(new_G, ax=ax, with_node_counts=True, with_node_labels=True)
                ax.set_title(f"label {g_label} {'hedge' if is_hedge_concept else 'node'} {idx}", fontsize=14)
        else:
            raise NotImplementedError

        sample_graphs.append((top_graphs[0], top_indices[0]))
        sample_feat.append(color_maps[0])

    if path is not None:
        pass 

    plt.show()

    return sample_graphs, sample_feat