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



def get_local_hypergraph(node_idx, hgraph: hnx.Hypergraph, num_expansions: int, graph_data=None) -> hnx.Hypergraph:

    assert isinstance(hgraph, hnx.Hypergraph)
    if isinstance(node_idx, int) or isinstance(node_idx, np.integer): node_idx = [node_idx]

    neighb_nodes = set(node_idx)
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
    
    H_neighb = hnx.Hypergraph(neighb_dict)

    if graph_data is not None: raise NotImplementedError

    return H_neighb



def get_local_hypergraphs(node_idxs, y, hgraph, num_expansions, graph_data=None):

    graphs = []
    color_maps = []
    labels = []
    node_labels = []
    
    for node_idx in node_idxs:
        
        neighb_hgraph = get_local_hypergraph(node_idx, hgraph, num_expansions, graph_data)
        
        color_map = [] # how to plot node color into hypergraph?
        node_label = {}
        
        graphs.append(neighb_hgraph)
        color_maps.append(color_map)
        labels.append(y[node_idx])
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



def plot_samples(activ, kmeans_model, y, hgraph, num_expansions, num_nodes_view=2, path=None, graph_data=None):

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

            tg, cm, labels, node_labels = get_local_hypergraphs(top_indices, y, hgraph, num_expansions, graph_data)
            top_graphs = top_graphs + tg
            color_maps = color_maps + cm

        if graph_data is None:
            for ax, new_G, node_idx, g_label in zip(ax_list, top_graphs, top_indices, labels):
                hnx.draw(new_G.collapse_nodes(), ax=ax, with_node_counts=True, with_node_labels=True)
                ax.set_title(f"label {g_label} node {node_idx}", fontsize=14)
        else:
            raise NotImplementedError

        sample_graphs.append((top_graphs[0], top_indices[0]))
        sample_feat.append(color_maps[0])

    if path is not None:
        pass 

    plt.show()

    return sample_graphs, sample_feat