import numpy as np
import random

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