import torch



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