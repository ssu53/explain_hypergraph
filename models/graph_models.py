import torch.nn as nn
from torch_geometric.nn import GCNConv



class GCN(nn.Module):
    # c.f. https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.nn.models.GCN.html

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=3):
        """
        Args:
            input_dim: dimension of input features per node
            output_dim: dimension of output features per node
            hidden_dim: dimension of hidden features per node
            num_layers: number of layers
        """

        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers

        self.gcn_layers = []
        if num_layers > 1:
            self.gcn_layers += [GCNConv(input_dim, hidden_dim)]
            self.gcn_layers += [GCNConv(hidden_dim, hidden_dim) for _ in range(num_layers-2)]
            self.gcn_layers += [GCNConv(hidden_dim, output_dim)]
        else:
            self.gcn_layers += [GCNConv(input_dim, output_dim)]
        self.gcn_layers = nn.ModuleList(self.gcn_layers)
    

    def forward(self, x, edge_index):
        """
        Args:
            graph: graph object (wraps features x and adjacency matrix A)
        Returns:
            y: logits for each node [num_nodes, output_dim]
        """

        for ind_layer,layer in enumerate(self.gcn_layers):
            x = layer(x, edge_index)
            if ind_layer != self.num_layers-1: # no activation before logits
                x = nn.functional.relu(x)
        
        return x