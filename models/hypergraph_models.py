import torch
import torch.nn as nn
from torch_geometric.nn import HypergraphConv
from .hypergraph_layers import MyHypergraphConv, MyHypergraphConvResid



class HyperGCN(nn.Module):
    
    def __init__(self, input_dim, output_dim, hidden_dim, num_layers, use_attention=False, dropout=0):
        """
        Args:
            input_dim: dimension of input features per node
            output_dim: dimension of output features per node
            hidden_dim: dimension of hidden features per node
            num_layers: number of layers
            use_attention: whether to implement attention in all HypergraphConv layers
        """

        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.use_attention = use_attention
        if dropout != 0: raise NotImplementedError

        self.gcn_layers = []
        if num_layers > 1:
            self.gcn_layers += [HypergraphConv(input_dim, hidden_dim, use_attention)]
            self.gcn_layers += [HypergraphConv(hidden_dim, hidden_dim, use_attention) for _ in range(num_layers-2)]
            self.gcn_layers += [HypergraphConv(hidden_dim, output_dim, use_attention)]
        else:
            self.gcn_layers += [HypergraphConv(input_dim, output_dim, use_attention)]
        self.gcn_layers = nn.ModuleList(self.gcn_layers)
    
    
    def forward(self, hgraph):
        """
        Args:
            x: node features
            hyperedge_index: 
        Returns:
            y: logits for each node [num_nodes, output_dim]
        """

        x = hgraph.x
        edge_index = hgraph.edge_index
        if self.use_attention: 
            H = hgraph.H.to(torch.float32)

        for ind_layer,layer in enumerate(self.gcn_layers):
            if self.use_attention:  
                edge_feat =  (H.T @ x) / len(x)
                print(f"{x.shape=}")
                print(f"{edge_index.shape=}")
                print(f"{edge_feat.shape=}")
                x = layer(x, edge_index, hyperedge_attr=edge_feat)
            else:
                x = layer(x, edge_index)
            if ind_layer != self.num_layers-1: # no activation before logits
                x = nn.functional.relu(x)
        
        return x
    


class MyHyperGCN(nn.Module):
    
    def __init__(self, input_dim, output_dim, hidden_dim, num_layers, dropout=0):
        """
        Args:
            input_dim: dimension of input features per node
            output_dim: dimension of output features per node
            hidden_dim: dimension of hidden features per node
            num_layers: number of layers
        """

        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        if dropout != 0: raise NotImplementedError

        self.hnn_layers = []
        if num_layers > 1:
            self.hnn_layers += [MyHypergraphConv(input_dim, hidden_dim)]
            self.hnn_layers += [MyHypergraphConv(hidden_dim, hidden_dim) for _ in range(num_layers-2)]
            self.hnn_layers += [MyHypergraphConv(hidden_dim, output_dim)]
        else:
            self.hnn_layers += [MyHypergraphConv(input_dim, output_dim)]
        self.hnn_layers = nn.ModuleList(self.hnn_layers)
    
    def forward(self, hgraph):
        """
        Args:
            hgraph: hypergraph object (needs attributes x and incidence matrix H)
        Returns:
            y: logits for each node [num_nodes, output_dim]
        """

        x = hgraph.x.to(torch.float32)
        H = hgraph.H.to(torch.float32)

        for ind_layer,layer in enumerate(self.hnn_layers):
            x = layer(x, H)
            if ind_layer != self.num_layers-1: # no activation before logits
                x = torch.nn.functional.relu(x)
        
        return x
    

class HyperResidGCN(nn.Module):
    
    def __init__(self, input_dim, output_dim, hidden_dim, num_layers, alpha, beta, dropout=0):
        """
        Args:
            input_dim: dimension of input features per node
            output_dim: dimension of output features per node
            hidden_dim: dimension of hidden features per node
            num_layers: number of layers
            dropout: dropout rate. dropout between all layers.

            For now, treat alpha and beta as constant across all layers
        """

        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.alpha = alpha
        self.beta = beta

        self.fc_in = nn.Linear(input_dim, hidden_dim)

        self.hnn_layers = [MyHypergraphConvResid(hidden_dim, hidden_dim, alpha, beta) for _ in range(num_layers)]
        self.hnn_layers = nn.ModuleList(self.hnn_layers)

        self.fc_out = nn.Linear(hidden_dim, output_dim)
        self.drop = nn.Dropout(p=dropout)
    
    
    def forward(self, hgraph):
        """
        Args:
            hgraph: hypergraph object (needs attributes x and incidence matrix H)
        Returns:
            y: logits for each node [num_nodes, output_dim]
        """

        x = hgraph.x.to(torch.float32)
        H = hgraph.H.to(torch.float32)

        x0 = self.fc_in(x)
        x = x0

        for layer in self.hnn_layers:
            x = self.drop(x)
            x = layer(x, H, x0)
            x = torch.nn.functional.relu(x)
        
        x = self.drop(x)
        x = self.fc_out(x)
        
        return x