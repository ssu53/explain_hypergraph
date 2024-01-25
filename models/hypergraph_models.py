import numpy as np
import torch
import torch.nn as nn
# from torch_geometric.nn import HypergraphConv
from .hypergraph_layers import MyHypergraphConv, MyHypergraphAtt, MyHypergraphConvResid, HypergraphConv



class HyperGCN(nn.Module):
    
    def __init__(self, input_dim, output_dim, hidden_dim, num_layers, use_attention=False, attention_mode='node', heads=1, bias=True, dropout=0):
        """
        Args:
            input_dim: dimension of input features per node
            output_dim: dimension of output features per node
            hidden_dim: dimension of hidden features per node
            num_layers: number of layers
            use_attention: whether to implement attention in all HypergraphConv layers
            attention_mode: 'node' or 'edge'. default for torch-geometric HypergraphConv is 'node'
            heads: number of heads. default for torch-geometric HypergraphConv is 1
        """

        assert attention_mode in ['node', 'edge'] 

        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.use_attention = use_attention
        self.attention_mode = attention_mode
        self.heads = heads

        self.gcn_layers = []
        if num_layers > 1:
            self.gcn_layers += [HypergraphConv(input_dim, hidden_dim, use_attention, attention_mode, heads, bias=bias)]
            self.gcn_layers += [HypergraphConv(hidden_dim, hidden_dim, use_attention, attention_mode, heads, bias=bias) for _ in range(num_layers-2)]
            self.gcn_layers += [HypergraphConv(hidden_dim, output_dim, use_attention, attention_mode, heads, bias=bias)]
        else:
            self.gcn_layers += [HypergraphConv(input_dim, output_dim, use_attention, attention_mode, heads)]
        self.gcn_layers = nn.ModuleList(self.gcn_layers)

        self.drop = nn.Dropout(p=dropout)

        self.emb_layer = self.gcn_layers[-2] if num_layers > 1 else None # this layer's activations are the final latent embedding
    
    
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
                x = layer(x, edge_index, hyperedge_attr=edge_feat)
            else:
                x = layer(x, edge_index)
            if ind_layer != self.num_layers-1: # no activation before logits
                x = nn.functional.relu(x)
                x = self.drop(x)
        
        return x
    


class MyHyperGCN(nn.Module):
    
    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int, num_layers: int, symmetric_norm: bool = True, bias: bool = True, use_attention=False, dropout=0):
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

        self.symmetric_norm = symmetric_norm
        self.bias = bias

        self.use_attention = use_attention
        if use_attention:   MyHypergraphLayer = MyHypergraphAtt
        else:               MyHypergraphLayer = MyHypergraphConv

        self.gcn_layers = []
        if num_layers > 1:
            self.gcn_layers += [MyHypergraphLayer(input_dim, hidden_dim, symmetric_norm, bias)]
            self.gcn_layers += [MyHypergraphLayer(hidden_dim, hidden_dim, symmetric_norm, bias) for _ in range(num_layers-2)]
            self.gcn_layers += [MyHypergraphLayer(hidden_dim, output_dim, symmetric_norm, bias)]
        else:
            self.gcn_layers += [MyHypergraphLayer(input_dim, output_dim, symmetric_norm, bias)]
        self.gcn_layers = nn.ModuleList(self.gcn_layers)

        self.drop = nn.Dropout(p=dropout)

        self.emb_layer = self.gcn_layers[-2] if num_layers > 1 else None # this layer's activations are the final latent embedding

    
    def forward(self, hgraph):
        """
        Args:
            hgraph: hypergraph object (needs attributes x and incidence matrix H)
        Returns:
            y: logits for each node [num_nodes, output_dim]
        """

        x = hgraph.x.to(torch.float32)
        H = hgraph.H.to(torch.float32)

        for ind_layer,layer in enumerate(self.gcn_layers):
            x = layer(x, H)
            if ind_layer != self.num_layers-1: # no activation before logits
                x = torch.nn.functional.relu(x)
                x = self.drop(x)
        
        return x
    

class HyperResidGCN(nn.Module):
    
    def __init__(self, input_dim, output_dim, hidden_dim, num_layers, alpha, beta, dropout=0, symmetric_norm: bool = True, bias: bool = True):
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

        self.symmetric_norm = symmetric_norm
        self.bias = bias

        self.fc_in = nn.Linear(input_dim, hidden_dim)

        self.gcn_layers = [MyHypergraphConvResid(hidden_dim, hidden_dim, alpha, np.log(beta / (ind_layer+1) + 1), symmetric_norm, bias) for ind_layer in range(num_layers)]
        self.gcn_layers = nn.ModuleList(self.gcn_layers)

        self.fc_out = nn.Linear(hidden_dim, output_dim)
        self.drop = nn.Dropout(p=dropout)

        self.emb_layer = self.gcn_layers[-1] # this layer's activations are the final latent embedding
    
    
    def forward(self, hgraph):
        """
        Args:
            hgraph: hypergraph object (needs attributes x and incidence matrix H)
        Returns:
            y: logits for each node [num_nodes, output_dim]
        """

        x = hgraph.x.to(torch.float32)
        H = hgraph.H.to(torch.float32)

        x = self.fc_in(x)
        x0 = x

        for ind_layer,layer in enumerate(self.gcn_layers):
            x = self.drop(x)
            x = layer(x, H, x0)
            if ind_layer != self.num_layers-1: # no activation before logits
                x = torch.nn.functional.relu(x)
        
        x = self.drop(x)
        x = self.fc_out(x)
        
        return x