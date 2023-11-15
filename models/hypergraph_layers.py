import torch
import torch.nn as nn


class MyHypergraphConv(nn.Module):
    # c.f. https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.nn.conv.HypergraphConv.html

    def __init__(self, input_dim, output_dim):
        """
        One layer of a hyper neural net

        Args:
            input_dim: dimension of input features per node
            output_dim: dimension of output features per node
        """

        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.Linear = nn.Linear(input_dim, output_dim) # implements W
    
    def forward(self, x, H):
        """
        Args:
            x: feature matrix [num_nodes, input_dim]
            H: incidence matrix [num_nodes, num_hyper_edges]
        Returns:
            x: output of one HNN layer [num_nodes, output_dim]
        """

        deg_node_inv_half = torch.nan_to_num(torch.pow(torch.diag(torch.sum(H, axis=1)), -0.5), nan=0, posinf=0, neginf=0).to(torch.float32) # (D_v)^(-0.5)
        deg_edge_inv       = torch.nan_to_num(torch.pow(torch.diag(torch.sum(H, axis=0)), -1.0), nan=0, posinf=0, neginf=0).to(torch.float32) # (D_e)^(-1)

        x = deg_node_inv_half @ H @ deg_edge_inv @ H.T @ deg_node_inv_half @ x
        x = self.Linear(x)

        return x
    


class MyHypergraphConvResid(nn.Module):

    def __init__(self, input_dim, output_dim, alpha, beta):
        """
        One layer of a hyper neural net

        Args:
            input_dim: dimension of input features per node
            output_dim: dimension of output features per node
        """

        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.Linear = nn.Linear(input_dim, output_dim) # implements W
        self.alpha = alpha
        self.beta = beta
    
    def forward(self, x, H, x0):
        """
        Args:
            x: feature matrix [num_nodes, input_dim]
            H: incidence matrix [num_nodes, num_hyper_edges]
        Returns:
            x: output of one HNN layer [num_nodes, output_dim]
        """

        deg_node_inv_half = torch.nan_to_num(torch.pow(torch.diag(torch.sum(H, axis=1)), -0.5), nan=0, posinf=0, neginf=0).to(torch.float32) # (D_v)^(-0.5)
        deg_edge_inv       = torch.nan_to_num(torch.pow(torch.diag(torch.sum(H, axis=0)), -1.0), nan=0, posinf=0, neginf=0).to(torch.float32) # (D_e)^(-1)
        
        x = (1 - self.alpha) * deg_node_inv_half @ H @ deg_edge_inv @ H.T @ deg_node_inv_half @ x + self.alpha * x0
        x = self.beta * x + (1 - self.beta) * self.Linear(x)

        return x