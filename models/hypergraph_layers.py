from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Parameter
import einops

from torch_geometric.experimental import disable_dynamic_shapes
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.nn.inits import glorot, zeros
from torch_geometric.utils import scatter, softmax


class MyHypergraphConv(nn.Module):
    # c.f. https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.nn.conv.HypergraphConv.html

    def __init__(self, input_dim: int, output_dim: int, symmetric_norm: bool = True, bias: bool = True):
        """
        One layer of a hyper neural net

        Args:
            input_dim: dimension of input features per node
            output_dim: dimension of output features per node
        """

        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.Linear = nn.Linear(input_dim, output_dim, bias=bias) # implements W
        self.symmetric_norm = symmetric_norm

        self.activ_node = None
        self.activ_hedge = None
    
    def forward(self, x, H, save_activ=False):
        """
        Args:
            x: feature matrix [num_nodes, input_dim]
            H: incidence matrix [num_nodes, num_hyper_edges]
        Returns:
            x: output of one HNN layer [num_nodes, output_dim]
        """

        if self.symmetric_norm:
            deg_node_inv_half = torch.nan_to_num(torch.pow(torch.diag(torch.sum(H, axis=1)), -0.5), nan=0, posinf=0, neginf=0).to(torch.float32) # (D_v)^(-0.5)
            deg_edge_inv       = torch.nan_to_num(torch.pow(torch.diag(torch.sum(H, axis=0)), -1.0), nan=0, posinf=0, neginf=0).to(torch.float32) # (D_e)^(-1)
            x = deg_node_inv_half @ H @ deg_edge_inv @ H.T @ deg_node_inv_half @ x
            if save_activ:
                self.activ_hedge = deg_edge_inv @ H.T @ deg_node_inv_half @ x
        
        else:
            x = H @ H.T @ x
            if save_activ:
                self.activ_hedge = H.T @ x

        x = self.Linear(x)

        if save_activ:
            self.activ_node = x

        return x
    


class MyHypergraphAtt(nn.Module):

    def __init__(self, input_dim: int, output_dim: int, symmetric_norm: bool = True, bias: bool = True):
        """
        One layer of a hyper attention neural net
        Edge features are imputed as the sum of features of contained nodes

        Args:
            input_dim: dimension of input features per node
            output_dim: dimension of output features per node
        """

        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.linear = nn.Linear(input_dim, output_dim, bias=bias) 
        self.symmetric_norm = symmetric_norm

        # for computing attentions i.e. node-hyperedge weights
        self.att_proj1 = nn.Linear(2*input_dim, 2*input_dim)
        self.att_proj2 = nn.Linear(2*input_dim, 1)

        self.activ_node = None
        self.activ_hedge = None
    
    def forward(self, x, H, save_activ=False):
        """
        Args:
            x: feature matrix [num_nodes, input_dim]
            H: incidence matrix [num_nodes, num_hyper_edges]
        Returns:
            x: output of one hypergraph attention layer [num_nodes, output_dim]
        """

        num_nodes, num_hyper_edges = H.shape
        assert x.shape[1] == self.input_dim

        # create stacked features of all node-hyperedge pairs
        node_feats = einops.repeat(x,       "h w -> h r w", r=num_hyper_edges)
        edge_feats = einops.repeat(H.T @ x, "h w -> r h w", r=num_nodes)
        concat_feats = torch.concat((node_feats, edge_feats), dim=-1)
        assert concat_feats.shape == (num_nodes, num_hyper_edges, 2*self.input_dim)

        # compute the attention-weighted incidence matrix H_att
        H_att = self.att_proj2(nn.functional.relu(self.att_proj1(concat_feats)))
        assert H_att.shape == (num_nodes, num_hyper_edges, 1)
        H_att = torch.sigmoid(H_att.squeeze(-1))
        H_att = H_att * H # elementwise multiply to mask out non-incident node-hyperedge pairs

        if self.symmetric_norm:
            deg_node_inv_half = torch.nan_to_num(torch.pow(torch.diag(torch.sum(H, axis=1)), -0.5), nan=0, posinf=0, neginf=0).to(torch.float32) # (D_v)^(-0.5)
            deg_edge_inv      = torch.nan_to_num(torch.pow(torch.diag(torch.sum(H, axis=0)), -1.0), nan=0, posinf=0, neginf=0).to(torch.float32) # (D_e)^(-1)
            x = deg_node_inv_half @ H_att @ deg_edge_inv @ H_att.T @ deg_node_inv_half @ x
            if save_activ:
                self.activ_hedge = deg_edge_inv @ H_att.T @ deg_node_inv_half @ x
        
        else:
            x = H_att @ H_att.T @ x
            if save_activ:
                self.activ_hedge = H_att.T @ x

        
        x = self.linear(x)

        if save_activ:
            self.activ_node = x

        return x



class MyHypergraphConvResid(nn.Module):

    def __init__(self, input_dim, output_dim, alpha, beta, symmetric_norm: bool = True, bias: bool = True):
        """
        One layer of a hyper neural net

        Args:
            input_dim: dimension of input features per node
            output_dim: dimension of output features per node
        """

        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.Linear = nn.Linear(input_dim, output_dim, bias=bias) # implements W
        self.alpha = alpha
        self.beta = beta
        self.symmetric_norm = symmetric_norm

        self.activ_node = None
        self.activ_hedge = None
    
    def forward(self, x, H, x0, save_activ=False):
        """
        Args:
            x: feature matrix [num_nodes, input_dim]
            H: incidence matrix [num_nodes, num_hyper_edges]
        Returns:
            x: output of one HNN layer [num_nodes, output_dim]
        """

        if self.symmetric_norm:
            deg_node_inv_half = torch.nan_to_num(torch.pow(torch.diag(torch.sum(H, axis=1)), -0.5), nan=0, posinf=0, neginf=0).to(torch.float32) # (D_v)^(-0.5)
            deg_edge_inv       = torch.nan_to_num(torch.pow(torch.diag(torch.sum(H, axis=0)), -1.0), nan=0, posinf=0, neginf=0).to(torch.float32) # (D_e)^(-1)
            x = (1 - self.alpha) * deg_node_inv_half @ H @ deg_edge_inv @ H.T @ deg_node_inv_half @ x + self.alpha * x0
            if save_activ:
                self.activ_hedge = deg_edge_inv @ H.T @ deg_node_inv_half @ x
        else:
            x = (1 - self.alpha) * H @ H.T @ x + self.alpha * x0
            if save_activ:
                self.activ_hedge = H.T @ x
        
        x = self.beta * x + (1 - self.beta) * self.Linear(x)

        if save_activ:
            self.activ_node = x

        return x



class HypergraphConv(MessagePassing):
    # Taken from source, with one bug gix
    # https://pytorch-geometric.readthedocs.io/en/latest/_modules/torch_geometric/nn/conv/hypergraph_conv.html#HypergraphConv

    r"""The hypergraph convolutional operator from the `"Hypergraph Convolution
    and Hypergraph Attention" <https://arxiv.org/abs/1901.08150>`_ paper.

    .. math::
        \mathbf{X}^{\prime} = \mathbf{D}^{-1} \mathbf{H} \mathbf{W}
        \mathbf{B}^{-1} \mathbf{H}^{\top} \mathbf{X} \mathbf{\Theta}

    where :math:`\mathbf{H} \in {\{ 0, 1 \}}^{N \times M}` is the incidence
    matrix, :math:`\mathbf{W} \in \mathbb{R}^M` is the diagonal hyperedge
    weight matrix, and
    :math:`\mathbf{D}` and :math:`\mathbf{B}` are the corresponding degree
    matrices.

    For example, in the hypergraph scenario
    :math:`\mathcal{G} = (\mathcal{V}, \mathcal{E})` with
    :math:`\mathcal{V} = \{ 0, 1, 2, 3 \}` and
    :math:`\mathcal{E} = \{ \{ 0, 1, 2 \}, \{ 1, 2, 3 \} \}`, the
    :obj:`hyperedge_index` is represented as:

    .. code-block:: python

        hyperedge_index = torch.tensor([
            [0, 1, 2, 1, 2, 3],
            [0, 0, 0, 1, 1, 1],
        ])

    Args:
        in_channels (int): Size of each input sample, or :obj:`-1` to derive
            the size from the first input(s) to the forward method.
        out_channels (int): Size of each output sample.
        use_attention (bool, optional): If set to :obj:`True`, attention
            will be added to this layer. (default: :obj:`False`)
        attention_mode (str, optional): The mode on how to compute attention.
            If set to :obj:`"node"`, will compute attention scores of nodes
            within all nodes belonging to the same hyperedge.
            If set to :obj:`"edge"`, will compute attention scores of nodes
            across all edges holding this node belongs to.
            (default: :obj:`"node"`)
        heads (int, optional): Number of multi-head-attentions.
            (default: :obj:`1`)
        concat (bool, optional): If set to :obj:`False`, the multi-head
            attentions are averaged instead of concatenated.
            (default: :obj:`True`)
        negative_slope (float, optional): LeakyReLU angle of the negative
            slope. (default: :obj:`0.2`)
        dropout (float, optional): Dropout probability of the normalized
            attention coefficients which exposes each node to a stochastically
            sampled neighborhood during training. (default: :obj:`0`)
        bias (bool, optional): If set to :obj:`False`, the layer will not learn
            an additive bias. (default: :obj:`True`)
        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.nn.conv.MessagePassing`.

    Shapes:
        - **input:**
          node features :math:`(|\mathcal{V}|, F_{in})`,
          hyperedge indices :math:`(|\mathcal{V}|, |\mathcal{E}|)`,
          hyperedge weights :math:`(|\mathcal{E}|)` *(optional)*
          hyperedge features :math:`(|\mathcal{E}|, D)` *(optional)*
        - **output:** node features :math:`(|\mathcal{V}|, F_{out})`
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        use_attention: bool = False,
        attention_mode: str = 'node',
        heads: int = 1,
        concat: bool = True,
        negative_slope: float = 0.2,
        dropout: float = 0,
        bias: bool = True,
        **kwargs,
    ):
        kwargs.setdefault('aggr', 'add')
        super().__init__(flow='source_to_target', node_dim=0, **kwargs)

        assert attention_mode in ['node', 'edge']

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.use_attention = use_attention
        self.attention_mode = attention_mode

        if self.use_attention:
            self.heads = heads
            self.concat = concat
            self.negative_slope = negative_slope
            self.dropout = dropout
            self.lin = Linear(in_channels, heads * out_channels, bias=False,
                              weight_initializer='glorot')
            self.att = Parameter(torch.empty(1, heads, 2 * out_channels))
        else:
            self.heads = 1
            self.concat = True
            self.lin = Linear(in_channels, out_channels, bias=False,
                              weight_initializer='glorot')

        if bias and concat:
            self.bias = Parameter(torch.empty(heads * out_channels))
        elif bias and not concat:
            self.bias = Parameter(torch.empty(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        super().reset_parameters()
        self.lin.reset_parameters()
        if self.use_attention:
            glorot(self.att)
        zeros(self.bias)

    @disable_dynamic_shapes(required_args=['num_edges'])
    def forward(self, x: Tensor, hyperedge_index: Tensor,
                hyperedge_weight: Optional[Tensor] = None,
                hyperedge_attr: Optional[Tensor] = None,
                num_edges: Optional[int] = None) -> Tensor:
        r"""Runs the forward pass of the module.

        Args:
            x (torch.Tensor): Node feature matrix
                :math:`\mathbf{X} \in \mathbb{R}^{N \times F}`.
            hyperedge_index (torch.Tensor): The hyperedge indices, *i.e.*
                the sparse incidence matrix
                :math:`\mathbf{H} \in {\{ 0, 1 \}}^{N \times M}` mapping from
                nodes to edges.
            hyperedge_weight (torch.Tensor, optional): Hyperedge weights
                :math:`\mathbf{W} \in \mathbb{R}^M`. (default: :obj:`None`)
            hyperedge_attr (torch.Tensor, optional): Hyperedge feature matrix
                in :math:`\mathbb{R}^{M \times F}`.
                These features only need to get passed in case
                :obj:`use_attention=True`. (default: :obj:`None`)
            num_edges (int, optional) : The number of edges :math:`M`.
                (default: :obj:`None`)
        """
        num_nodes = x.size(0)

        if num_edges is None:
            num_edges = 0
            if hyperedge_index.numel() > 0:
                num_edges = int(hyperedge_index[1].max()) + 1

        if hyperedge_weight is None:
            hyperedge_weight = x.new_ones(num_edges)

        x = self.lin(x)

        alpha = None
        if self.use_attention:
            assert hyperedge_attr is not None
            x = x.view(-1, self.heads, self.out_channels)
            hyperedge_attr = self.lin(hyperedge_attr)
            hyperedge_attr = hyperedge_attr.view(-1, self.heads,
                                                 self.out_channels)
            x_i = x[hyperedge_index[0]]
            x_j = hyperedge_attr[hyperedge_index[1]]
            alpha = (torch.cat([x_i, x_j], dim=-1) * self.att).sum(dim=-1)
            alpha = F.leaky_relu(alpha, self.negative_slope)
            if self.attention_mode == 'node':
                alpha = softmax(alpha, hyperedge_index[1], num_nodes=hyperedge_attr.size(0))
            else:
                alpha = softmax(alpha, hyperedge_index[0], num_nodes=x.size(0))
            alpha = F.dropout(alpha, p=self.dropout, training=self.training)

        D = scatter(hyperedge_weight[hyperedge_index[1]], hyperedge_index[0],
                    dim=0, dim_size=num_nodes, reduce='sum')
        D = 1.0 / D
        D[D == float("inf")] = 0

        B = scatter(x.new_ones(hyperedge_index.size(1)), hyperedge_index[1],
                    dim=0, dim_size=num_edges, reduce='sum')
        B = 1.0 / B
        B[B == float("inf")] = 0

        out = self.propagate(hyperedge_index, x=x, norm=B, alpha=alpha,
                             size=(num_nodes, num_edges))
        out = self.propagate(hyperedge_index.flip([0]), x=out, norm=D,
                             alpha=alpha, size=(num_edges, num_nodes))

        if self.concat is True:
            out = out.view(-1, self.heads * self.out_channels)
        else:
            out = out.mean(dim=1)

        if self.bias is not None:
            out = out + self.bias

        return out

    def message(self, x_j: Tensor, norm_i: Tensor, alpha: Tensor) -> Tensor:
        H, F = self.heads, self.out_channels

        out = norm_i.view(-1, 1, 1) * x_j.view(-1, H, F)

        if alpha is not None:
            out = alpha.view(-1, self.heads, 1) * out

        return out