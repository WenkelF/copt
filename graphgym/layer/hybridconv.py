from typing import List, Literal, Tuple, Optional
from collections import OrderedDict

from torch import Tensor

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.nn.models import MLP
from torch_geometric.typing import SparseTensor
from torch_geometric.utils import spmm, scatter, remove_self_loops, add_remaining_self_loops
from torch_geometric.nn.inits import glorot
from torch_geometric.utils.num_nodes import maybe_num_nodes

from torch_geometric.graphgym.models.layer import LayerConfig
from torch_geometric.graphgym.register import register_layer
from torch_geometric.graphgym.config import cfg

import torch_geometric
from torch_geometric.data import Data
from torch_geometric.data.datapipes import functional_transform
from torch_geometric.transforms import BaseTransform

from modules.architecture.layers import ACTIVATION_DICT


class HybridConv(MessagePassing):
    r"""The hybrid scattering operator from the `"Scattering GCN" <https://arxiv.org/abs/2003.08414>`_ paper

    .. math::
        \mathbf{X}^{\prime} = {\left(\mathbf{\hat{D}}^{-1/2} \mathbf{\hat{A}}
        \mathbf{\hat{D}}^{-1/2} \right)}^K \mathbf{X} \mathbf{\Theta},

    where :math:`\mathbf{\hat{A}} = \mathbf{A} + \mathbf{I}` denotes the
    adjacency matrix with inserted self-loops and
    :math:`\hat{D}_{ii} = \sum_{j=0} \hat{A}_{ij}` its diagonal degree matrix.
    The adjacency matrix can include other values than :obj:`1` representing
    edge weights via the optional :obj:`edge_weight` tensor.

    Args:
        in_channels (int): Size of each input sample, or :obj:`-1` to derive
            the size from the first input(s) to the forward method.
        out_channels (int): Size of each output sample.
        agg_list (list, optional): Number of hops :math:`K`. (default: :obj:`1`)
        cached (bool, optional): If set to :obj:`True`, the layer will cache
            the computation of :math:`{\left(\mathbf{\hat{D}}^{-1/2}
            \mathbf{\hat{A}} \mathbf{\hat{D}}^{-1/2} \right)}^K \mathbf{X}` on
            first execution, and will use the cached version for further
            executions.
            This parameter should only be set to :obj:`True` in transductive
            learning scenarios. (default: :obj:`False`)
        add_self_loops (bool, optional): If set to :obj:`False`, will not add
            self-loops to the input graph. (default: :obj:`True`)
        bias (bool, optional): If set to :obj:`False`, the layer will not learn
            an additive bias. (default: :obj:`True`)
        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.nn.conv.MessagePassing`.

    Shapes:
        - **input:**
          node features :math:`(|\mathcal{V}|, F_{in})`,
          edge indices :math:`(2, |\mathcal{E}|)`,
          edge weights :math:`(|\mathcal{E}|)` *(optional)*
        - **output:**
          node features :math:`(|\mathcal{V}|, F_{out})`
    """

    def __init__(
            self,
            input_dim: int,
            output_dim: int,
            channel_list: List[Tuple[int]] = [[1], [2], [4], [0, 1], [1, 2], [2, 4]],
            combine_fn: Literal['cat', 'att', 'att_bias'] = "cat",
            residual: bool = True,
            activation_att1: str = 'relu',
            activation_att2: str = 'relu',
            activation: str = 'leaky_relu',
            depth_mlp: int = 1,
            num_heads: int = 1,
            add_self_loops: bool = True,
            norm: str = 'gcn',
            filter_norm_dim: int = 1,
            bias: bool = True,
            **kwargs
    ):
        kwargs.setdefault('aggr', 'add')
        super().__init__(**kwargs)

        num_channels = len(channel_list)
        if residual:
            num_channels += 1

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.channel_list = channel_list
        self.residual = residual
        self.radius_list = list(set([agg for channel in channel_list for agg in channel]))
        self.radius_list.sort()
        self.combine_fn = combine_fn
        self.num_heads = num_heads
        self.add_self_loops = add_self_loops
        self.norm = norm
        self.filter_norm_dim = filter_norm_dim
        self.activation_att1 = ACTIVATION_DICT[activation_att1]
        self.activation_att2 = ACTIVATION_DICT[activation_att2]
        self.activation = ACTIVATION_DICT[activation]

        if self.combine_fn == 'cat':
            self.lin_combine = Linear(num_channels * output_dim, output_dim, bias=bias)
        elif self.combine_fn in ['att', 'att_bias']:
            self.lin_att_list = nn.ModuleList()
            for _ in range(len(channel_list) + 1):
                self.lin_att_list.append(Linear(input_dim, input_dim, bias=bias))
            self.att_pre = nn.Parameter(torch.empty(input_dim,num_heads))
            self.att_channel = nn.Parameter(torch.empty(input_dim, num_heads))
            self.mlp_out = MLP([input_dim] + (depth_mlp - 1) * [input_dim] + [output_dim], bias=bias, activation=activation, norm=None)
        else:
            raise ValueError()

        self.reset_parameters()

    def reset_parameters(self):
        super().reset_parameters()
        if self.combine_fn == 'cat':
            self.lin_combine.reset_parameters()
        elif self.combine_fn in ['att', 'att_bias']:
            for idx in range(len(self.lin_att_list)):
                self.lin_att_list[idx].reset_parameters()
            glorot(self.att_pre)
            glorot(self.att_channel)
            self.mlp_out.reset_parameters()

    def forward(self, x, edge_index) -> Tensor:
        edge_weight = None
        
        if self.norm == 'gcn':
            edge_index, edge_weight = gcn_norm(  # yapf: disable
                edge_index, edge_weight, x.size(self.node_dim), False,
                self.add_self_loops, self.flow, dtype=x.dtype)
        elif self.norm == 'rw':
            edge_index, edge_weight = rw_norm(  # yapf: disable
                edge_index, edge_weight, x.size(self.node_dim), False,
                self.add_self_loops, self.flow, dtype=x.dtype)
        elif self.norm == 'sym':
            edge_index, edge_weight = sym_norm(  # yapf: disable
                edge_index, edge_weight, x.size(self.node_dim), False,
                self.add_self_loops, self.flow, dtype=x.dtype)
        elif self.norm == 'avg':
            edge_index, edge_weight = avg_norm(  # yapf: disable
                edge_index, edge_weight, x.size(self.node_dim), False,
                self.add_self_loops, self.flow, dtype=x.dtype)
        else:
            raise NotImplementedError('norm type not supported')

        x_channel_list = [x]

        r_tmp = 0
        x_agg_dict = OrderedDict()
        x_agg_dict.update({0: x})
        for this_r in self.radius_list:
            x = list(x_agg_dict.values())[-1]
            for _ in range(this_r - r_tmp):
                # propagate_type: (x: Tensor, edge_weight: OptTensor)
                x = self.propagate(edge_index, x=x, edge_weight=edge_weight, size=None)
            x_agg_dict[this_r] = x
            r_tmp = this_r

        for channel in self.channel_list:
            if len(channel) == 1:
                x_channel_list.append(self.normalize_filter(x_agg_dict[channel[0]], dim=self.filter_norm_dim))
            else:
                x_channel_list.append(self.normalize_filter(x_agg_dict[channel[0]] - x_agg_dict[channel[1]], dim=self.filter_norm_dim))

        if self.combine_fn == 'cat':
            if not self.residual:
                x_channel_list = x_channel_list[1:]
            x = torch.cat(x_channel_list, dim=-1)
            x = self.lin_combine(x)
            x = self.activation(x)
        elif self.combine_fn == 'att':
            x = self.channel_attention(x_channel_list)
        elif self.combine_fn == 'att_bias':
            x = self.mlp_out(x + self.channel_attention(x_channel_list))
        else:
            raise ValueError()

        return x

    def message(self, x_j: Tensor, edge_weight: Tensor) -> Tensor:
        return edge_weight.view(-1, 1) * x_j

    def message_and_aggregate(self, adj_t: SparseTensor, x: Tensor) -> Tensor:
        return spmm(adj_t, x, reduce=self.aggr)
    
    def channel_attention(self, x_channel_list: Tensor) -> Tensor:
        for idx, h in enumerate(x_channel_list):
            x_channel_list[idx] = self.activation_att1(self.lin_att_list[idx](h))
        e_pre = torch.matmul(self.activation_att1(x_channel_list[0]), self.att_pre)
        e_channel_list = [self.activation_att2(torch.matmul(x_channel, self.att_channel)) for x_channel in x_channel_list]
        e = torch.stack(e_channel_list, dim=0) + e_pre
        channel_weights = torch.softmax(e, dim=0)

        weighted_channels = torch.mul(channel_weights.unsqueeze(-2), torch.stack(x_channel_list, dim=0).unsqueeze(-1))
        out = weighted_channels.sum(dim=0).mean(dim=-1)

        return out

    def normalize_filter(self, x: Tensor, dim: int = 1, eps: float = 1e-5) -> Tensor:
        if dim in [0, 1]:
            mean, var = x.mean(dim), x.var(dim)
            x = (x - mean.unsqueeze(dim)) / torch.sqrt(var + eps).unsqueeze(dim)
        
        return x

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.input_dim}, '
                f'{self.output_dim}, channel_list={self.channel_list})')


@register_layer('hybridconv')
class HybridConvLayer(nn.Module):
    """HybridConv layer"""

    def __init__(self, layer_config: LayerConfig, **kwargs):
        super().__init__()
        self.model = HybridConv(
            layer_config.dim_in,
            layer_config.dim_out,
            channel_list=cfg.gnn.hybrid.channel_list,
            combine_fn=cfg.gnn.hybrid.combine_fn,
            residual=cfg.gnn.hybrid.residual,
            activation_att1=cfg.gnn.hybrid.activation_att1,
            activation_att2=cfg.gnn.hybrid.activation_att2,
            activation=cfg.gnn.hybrid.activation,
            num_heads=cfg.gnn.hybrid.num_heads,
            add_self_loops=cfg.gnn.hybrid.add_self_loops,
            filter_norm_dim=cfg.gnn.hybrid.filter_norm_dim,
            bias=layer_config.has_bias,
            **kwargs
        )

    def forward(self, batch):
        batch.x = self.model(batch.x, batch.edge_index)
        return batch


def rw_norm(edge_index, edge_weight=None, num_nodes=None, improved=False,
             add_self_loops=True, flow="source_to_target", dtype=None):

    fill_value = 2. if improved else 1.

    assert flow in ['source_to_target', 'target_to_source']
    num_nodes = maybe_num_nodes(edge_index, num_nodes)
    
    edge_index_wo_self_loops, _ = remove_self_loops(edge_index)

    if add_self_loops:
        edge_index_w_self_loops, edge_weight = add_remaining_self_loops(
            edge_index_wo_self_loops, edge_weight, fill_value, num_nodes)

    if edge_weight is None:
        edge_weight_w_self_loops = torch.ones((edge_index_w_self_loops.size(1), ), dtype=dtype,
                                 device=edge_index_w_self_loops.device)
        edge_weight_wo_self_loops = torch.ones((edge_index_wo_self_loops.size(1), ), dtype=dtype,
                                 device=edge_index_wo_self_loops.device)

    deg = scatter(edge_weight_wo_self_loops, edge_index_wo_self_loops[1], dim=0, dim_size=num_nodes, reduce='sum')
    deg_inv = deg.pow_(-1.)
    deg_inv.masked_fill_(deg_inv == float('inf'), 0)
    edge_weight = edge_weight_w_self_loops * deg_inv[edge_index_w_self_loops[1]]

    self_loop_idx = (edge_index_w_self_loops[0] == edge_index_w_self_loops[1])
    edge_weight[self_loop_idx] = 1.

    return edge_index_w_self_loops, edge_weight / 2


def sym_norm(edge_index, edge_weight=None, num_nodes=None, improved=False,
             add_self_loops=True, flow="source_to_target", dtype=None):

    fill_value = 2. if improved else 1.

    assert flow in ['source_to_target', 'target_to_source']
    num_nodes = maybe_num_nodes(edge_index, num_nodes)
    
    edge_index_wo_self_loops, _ = remove_self_loops(edge_index)

    if add_self_loops:
        edge_index_w_self_loops, edge_weight = add_remaining_self_loops(
            edge_index_wo_self_loops, edge_weight, fill_value, num_nodes)

    if edge_weight is None:
        edge_weight_w_self_loops = torch.ones((edge_index_w_self_loops.size(1), ), dtype=dtype,
                                 device=edge_index_w_self_loops.device)
        edge_weight_wo_self_loops = torch.ones((edge_index_wo_self_loops.size(1), ), dtype=dtype,
                                 device=edge_index_wo_self_loops.device)

    deg = scatter(edge_weight_wo_self_loops, edge_index_wo_self_loops[1], dim=0, dim_size=num_nodes, reduce='sum')
    deg_inv_sqr = deg.pow_(-0.5)
    deg_inv_sqr.masked_fill_(deg_inv_sqr == float('inf'), 0)
    edge_weight = deg_inv_sqr[edge_index_w_self_loops[0]] * edge_weight_w_self_loops * deg_inv_sqr[edge_index_w_self_loops[1]]

    self_loop_idx = (edge_index_w_self_loops[0] == edge_index_w_self_loops[1])
    edge_weight[self_loop_idx] = 1.

    return edge_index_w_self_loops, edge_weight / 2


def avg_norm(edge_index, edge_weight=None, num_nodes=None, improved=False,
             add_self_loops=True, flow="source_to_target", dtype=None):

    fill_value = 2. if improved else 1.

    assert flow in ['source_to_target', 'target_to_source']
    num_nodes = maybe_num_nodes(edge_index, num_nodes)
    
    edge_index_wo_self_loops, _ = remove_self_loops(edge_index)

    if add_self_loops:
        edge_index_w_self_loops, edge_weight = add_remaining_self_loops(
            edge_index_wo_self_loops, edge_weight, fill_value, num_nodes)

    if edge_weight is None:
        edge_weight_w_self_loops = torch.ones((edge_index_w_self_loops.size(1), ), dtype=dtype,
                                 device=edge_index_w_self_loops.device)
        edge_weight_wo_self_loops = torch.ones((edge_index_wo_self_loops.size(1), ), dtype=dtype,
                                 device=edge_index_wo_self_loops.device)

    deg = scatter(edge_weight_wo_self_loops, edge_index_wo_self_loops[1], dim=0, dim_size=num_nodes, reduce='sum')
    deg_inv = deg.pow_(-1.)
    deg_inv.masked_fill_(deg_inv == float('inf'), 0)
    edge_weight = edge_weight_w_self_loops * deg_inv[edge_index_w_self_loops[0]]

    self_loop_idx = (edge_index_w_self_loops[0] == edge_index_w_self_loops[1])
    edge_weight[self_loop_idx] = 1.

    return edge_index_w_self_loops, edge_weight / 2