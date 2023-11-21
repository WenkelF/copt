from typing import List, Tuple, Optional
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
from torch_geometric.utils import spmm
from torch_geometric.nn.inits import glorot

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
            combine_fn: str = 'cat',
            residual: bool = False,
            activation_channel: str = 'relu',
            num_heads: int = 1,
            add_self_loops: bool = True,
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
        self.activation_channel = ACTIVATION_DICT[activation_channel]

        if self.combine_fn == 'cat':
            self.lin_combine = Linear(num_channels * output_dim, output_dim, bias=bias)
        elif self.combine_fn == 'att':
            self.lin_att_list = nn.ModuleList()
            for _ in range(len(self.channel_list) + 1):
                self.lin_att_list.append(Linear(input_dim, output_dim, bias=bias))
            self.att_pre = nn.Parameter(torch.empty(output_dim, self.num_heads))
            self.att_channel = nn.Parameter(torch.empty(output_dim, self.num_heads))
            self.activatetion_att = ACTIVATION_DICT['elu']
            # self.mlp_out = MLP([input_dim] + 1 * [input_dim] + [output_dim], bias=bias, activation='relu', norm=None)
        else:
            raise ValueError()

        self.reset_parameters()

    def reset_parameters(self):
        super().reset_parameters()
        if self.combine_fn == 'cat':
            self.lin_combine.reset_parameters()
        elif self.combine_fn == 'att':
            for idx in range(len(self.lin_att_list)):
                self.lin_att_list[idx].reset_parameters()
            glorot(self.att_pre)
            glorot(self.att_channel)
            # self.mlp_out.reset_parameters()

    def forward(self, x, edge_index) -> Tensor:
        edge_weight = None

        if isinstance(edge_index, Tensor):
            edge_index, edge_weight = gcn_norm(  # yapf: disable
                edge_index, edge_weight, x.size(self.node_dim), False,
                self.add_self_loops, self.flow, dtype=x.dtype)
        elif isinstance(edge_index, SparseTensor):
            edge_index = gcn_norm(  # yapf: disable
                edge_index, edge_weight, x.size(self.node_dim), False,
                self.add_self_loops, self.flow, dtype=x.dtype)

        x_channel_list = [x]

        r = 0
        x_agg_dict = OrderedDict()
        x_agg_dict[0] = x
        for this_r in self.radius_list:
            x = list(x_agg_dict.values())[-1]
            for _ in range(this_r - r):
                # propagate_type: (x: Tensor, edge_weight: OptTensor)
                x = self.propagate(edge_index, x=x, edge_weight=edge_weight, size=None)
            x_agg_dict[this_r] = x

        for channel in self.channel_list:
            if len(channel) == 1:
                x_channel_list.append(x_agg_dict[channel[0]])
            else:
                x_channel_list.append(x_agg_dict[channel[0]] - x_agg_dict[channel[1]])

        if self.combine_fn == 'cat':
            if not self.residual:
                x_channel_list = x_channel_list[1:]
            x = torch.cat(x_channel_list, dim=-1)
            x = self.lin_combine(x)
            x = self.activation_channel(x)
        elif self.combine_fn == 'att':
            x = self.channel_attention(x_channel_list)
        else:
            raise ValueError()

        return x

    def message(self, x_j: Tensor, edge_weight: Tensor) -> Tensor:
        return edge_weight.view(-1, 1) * x_j

    def message_and_aggregate(self, adj_t: SparseTensor, x: Tensor) -> Tensor:
        return spmm(adj_t, x, reduce=self.aggr)
    
    def channel_attention(self, x_channel_list: Tensor) -> Tensor:
        for idx, h in enumerate(x_channel_list):
            x_channel_list[idx] = self.activation_channel(self.lin_att_list[idx](h))
        e_pre = torch.matmul(self.activation_channel(x_channel_list[0]), self.att_pre)
        e_channel_list = [torch.matmul(x_channel, self.att_channel) for x_channel in x_channel_list[1:]]
        e = torch.stack(e_channel_list, dim=0) + e_pre
        channel_weights = torch.softmax(e, dim=0)

        weighted_channels = torch.mul(channel_weights.unsqueeze(-2), torch.stack(x_channel_list[1:], dim=0).unsqueeze(-1))
        x = weighted_channels.sum(dim=0).mean(dim=-1)

        # x = self.mlp_out(x + x_channel_list[0])

        # x = F.leaky_relu(x)

        return x

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.input_dim}, '
                f'{self.output_dim}, channel_list={self.channel_list})')


@register_layer('hybridconv')
class HybridConvLayer(nn.Module):
    """HybridConv layer"""

    def __init__(self, layer_config: LayerConfig, **kwargs):
        super().__init__()
        self.model = HybridConv(layer_config.dim_in,
                                layer_config.dim_out,
                                channel_list=cfg.gnn.hybrid.channel_list,
                                combine_fn=cfg.gnn.hybrid.combine_fn,
                                residual=cfg.gnn.hybrid.residual,
                                activation_channel=cfg.gnn.hybrid.activation_channel,
                                num_heads=cfg.gnn.hybrid.num_heads,
                                add_self_loops=cfg.gnn.hybrid.add_self_loops,
                                bias=layer_config.has_bias,
                                **kwargs)

    def forward(self, batch):
        batch.x = self.model(batch.x, batch.edge_index)
        return batch



@functional_transform('rw_norm')
class RWNorm(BaseTransform):
    r"""Applies the GCN normalization from the `"Semi-supervised Classification
    with Graph Convolutional Networks" <https://arxiv.org/abs/1609.02907>`_
    paper (functional name: :obj:`gcn_norm`).

    .. math::
        \mathbf{\hat{A}} = \mathbf{\hat{D}}^{-1/2} (\mathbf{A} + \mathbf{I})
        \mathbf{\hat{D}}^{-1/2}

    where :math:`\hat{D}_{ii} = \sum_{j=0} \hat{A}_{ij} + 1`.
    """

    def __init__(self, add_self_loops: bool = True):
        self.add_self_loops = add_self_loops

    def forward(self, data: Data) -> Data:
        gcn_norm = torch_geometric.nn.conv.gcn_conv.gcn_norm
        assert 'edge_index' in data or 'adj_t' in data

        if 'edge_index' in data:
            data.edge_index, data.edge_weight = gcn_norm(
                data.edge_index, data.edge_weight, data.num_nodes,
                add_self_loops=self.add_self_loops)
        else:
            data.adj_t = gcn_norm(data.adj_t,
                                  add_self_loops=self.add_self_loops)

        return data

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}('
                f'add_self_loops={self.add_self_loops})')