from typing import List, Tuple, Optional, Dict
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


class HybridConv_v3(MessagePassing):
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
            channel_list: List[Tuple[int]] = [[0], [1], [2], [4], [0, 1], [1, 2], [2, 4]],
            num_heads: int = 1,
            activation_att1: str = 'elu',
            activation_att2: str = None,
            activation: str = 'relu',
            depth_mlp: int = 1,
            skip: bool = False,
            add_self_loops: bool = True,
            bias: bool = True,
            **kwargs
    ):
        kwargs.setdefault('aggr', 'add')
        super().__init__(**kwargs)

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.channel_list = channel_list
        self.radius_list = list(set([agg for channel in channel_list for agg in channel]))
        self.radius_list.sort()
        self.num_heads = num_heads
        self.skip = skip
        self.add_self_loops = add_self_loops
        self.activation_att1 = ACTIVATION_DICT[activation_att1]
        self.activation_att2 = ACTIVATION_DICT[activation_att2]
        self.activation = ACTIVATION_DICT[activation]

        self.head_dict = nn.ModuleDict()
        for channel in channel_list:
            self.head_dict[str(channel)] = nn.ModuleList()
            for _ in range(num_heads):
                self.head_dict[str(channel)].append(nn.Linear(input_dim, output_dim, bias=bias))
        self.att_pre_low = nn.Parameter(torch.empty(num_heads, output_dim, output_dim))
        self.att_pre_band = nn.Parameter(torch.empty(num_heads, output_dim, output_dim))
        self.att_channel_low = nn.Parameter(torch.empty(num_heads, output_dim, output_dim))
        self.att_channel_band = nn.Parameter(torch.empty(num_heads, output_dim, output_dim))
        if depth_mlp > 0:
            m = 2
            if skip:
                m += 1
            self.mlp_out = MLP([m * output_dim] + depth_mlp * [output_dim], bias=bias, activation=activation, norm=None)
        else:
            self.mlp_out = None

        self.reset_parameters()

    def reset_parameters(self):
        super().reset_parameters()
        for channel in self.channel_list:
            for k in range(self.num_heads):
                self.head_dict[str(channel)][k].reset_parameters()
        glorot(self.att_pre_low)
        glorot(self.att_pre_band)
        glorot(self.att_channel_low)
        glorot(self.att_channel_band)
        if self.mlp_out is not None:
            self.mlp_out.reset_parameters()

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

        r_tmp = 0
        x_agg_dict = OrderedDict()
        x_agg_dict[0] = x
        for this_r in self.radius_list:
            if this_r == 0:
                continue
            x = list(x_agg_dict.values())[-1]
            for _ in range(this_r - r_tmp):
                # propagate_type: (x: Tensor, edge_weight: OptTensor)
                x = self.propagate(edge_index, x=x, edge_weight=edge_weight, size=None)
            x_agg_dict[this_r] = x
            r_tmp = this_r

        x_channel_dict = {}

        for channel in self.channel_list:
            if len(channel) == 1:
                x_channel_dict[str(channel)] = x_agg_dict[channel[0]]
            else:
                x_channel_dict[str(channel)] = x_agg_dict[channel[0]] - x_agg_dict[channel[1]]

        x = self.channel_attention(x_channel_dict)

        if self.skip:
            x = torch.cat([x_agg_dict[0], x], dim=-1)

        if self.mlp_out is not None:
            x = self.mlp_out(x)
        else:
            x = self.activation(x)

        return x

    def message(self, x_j: Tensor, edge_weight: Tensor) -> Tensor:
        return edge_weight.view(-1, 1) * x_j

    def message_and_aggregate(self, adj_t: SparseTensor, x: Tensor) -> Tensor:
        return spmm(adj_t, x, reduce=self.aggr)
    
    def channel_attention(self, x_channel_dict: Dict) -> Tensor:
        x_channel_low, x_channel_band = [], []
        for channel, h in x_channel_dict.items():
            channel = eval(channel)
            if len(channel) == 1:
                heads = []
                for k in range(self.num_heads):
                    heads.append(self.activation_att1(self.head_dict[str(channel)][k](h)))
                x_channel_low.append(torch.stack(heads, dim=0))
            else:
                heads = []
                for k in range(self.num_heads):
                    heads.append(self.activation_att1(self.head_dict[str(channel)][k](h)))
                x_channel_band.append(torch.stack(heads, dim=0))
            
        e_pre_low = self.activation_att2(torch.matmul(x_channel_low[0], self.att_pre_low))
        e_pre_band = self.activation_att2(torch.matmul(x_channel_band[0], self.att_pre_band))
        e_channel_list_low = [self.activation_att2(torch.matmul(x_channel, self.att_channel_low)) for x_channel in x_channel_low[1:]]
        e_channel_list_band = [self.activation_att2(torch.matmul(x_channel, self.att_channel_band)) for x_channel in x_channel_band]
        e_low = torch.stack(e_channel_list_low, dim=0) + e_pre_low
        e_band = torch.stack(e_channel_list_band, dim=0) + e_pre_band
        channel_weights_low = torch.softmax(e_low, dim=0)
        channel_weights_band = torch.softmax(e_band, dim=0)

        weighted_channels_low = torch.mul(channel_weights_low, torch.stack(x_channel_low[1:], dim=0))
        weighted_channels_band = torch.mul(channel_weights_band, torch.stack(x_channel_band, dim=0))
        x_low = weighted_channels_low.sum(dim=0).mean(dim=0)
        x_band = weighted_channels_band.sum(dim=0).mean(dim=0)

        x = torch.cat([x_low, x_band], dim=-1)

        return x

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.input_dim}, '
                f'{self.output_dim}, channel_list={self.channel_list})')


@register_layer('hybridconv-v3')
class HybridConvLayer(nn.Module):
    """HybridConv layer"""

    def __init__(self, layer_config: LayerConfig, **kwargs):
        super().__init__()
        self.model = HybridConv_v3(layer_config.dim_in,
                                layer_config.dim_out,
                                channel_list=cfg.gnn.hybrid_v2.channel_list,
                                num_heads=cfg.gnn.hybrid_v2.num_heads,
                                activation_att1=cfg.gnn.hybrid_v2.activation_att1,
                                activation_att2=cfg.gnn.hybrid_v2.activation_att2,
                                activation=cfg.gnn.hybrid_v2.activation,
                                depth_mlp=cfg.gnn.hybrid_v2.depth_mlp,
                                skip=cfg.gnn.hybrid_v2.skip,
                                add_self_loops=cfg.gnn.hybrid.add_self_loops,
                                bias=layer_config.has_bias,
                                **kwargs)

    def forward(self, batch):
        batch.x = self.model(batch.x, batch.edge_index)
        return batch



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