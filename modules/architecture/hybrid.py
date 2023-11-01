from typing import List, Tuple, Optional
from collections import OrderedDict

from torch import Tensor

import torch
import torch.nn as nn

from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.typing import Adj, OptTensor, SparseTensor
from torch_geometric.utils import spmm
from torch_geometric.nn.inits import glorot

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
            # activation: str = 'relu',
            activation_channel: str = 'abs',
            # num_heads: int = None,
            # cached: bool = False,
            add_self_loops: bool = True,
            bias: bool = True,
            **kwargs
    ):
        kwargs.setdefault('aggr', 'add')
        super().__init__(**kwargs)

        num_channels = len(channel_list)

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.channel_list = channel_list
        self.radius_list = list(set([agg for channel in channel_list for agg in channel]))
        self.radius_list.sort()
        self.combine_fn = combine_fn
        self.add_self_loops = add_self_loops

        self.lin = Linear(input_dim, output_dim, bias=bias)
        # self.activation = ACTIVATION_DICT[activation]
        self.activation_channel = ACTIVATION_DICT[activation_channel]
        # self.lin_channel = Linear(input_dim, output_dim, bias=bias)

        if self.combine_fn == 'cat':
            self.lin_combine = Linear(num_channels * output_dim, output_dim, bias=bias)
        elif self.combine_fn == 'att':
            self.att_pre = nn.Parameter(torch.empty(1, output_dim))
            self.att_channel = nn.Parameter(torch.empty(1, output_dim))
            self.activatetion_att = ACTIVATION_DICT['elu']
        else:
            raise ValueError()

        self.reset_parameters()

    def reset_parameters(self):
        super().reset_parameters()
        self.lin.reset_parameters()
        if self.combine_fn == 'cat':
            self.lin_combine.reset_parameters()
        elif self.combine_fn == 'att':
            glorot(self.att_pre)
            glorot(self.att_channel)


    def forward(self, data) -> Tensor:

        x, edge_index = data.x, data.edge_index
        edge_weight = None

        if isinstance(edge_index, Tensor):
            edge_index, edge_weight = gcn_norm(  # yapf: disable
                edge_index, edge_weight, x.size(self.node_dim), False,
                self.add_self_loops, self.flow, dtype=x.dtype)
        elif isinstance(edge_index, SparseTensor):
            edge_index = gcn_norm(  # yapf: disable
                edge_index, edge_weight, x.size(self.node_dim), False,
                self.add_self_loops, self.flow, dtype=x.dtype)

        x = self.lin(x)
        
        r = 0
        x_agg_dict = OrderedDict()
        x_agg_dict[0] = x   
        for this_r in self.radius_list:
            x = list(x_agg_dict.values())[-1]
            for _ in range(this_r - r):
                # propagate_type: (x: Tensor, edge_weight: OptTensor)
                x = self.propagate(edge_index, x=x, edge_weight=edge_weight, size=None)
            x_agg_dict[this_r] = x

        x_channel_list = []
        for channel in self.channel_list:
            if len(channel) == 1:
                x_channel_list.append(self.activation_channel(x_agg_dict[channel[0]]))
            else:
                x_channel_list.append(self.activation_channel(x_agg_dict[channel[0]] - x_agg_dict[channel[1]]))

        if self.combine_fn == 'cat':
            x = torch.cat(x_channel_list, dim=-1)
            x = self.lin_combine(x)
            # x = self.activation(x)
        elif self.combine_fn == 'att':
            raise NotImplementedError()
        else:
            raise ValueError()

        data.x = x
        
        return data


    def message(self, x_j: Tensor, edge_weight: Tensor) -> Tensor:
        return edge_weight.view(-1, 1) * x_j

    def message_and_aggregate(self, adj_t: SparseTensor, x: Tensor) -> Tensor:
        return spmm(adj_t, x, reduce=self.aggr)

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.input_dim}, '
                f'{self.output_dim}, channel_list={self.channel_list})')
    

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