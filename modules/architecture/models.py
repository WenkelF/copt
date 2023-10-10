from typing import Dict, Any, List

import torch
import torch.nn as nn
import torch.nn.functional as F

from modules.architecture.layers import GNNLayer, GATLayer, HYBLayer, HYBLayer_pre, RWLayer

from modules.architecture.layers import ACTIVATION_DICT


class GCN(nn.Module):
    def __init__(
            self,
            input_dim: int,
            output_dim: int,
            hidden_dim: int = 16,
            num_layers: int = 2,
            bias: bool = True,
            dropout: float = 0.,
            activation = 'relu',
            last_activation = 'none',
    ):
        super().__init__()

        hidden_dim_list = [hidden_dim] * (num_layers - 1)

        self.gnn_layers = nn.ModuleList()
        temp_dim = input_dim
        for this_hidden_dim in hidden_dim_list:
            self.gnn_layers.append(GNNLayer(temp_dim, this_hidden_dim, bias, dropout, activation))
            temp_dim = this_hidden_dim

        self.gnn_layers.append(GNNLayer(temp_dim, output_dim, bias, dropout, activation=last_activation))

    def forward(self, data):

        x = data['x']
        gcn_mat = data['gcn']

        for layer in self.gnn_layers:
            x = layer(x, gcn_mat)

        data['x'] = x

        return data


class GAT(nn.Module):
    def __init__(
            self,
            input_dim: int,
            output_dim: int,
            hidden_dim: int = 8,
            num_heads_list: list = [8, 1],
            bias: bool = True,
            dropout: float = 0.6,
            activation_att = nn.LeakyReLU(negative_slope=0.2),
            activation = nn.ReLU(),
            activation_last = None,
            skip: bool = False,
            self_loops: bool = True,
            analysis_mode: bool = False,
    ):
        super().__init__()

        self.self_loops = self_loops
        self.analysis_mode = analysis_mode

        self.layers = nn.ModuleList()
        temp_dim = input_dim
        for num_heads in num_heads_list[:-1]:
            self.layers.append(GATLayer(temp_dim, hidden_dim, num_heads, bias, dropout, activation_att, activation, \
                                                                aggregation='cat', skip=skip, analysis_mode=analysis_mode))
            temp_dim = num_heads * hidden_dim
            
        self.layers.append(GATLayer(temp_dim, output_dim, num_heads_list[-1], bias, dropout, activation_att, activation=activation_last, \
                                                                aggregation='mean', skip=False, analysis_mode=analysis_mode))

    def forward(self, data):

        x = data['x']
        adj = data['adj']
        if self.self_loops:
            adj = adj + torch.eye(adj.size(-1))

        # att_mat_list = []
        for layer in self.layers:
            # if self.analysis_mode:
            #     x, att_mat = layer(x, adj)
            #     att_mat_list.append(att_mat)
            # else:
                x = layer(x, adj)

        # if self.analysis_mode:
        #     return x, att_mat_list
        # else:
        #     return x

        data['x'] = x

        return data


class ScGCN(nn.Module):
    def __init__(
            self,
            input_dim: int,
            output_dim: int,
            hidden_dim: int = 8,
            num_layers: int = 1,
            config: list = [-1, -2, -3, 1, 2, 3],
            bias: bool = True,
            dropout: float = 0.,
            activation = 'relu',
            last_activation = 'none',
    ):
        super().__init__()

        self.hyb_layers = nn.ModuleList()
        temp_dim = input_dim
        for _ in range(num_layers):
            self.hyb_layers.append(HYBLayer(temp_dim, hidden_dim, config, bias, dropout, activation))
            temp_dim = hidden_dim * len(config)

        self.res_layer = GNNLayer(temp_dim, output_dim, bias, dropout, activation=last_activation)

    def forward(self, data):

        x = data['x']
        gcn_mat = data['gcn']
        sct_mat = data['sct']

        for hyb_layer in self.hyb_layers:
            x = hyb_layer(x, gcn_mat, sct_mat)

        x = self.res_layer(x, gcn_mat)

        data['x'] = x

        return data


class ScGCN_pre(nn.Module):
    def __init__(
            self,
            input_dim: int,
            output_dim: int,
            hidden_dim: int = 8,
            num_layers: int = 1,
            config: list = [-1, -2, -3, 1, 2, 3],
            bias: bool = True,
            dropout: float = 0.,
            activation = nn.ReLU()
    ):
        super().__init__()

        num_channels = len(config)

        self.hyb_layers = nn.ModuleList()
        temp_dim = input_dim
        for _ in range(num_layers):
            self.hyb_layers.append(HYBLayer_pre(temp_dim, hidden_dim, num_channels, bias, dropout, activation))
            temp_dim = hidden_dim * num_channels

        self.res_layer = GNNLayer(temp_dim, output_dim, bias, dropout, activation=None)

    def forward(self, data):

        x = data['x']

        for hyb_layer in self.hyb_layers:
            x = hyb_layer(x, data.mat_list)

        x = self.res_layer(x, data.res_mat)

        data['x'] = x

        return data


class ScGCN_rwg(nn.Module):
    def __init__(
            self,
            input_dim: int,
            output_dim: int,
            hidden_dim: int = 8,
            num_heads: int = 8,
            num_layers: int = 1,
            config: list = [-1, -2, -3, 1, 2, 3],
            bias: bool = True,
            dropout: float = 0.,
            activation = nn.ReLU(),
            self_loops: bool = True,
            activation_att = nn.LeakyReLU(negative_slope=0.2)
    ):
        super().__init__()

        self.num_layers = num_layers
        self.self_loops = self_loops

        self.rwg_layers = nn.ModuleList()
        self.hyb_layers = nn.ModuleList()
        temp_dim = input_dim
        for _ in range(num_layers):
            self.rwg_layers.append(RWLayer(temp_dim, hidden_dim, num_heads, dropout, activation=activation_att))
            self.hyb_layers.append(HYBLayer(temp_dim, hidden_dim, config, bias, dropout, activation))
            temp_dim = hidden_dim * len(config)

        self.res_layer = GNNLayer(temp_dim, output_dim, bias, dropout, activation=None)

    def forward(self, data):

        x = data.x
        adj = data.adj
        if self.self_loops:
            adj = adj + torch.eye(adj.size(-1))

        for i in range(self.num_layers):
            sct_mat = self.rwg_layers[i](x, adj)
            x = self.hyb_layers[i](x, data.gcn_mat, sct_mat)

        x = self.res_layer(x, data.res_mat)

        return F.softmax(x, dim=-1)
    

class MLP(nn.Module):
    def __init__(
            self,
            input_dim: int,
            output_dim: int,
            hidden_dim: int = 16,
            num_layers: int = 2,
            bias: bool = True,
            dropout: float = 0.,
            activation = 'relu',
            last_activation = 'none',
    ):
        super().__init__()

        hidden_dim_list = [hidden_dim] * (num_layers - 1)

        self.layers = nn.ModuleList()
        temp_dim = input_dim
        for hidden_dim in hidden_dim_list:
            self.layers.append(nn.Linear(temp_dim, hidden_dim, bias=bias))
            temp_dim = hidden_dim

        self.layers.append(nn.Linear(temp_dim, output_dim, bias=bias))

        self.dropout = nn.Dropout(p=dropout)
        self.activation = ACTIVATION_DICT[activation]
        self.last_activation = ACTIVATION_DICT[last_activation]

    def forward(self, x):

        x = self.dropout(x)

        for layer in self.layers[:-1]:
            x = layer(x)
            if self.activation is not None:
                x = self.activation(x)
            x = self.dropout(x)

        x = self.layers[-1](x)
        if self.last_activation is not None:
            x = self.last_activation(x)

        return x