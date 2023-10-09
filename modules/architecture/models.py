from typing import Dict, Any, List

import torch
import torch.nn as nn
import torch.nn.functional as F

from modules.architecture.layers import GNNLayer, GATLayer, HYBLayer, HYBLayer_pre, RWLayer


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
            device: str = 'cpu'
    ):
        super().__init__()

        self.device = device
        hidden_dim_list = [hidden_dim] * (num_layers - 1) + [output_dim]

        self.gnn_layers = nn.ModuleList()
        temp_dim = input_dim
        for hidden_dim in hidden_dim_list:
            self.gnn_layers.append(GNNLayer(temp_dim, hidden_dim, bias, dropout, activation))
            temp_dim = hidden_dim

    def forward(self, data):

        x = data['x'].to(self.device)
        gcn_mat = data['gcn'].to(self.device)

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
            device: str = "cpu",
    ):
        super().__init__()

        self.self_loops = self_loops
        self.analysis_mode = analysis_mode
        self.device = device

        self.layers = nn.ModuleList()
        temp_dim = input_dim
        for num_heads in num_heads_list[:-1]:
            self.layers.append(GATLayer(temp_dim, hidden_dim, num_heads, bias, dropout, activation_att, activation, \
                                                                aggregation='cat', skip=skip, analysis_mode=analysis_mode))
            temp_dim = num_heads * hidden_dim
            
        self.layers.append(GATLayer(temp_dim, output_dim, num_heads_list[-1], bias, dropout, activation_att, activation=activation_last, \
                                                                aggregation='mean', skip=False, analysis_mode=analysis_mode))

    def forward(self, data):

        x = data['x'].to(self.device)
        adj = data['adj'].to(self.device)
        if self.self_loops:
            adj = adj + torch.eye(adj.size(-1)).to(adj.device)

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
            activation = nn.ReLU(),
            device: str = "cpu"
    ):
        super().__init__()

        self.device = device

        self.hyb_layers = nn.ModuleList()
        temp_dim = input_dim
        for _ in range(num_layers):
            self.hyb_layers.append(HYBLayer(temp_dim, hidden_dim, config, bias, dropout, activation))
            temp_dim = hidden_dim * len(config)

        self.res_layer = GNNLayer(temp_dim, output_dim, bias, dropout, activation=None)

    def forward(self, data):

        x = data['x'].to(self.device)
        gcn_mat = data['gcn_mat'].to(self.device)
        sct_mat = data['sct_mat'].to(self.device)

        for hyb_layer in self.hyb_layers:
            x = hyb_layer(x, gcn_mat, sct_mat)

        x = self.res_layer(x, data.res_mat)

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

        x = data['x'].to(self.device)

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
            adj = adj + torch.eye(adj.size(-1)).to(adj.device)

        for i in range(self.num_layers):
            sct_mat = self.rwg_layers[i](x, adj)
            x = self.hyb_layers[i](x, data.gcn_mat, sct_mat)

        x = self.res_layer(x, data.res_mat)

        return F.softmax(x, dim=-1)