import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import GCNConv

from modules.architecture.layers import ACTIVATION_DICT


class PygGCN(nn.Module):
    def __init__(
            self,
            input_dim: int,
            output_dim: int,
            hidden_dim: int = 32,
            num_layers: int = 2,
            last_activation = 'relu',
    ):
        super().__init__()

        hidden_dim_list = [hidden_dim] * (num_layers - 1)

        self.layers = nn.ModuleList()
        temp_dim = input_dim
        for this_hidden_dim in hidden_dim_list:
            self.layers.append(GCNConv(temp_dim, this_hidden_dim))
            temp_dim = this_hidden_dim 
            
        self.last_layer = GCNConv(temp_dim, output_dim)
        self.last_activation = ACTIVATION_DICT[last_activation]

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        for layer in self.layers:
            x = layer(x, edge_index)
            x = F.relu(x)

        x = self.last_layer(x, edge_index)
        
        if self.last_activation is not None:
            x = self.last_activation(x)

        data.x = x

        return data