import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as pyg_nn

from torch_geometric.graphgym.models.layer import LayerConfig
from torch_geometric.graphgym.register import register_layer
from torch_geometric.nn import Linear as Linear_pyg, GraphSizeNorm
from torch_scatter import scatter_add


def propagate(x, edge_index):
    row, col = edge_index
    out = scatter_add(x[col], row, dim=0)
    return out


def get_mask(x, edge_index, hops):
    for k in range(hops):
        x = propagate(x, edge_index)
    mask = (x>0).float()
    return mask


@register_layer('erdosginconv')
class ErdosGINConvGraphGymLayer(nn.Module):
    """Graph Isomorphism Network with Edge features (GINE) layer double hidden.

    The doubled hidden dimension in MLP follows the
    `"Strategies for Pre-training Graph Neural Networks"
    <https://arxiv.org/abs/1905.12265>`_ paper
    """
    def __init__(self, layer_config: LayerConfig, **kwargs):
        super().__init__()
        gin_nn = nn.Sequential(
            Linear_pyg(layer_config.dim_in, 2 * layer_config.dim_out), nn.ReLU(),
            Linear_pyg(2 * layer_config.dim_out, layer_config.dim_out), nn.ReLU(),
            nn.BatchNorm1d(layer_config.dim_out, eps=layer_config.bn_eps, momentum=layer_config.bn_mom))
        self.model = pyg_nn.GINConv(gin_nn)
        self.gnorm = GraphSizeNorm()

    def forward(self, batch):
        try:
            batch.mask = get_mask(batch.mask, batch.edge_index, 1).to(batch.x.dtype)
        except:
            batch.mask = get_mask(batch.x, batch.edge_index, 1).to(batch.x.dtype)

        batch.x = self.model(batch.x, batch.edge_index, batch.edge_attr)

        batch.x = batch.x * batch.mask
        batch.x = self.gnorm(batch.x)
        return batch