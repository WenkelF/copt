import torch.nn as nn
import torch_geometric.nn as pyg_nn

from torch_geometric.graphgym.config import cfg
from torch_geometric.graphgym.models.layer import LayerConfig
from torch_geometric.graphgym.register import register_layer
from torch_geometric.nn import Linear as Linear_pyg, GraphSizeNorm, Linear
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
        self.gsn = GraphSizeNorm()

    def forward(self, batch):
        try:
            batch.mask = get_mask(batch.mask, batch.edge_index, 1).to(batch.x.dtype)
        except:
            batch.mask = get_mask(batch.x, batch.edge_index, 1).to(batch.x.dtype)

        batch.x = self.model(batch.x, batch.edge_index, batch.edge_attr)

        batch.x = batch.x * batch.mask
        if cfg.gnn.gsn:
            batch.x = self.gsn(batch.x, batch.batch, batch.num_nodes)
        return batch


@register_layer('erdosgatconv')
class ErdosGATConvGraphGymLayer(nn.Module):
    """Graph Isomorphism Network with Edge features (GINE) layer double hidden.

    The doubled hidden dimension in MLP follows the
    `"Strategies for Pre-training Graph Neural Networks"
    <https://arxiv.org/abs/1905.12265>`_ paper
    """
    def __init__(self, layer_config: LayerConfig, **kwargs):
        super().__init__()
        self.model = pyg_nn.GATConv(layer_config.dim_in, layer_config.dim_out, heads=cfg.gnn.att_heads, concat=True)
        self.linear = Linear(cfg.gnn.att_heads * layer_config.dim_out, layer_config.dim_out)
        self.gsn = GraphSizeNorm()

    def forward(self, batch):
        try:
            batch.mask = get_mask(batch.mask, batch.edge_index, 1).to(batch.x.dtype)
        except:
            batch.mask = get_mask(batch.x, batch.edge_index, 1).to(batch.x.dtype)

        batch.x = self.model(batch.x, batch.edge_index, batch.edge_attr)

        if cfg.gnn.att_heads > 1:
            batch.mask.unsqueeze_(1)
            batch.mask = batch.mask.expand(-1, cfg.gnn.att_heads, -1)
            batch.mask = batch.mask.reshape(-1, batch.mask.shape[1] * batch.mask.shape[2])

        batch.x = self.linear(batch.x * batch.mask)
        if cfg.gnn.gsn:
            batch.x = self.gsn(batch.x, batch.batch, batch.num_nodes)
        return batch