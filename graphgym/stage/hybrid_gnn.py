import torch
import torch.nn.functional as F
from torch_geometric.nn import GraphSizeNorm

from torch_geometric.graphgym.register import register_stage
from torch_geometric.graphgym.config import cfg
from torch_geometric.graphgym.models import GNNLayer

from graphgym.layer.gnn_graphnorm import GNNLayerGN

@register_stage('stack_concat')
@register_stage('skipsum_concat')
@register_stage('skipconcat_concat')
class GNNStackStageConcat(torch.nn.Module):
    r"""Stacks a number of GNN layers.

    Args:
        dim_in (int): The input dimension
        dim_out (int): The output dimension.
        num_layers (int): The number of layers.
    """
    def __init__(self, dim_in, dim_out, num_layers):
        super().__init__()
        self.num_layers = num_layers
        self.x_dims = list()
        self.gsn = GraphSizeNorm()
        for i in range(num_layers):
            if cfg.gnn.stage_type == 'skipconcat_concat':
                d_in = dim_in if i == 0 else dim_in + i * dim_out
            else:
                d_in = dim_in if i == 0 else dim_out
            self.x_dims.append(d_in)
            if cfg.gnn.graphnorm:
                layer = GNNLayerGN(d_in, dim_out, has_act=False)
            else:
                layer = GNNLayer(d_in, dim_out,has_act=False)
            self.add_module(f'layer{i}', layer)

    def forward(self, batch):
        x_list = []
        for i, layer in enumerate(self.children()):
            if i == 0:
                continue
            x = batch.x
            batch = layer(batch)
            if cfg.gnn.gsn:
                batch.x = self.gsn(batch.x, batch.batch, batch.num_nodes)
            x_list.append(batch.x)
            if cfg.gnn.stage_type == 'skipsum_concat':
                batch.x = x + batch.x
            elif (cfg.gnn.stage_type == 'skipconcat_concat'
                  and i < self.num_layers - 1):
                batch.x = torch.cat([x, batch.x], dim=1)
        batch.x_list = x_list
        return batch


@register_stage('stack_gn')
@register_stage('skipsum_gn')
@register_stage('skipconcat_gn')
class GNNStackStage(torch.nn.Module):
    r"""Stacks a number of GNN layers.

    Args:
        dim_in (int): The input dimension
        dim_out (int): The output dimension.
        num_layers (int): The number of layers.
    """
    def __init__(self, dim_in, dim_out, num_layers):
        super().__init__()
        self.num_layers = num_layers
        for i in range(num_layers):
            if cfg.gnn.stage_type == 'skipconcat':
                d_in = dim_in if i == 0 else dim_in + i * dim_out
            else:
                d_in = dim_in if i == 0 else dim_out
            layer = GNNLayerGN(d_in, dim_out)
            self.add_module(f'layer{i}', layer)

    def forward(self, batch):
        for i, layer in enumerate(self.children()):
            x = batch.x
            batch = layer(batch)
            if cfg.gnn.stage_type == 'skipsum':
                batch.x = x + batch.x
            elif (cfg.gnn.stage_type == 'skipconcat'
                  and i < self.num_layers - 1):
                batch.x = torch.cat([x, batch.x], dim=1)
        if cfg.gnn.l2norm:
            batch.x = F.normalize(batch.x, p=2, dim=-1)
        return batch