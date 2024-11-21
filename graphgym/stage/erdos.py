import torch
import torch.nn.functional as F

from torch_geometric.graphgym.models.layer import GeneralLayer
from torch_geometric.graphgym.models.layer import new_layer_config
from torch_geometric.graphgym.register import register_stage
from torch_geometric.graphgym.config import cfg
from torch_geometric.graphgym.models import GNNLayer

from graphgym.layer.gnn_graphnorm import GNNLayerGN


def GATLayer(dim_in: int, dim_out: int, has_act: bool = True) -> GeneralLayer:
    r"""Creates a GNN layer, given the specified input and output dimensions
    and the underlying configuration in :obj:`cfg`.

    Args:
        dim_in (int): The input dimension
        dim_out (int): The output dimension.
        has_act (bool, optional): Whether to apply an activation function
            after the layer. (default: :obj:`True`)
    """
    return GeneralLayer(
        'erdosgatconv',
        layer_config=new_layer_config(
            dim_in,
            dim_out,
            1,
            has_act=has_act,
            has_bias=False,
            cfg=cfg,
        ),
    )


@register_stage('stack_erdos')
@register_stage('skipsum_erdos')
@register_stage('skipconcat_erdos')
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
        for i in range(num_layers - 1):
            if cfg.gnn.stage_type == 'skipconcat':
                d_in = dim_in if i == 0 else dim_in + i * dim_out
            else:
                d_in = dim_in if i == 0 else dim_out
            layer = GNNLayer(d_in, dim_out)
            self.add_module(f'layer{i}', layer)

        if cfg.gnn.stage_type == 'skipconcat':
            d_in = dim_in + num_layers * dim_out
        else:
            d_in = dim_out
        layer = GATLayer(d_in, dim_out)
        self.add_module(f'layer{num_layers - 1}', layer)

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

    @register_stage('stack_concat_erdos')
    @register_stage('skipsum_concat_erdos')
    @register_stage('skipconcat_concat_erdos')
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
            for i in range(num_layers - 1):
                if cfg.gnn.stage_type == 'skipconcat_concat_erdos':
                    d_in = dim_in if i == 0 else dim_in + i * dim_out
                else:
                    d_in = dim_in if i == 0 else dim_out
                self.x_dims.append(d_in)
                if cfg.gnn.graphnorm:
                    layer = GNNLayerGN(d_in, dim_out, has_act=False)
                else:
                    layer = GNNLayer(d_in, dim_out, has_act=False)
                self.add_module(f'layer{i}', layer)

            if cfg.gnn.stage_type == 'skipconcat':
                d_in = dim_in + num_layers * dim_out
            else:
                d_in = dim_out
            self.x_dims.append(d_in)
            layer = GATLayer(d_in, dim_out)
            self.add_module(f'layer{num_layers - 1}', layer)

        def forward(self, batch):
            x_list = []
            for i, layer in enumerate(self.children()):
                x = batch.x
                batch = layer(batch)
                x_list.append(batch.x)
                if cfg.gnn.stage_type == 'skipsum_concat_erdos':
                    batch.x = x + batch.x
                elif (cfg.gnn.stage_type == 'skipconcat_concat_erdos'
                      and i < self.num_layers - 1):
                    batch.x = torch.cat([x, batch.x], dim=1)
            batch.x_list = x_list
            return batch