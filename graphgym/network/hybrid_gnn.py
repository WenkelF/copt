import torch
import torch.nn.functional as F
from torch_scatter import scatter
from torch_geometric.nn.norm.graph_norm import GraphNorm
from torch_geometric.graphgym.models.layer import LayerConfig
from torch_geometric.graphgym.models.layer import new_layer_config
import torch_geometric.graphgym.register as register
from torch_geometric.graphgym.register import register_stage, register_network
from torch_geometric.graphgym.config import cfg
from torch_geometric.graphgym.models import GNN, GNNLayer


def norm_by_graph(batch):
    factor = torch.sqrt(scatter(torch.ones_like(batch.batch), batch.batch)[batch.batch])
    batch.x = batch.x / factor.unsqueeze(1)
    return batch

class GeneralLayerGN(torch.nn.Module):
    r"""A general wrapper for layers.

    Args:
        name (str): The registered name of the layer.
        layer_config (LayerConfig): The configuration of the layer.
        **kwargs (optional): Additional keyword arguments.
    """
    def __init__(self, name, layer_config: LayerConfig, **kwargs):
        super().__init__()
        self.has_l2norm = layer_config.has_l2norm
        has_bn = layer_config.has_batchnorm
        layer_config.has_bias = not has_bn
        self.layer = register.layer_dict[name](layer_config, **kwargs)
        layer_wrapper = []
        if has_bn:
            layer_wrapper.append(
                GraphNorm(
                    layer_config.dim_out,
                    eps=layer_config.bn_eps,
                ))
        if layer_config.dropout > 0:
            layer_wrapper.append(
                torch.nn.Dropout(
                    p=layer_config.dropout,
                    inplace=layer_config.mem_inplace,
                ))
        if layer_config.has_act:
            layer_wrapper.append(register.act_dict[layer_config.act]())
        self.post_layer = torch.nn.Sequential(*layer_wrapper)

    def forward(self, batch):
        batch = self.layer(batch)
        if isinstance(batch, torch.Tensor):
            batch = self.post_layer(batch)
            if self.has_l2norm:
                batch = F.normalize(batch, p=2, dim=1)
        else:
            batch.x = self.post_layer(batch.x)
            if self.has_l2norm:
                batch.x = F.normalize(batch.x, p=2, dim=1)
        return batch


def GNNLayerGN(dim_in: int, dim_out: int, has_act: bool = True) -> GeneralLayerGN:
    r"""Creates a GNN layer, given the specified input and output dimensions
    and the underlying configuration in :obj:`cfg`.

    Args:
        dim_in (int): The input dimension
        dim_out (int): The output dimension.
        has_act (bool, optional): Whether to apply an activation function
            after the layer. (default: :obj:`True`)
    """
    return GeneralLayerGN(
        cfg.gnn.layer_type,
        layer_config=new_layer_config(
            dim_in,
            dim_out,
            1,
            has_act=has_act,
            has_bias=False,
            cfg=cfg,
        ),
    )


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
            x = batch.x
            batch = layer(batch)
            if cfg.gnn.norm_by_graph:
                batch = norm_by_graph(batch)
            x_list.append(batch.x)
            if cfg.gnn.stage_type == 'skipsum_concat':
                batch.x = x + batch.x
            elif (cfg.gnn.stage_type == 'skipconcat_concat'
                  and i < self.num_layers - 1):
                batch.x = torch.cat([x, batch.x], dim=1)
        batch.x_list = x_list
        return batch


@register_network('hybrid_gnn')
class HybridGNN(GNN):
    def __init__(self, dim_in: int, dim_out: int, **kwargs):
        super().__init__(dim_in, dim_out, **kwargs)
        GNNHead = register.head_dict[cfg.gnn.head]
        # TODO: decide what to do. maybe sum x_dims
        self.stage = cfg.gnn.hybrid_stack
        if self.stage == 'sum':
            post_mp_dim_in = self.mp.x_dims[0]
        elif self.stage == 'concat':
            post_mp_dim_in = sum(self.mp.x_dims)
        else:
            raise ValueError('Stage {} is not supported.'.format(self.stage))
        self.post_mp = GNNHead(dim_in=post_mp_dim_in, dim_out=dim_out)

    def forward(self, batch):

        batch = self.encoder(batch)
        if cfg.gnn.layers_pre_mp > 0:
            batch = self.pre_mp(batch)
        if cfg.gnn.layers_mp > 0:
            batch = self.mp(batch)

        # TODO
        if self.stage == 'sum':
            x_list = torch.stack(batch.x_list, dim=-1)
            x_list = torch.sum(x_list, dim=-1)
        elif self.stage == 'concat':
            x_list = torch.cat(batch.x_list, dim=-1)
        batch.x = x_list
        batch = self.post_mp(batch)

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