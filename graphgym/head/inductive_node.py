import torch
import torch.nn as nn
from torch_geometric.graphgym import register
from torch_geometric.graphgym.config import cfg
from torch_geometric.graphgym.models.layer import new_layer_config, MLP, \
    LayerConfig, GeneralMultiLayer, Linear
from torch_geometric.graphgym.register import register_head
from torch_geometric.utils import unbatch

from dataclasses import dataclass, replace


def _apply_index(batch):
    pred, true = batch.x, batch.y
    if cfg.virtual_node:
        # Remove virtual node
        idx = torch.concat([
            torch.where(batch.batch == i)[0][:-1]
            for i in range(batch.batch.max().item() + 1)
        ])
        pred, true = pred[idx], true[idx]
    return pred, true


def minmax_norm_pyg(data):
    x_list = unbatch(data.x, data.batch)

    p_list = []
    for x in x_list:
        x_min = x.min()
        x_max = x.max()

        p_list.append((x - x_min) / (x_max - x_min + 1e-6))

    data.x = torch.cat(p_list, dim=0)

    return data


@register_head('inductive_node')
class GNNInductiveNodeHead(nn.Module):
    """
    GNN prediction head for inductive node prediction tasks.

    Args:
        dim_in (int): Input dimension
        dim_out (int): Output dimension. For binary prediction, dim_out=1.
    """

    def __init__(self, dim_in, dim_out):
        super(GNNInductiveNodeHead, self).__init__()
        self.layer_post_mp = MLP(
            new_layer_config(dim_in, dim_out, cfg.gnn.layers_post_mp,
                             has_act=False, has_bias=True, cfg=cfg))

    def forward(self, batch):
        batch = self.layer_post_mp(batch)
        pred, true = _apply_index(batch)
        return pred, true


class MLP_custom_act(torch.nn.Module):
    """A basic MLP model.

    Args:
        layer_config (LayerConfig): The configuration of the layer.
        **kwargs (optional): Additional keyword arguments.
    """
    def __init__(self, layer_config: LayerConfig, **kwargs):
        super().__init__()
        if layer_config.dim_inner is None:
            dim_inner = layer_config.dim_in
        else:
            dim_inner = layer_config.dim_inner

        layer_config.has_bias = True
        layers = []
        if layer_config.num_layers > 1:
            sub_layer_config = LayerConfig(
                num_layers=layer_config.num_layers - 1,
                dim_in=layer_config.dim_in, dim_out=dim_inner,
                dim_inner=dim_inner, final_act=True, act=cfg.gnn.act)
            layers.append(GeneralMultiLayer('linear', sub_layer_config))
            layer_config = replace(layer_config, dim_in=dim_inner)
            layers.append(Linear(layer_config))
        else:
            layers.append(Linear(layer_config))
        self.model = torch.nn.Sequential(*layers)

    def forward(self, batch):
        if isinstance(batch, torch.Tensor):
            batch = self.model(batch)
        else:
            batch.x = self.model(batch.x)
        return batch

@register_head('copt_inductive_node')
class COPTInductiveNodeHead(nn.Module):
    """
    GNN prediction head for inductive node prediction tasks.

    Args:
        dim_in (int): Input dimension
        dim_out (int): Output dimension. For binary prediction, dim_out=1.
    """

    def __init__(self, dim_in, dim_out):
        super(COPTInductiveNodeHead, self).__init__()
        norm_dict = {'minmax': minmax_norm_pyg}
        self.layer_post_mp = MLP_custom_act(
            new_layer_config(dim_in, dim_out, cfg.gnn.layers_post_mp,
                             has_act=False, has_bias=True, cfg=cfg))
        self.last_act = None if cfg.gnn.last_act is None else register.act_dict[cfg.gnn.last_act]()
        self.last_norm = None if cfg.gnn.last_norm is None else norm_dict[cfg.gnn.last_norm]

    def forward(self, batch):
        batch = self.layer_post_mp(batch)
        batch = batch if self.last_act is None else self.last_act(batch)
        batch = batch if self.last_norm is None else self.last_norm(batch)
        return batch

@register_head('inductive_node_multi')
class GNNInductiveNodeMultiHead(nn.Module):
    """
    GNN prediction head for inductive node prediction tasks (one MLP per task).

    Args:
        dim_in (int): Input dimension
        dim_out (int): Output dimension. For binary prediction, dim_out=1.
    """

    def __init__(self, dim_in, dim_out):
        super().__init__()

        layer_config = new_layer_config(dim_in, 1, cfg.gnn.layers_post_mp,
                                        has_act=False, has_bias=True, cfg=cfg)
        if cfg.gnn.multi_head_dim_inner is not None:
            layer_config.dim_inner = cfg.gnn.multi_head_dim_inner

        self.layer_post_mp = nn.ModuleList([MLP(layer_config)
                                            for _ in range(dim_out)])

    def forward(self, batch):
        batch.x = torch.hstack([m(batch.x) for m in self.layer_post_mp])
        pred, true = _apply_index(batch)
        return pred, true
