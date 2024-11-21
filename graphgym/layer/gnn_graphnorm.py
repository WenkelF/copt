import torch
import torch.nn.functional as F

from torch_geometric.nn.norm.graph_norm import GraphNorm
from torch_geometric.graphgym.models.layer import LayerConfig
from torch_geometric.graphgym.models.layer import new_layer_config
import torch_geometric.graphgym.register as register
from torch_geometric.graphgym.config import cfg


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
