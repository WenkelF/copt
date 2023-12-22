import torch
import torch.nn as nn
from torch_geometric.graphgym import register
from torch_geometric.graphgym.config import cfg
from torch_geometric.graphgym.models.layer import new_layer_config, MLP
from torch_geometric.graphgym.register import register_head, pooling_dict


@register_head('copt_graph')
class COPTGraphHead(nn.Module):
    """
    GNN prediction head for graph prediction tasks.

    Args:
        dim_in (int): Input dimension
        dim_out (int): Output dimension. For binary prediction, dim_out=1.
    """

    def __init__(self, dim_in, dim_out):
        super(COPTGraphHead, self).__init__()
        self.pooling_fn = pooling_dict[cfg.model.graph_pooling]

        self.layer_post_mp = MLP(
            new_layer_config(dim_in, dim_out, cfg.gnn.layers_post_mp,
                             has_act=False, has_bias=True, cfg=cfg))
        
        self.last_act = None if cfg.gnn.last_act is None else register.act_dict[cfg.gnn.last_act]()

    def _apply_index(self, batch):
        return batch.x, batch.y

    def forward(self, batch):
        batch.x = self.pooling_fn(batch.x, batch.batch)
        batch.x = self.layer_post_mp(batch.x)

        batch.x = batch.x if self.last_act is None else self.last_act(batch).x
        return self._apply_index(batch)
