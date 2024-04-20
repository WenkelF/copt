import torch
import torch.nn.functional as F
import torch_geometric.graphgym.register as register
from torch_geometric.graphgym.register import register_stage, register_network
from torch_geometric.graphgym.config import cfg
from torch_geometric.graphgym.models import GNN, GNNLayer


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
            if cfg.gnn.stage_type == 'skipconcat':
                d_in = dim_in if i == 0 else dim_in + i * dim_out
            else:
                d_in = dim_in if i == 0 else dim_out
            self.x_dims.append(d_in)
            layer = GNNLayer(d_in, dim_out)
            self.add_module(f'layer{i}', layer)

    def forward(self, batch):
        x_list = list()
        for i, layer in enumerate(self.children()):
            x = batch.x
            batch = layer(batch)
            if cfg.gnn.stage_type == 'skipsum':
                batch.x = x + batch.x
            elif (cfg.gnn.stage_type == 'skipconcat'
                  and i < self.num_layers - 1):
                batch.x = torch.cat([x, batch.x], dim=1)
                x_list.append(batch.x)
        if cfg.gnn.l2norm:
            batch.x = F.normalize(batch.x, p=2, dim=-1)
        batch.x_list = x_list
        return batch


@register_network('hybrid_gnn')
class HybridGNN(GNN):
    def __init__(self, dim_in: int, dim_out: int, **kwargs):
        super().__init__(dim_in, dim_out, **kwargs)
        GNNHead = register.head_dict[cfg.gnn.head]
        # TODO: decide what to do. maybe sum x_dims
        post_mp_dim_in = self.mp.x_dims
        self.post_mp = GNNHead(dim_in=post_mp_dim_in, dim_out=dim_out)

    def forward(self, batch):

        batch = self.encoder(batch)
        if cfg.gnn.layers_pre_mp > 0:
            batch = self.pre_mp(batch)
        if cfg.gnn.layers_mp > 0:
            batch = self.mp(batch)

        # TODO
        batch.x_list = torch.cat(batch.x_list, dim=-1)
        batch.x = batch.x_list
        batch = self.post_mp(batch)

        return batch
