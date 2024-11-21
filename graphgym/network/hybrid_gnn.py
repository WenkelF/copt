import torch

import torch_geometric.graphgym.register as register
from torch_geometric.graphgym.register import register_network
from torch_geometric.graphgym.config import cfg
from torch_geometric.graphgym.models import GNN


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
