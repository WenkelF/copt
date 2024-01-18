from abc import ABC, abstractmethod
from typing import Any

import torch
import torch.nn as nn
from torch_geometric.graphgym import cfg
from torch_geometric.graphgym.register import register_node_encoder


@register_node_encoder("GraphStats")
class GraphStatsEncoder(torch.nn.Module):

    def __init__(self, dim_emb, expand_x=True):
        super().__init__()

        dim_in = cfg.share.dim_in  # Expected original input node features dim
        stats_dim = len(cfg.dataset.graph_stats)

        dim_pe = cfg.posenc_GraphStats.dim_pe  # Size of the kernel-based PE embedding
        model_type = cfg.posenc_GraphStats.model.lower()  # Encoder NN model type for PEs
        n_layers = cfg.posenc_GraphStats.layers  # Num. layers in PE encoder model
        norm_type = cfg.posenc_GraphStats.raw_norm_type.lower()  # Raw PE normalization layer type
        self.pass_as_var = cfg.posenc_GraphStats.pass_as_var  # Pass PE also as a separate variable

        if dim_emb - dim_pe < 0:
            raise ValueError(f"PE dim size {dim_pe} is too large for "
                             f"desired embedding size of {dim_emb}.")

        if expand_x:
            self.linear_x = nn.Linear(dim_in, dim_emb - dim_pe)
        self.expand_x = expand_x

        if norm_type == 'batchnorm':
            self.raw_norm = nn.BatchNorm1d(stats_dim)
        else:
            self.raw_norm = None

        activation = nn.ReLU  # register.act_dict[cfg.gnn.act]
        if model_type == 'mlp':
            layers = []
            if n_layers == 1:
                layers.append(nn.Linear(stats_dim, dim_pe))
                layers.append(activation())
            else:
                layers.append(nn.Linear(stats_dim, 2 * dim_pe))
                layers.append(activation())
                for _ in range(n_layers - 2):
                    layers.append(nn.Linear(2 * dim_pe, 2 * dim_pe))
                    layers.append(activation())
                layers.append(nn.Linear(2 * dim_pe, dim_pe))
                layers.append(activation())
            self.pe_encoder = nn.Sequential(*layers)
        elif model_type == 'linear':
            self.pe_encoder = nn.Linear(stats_dim, dim_pe)
        else:
            raise ValueError(f"{self.__class__.__name__}: Does not support "
                             f"'{model_type}' encoder model.")

    def forward(self, batch):
        stats = list()
        for stat in cfg.dataset.graph_stats:
            stats.append(getattr(batch, stat))

        pos_enc = torch.cat(stats, dim=1).float()
        # pos_enc = batch.rw_landing  # (Num nodes) x (Num kernel times)
        if self.raw_norm:
            pos_enc = self.raw_norm(pos_enc)
        pos_enc = self.pe_encoder(pos_enc)  # (Num nodes) x dim_pe

        # Expand node features if needed
        if self.expand_x:
            h = self.linear_x(batch.x)
        else:
            h = batch.x
        # Concatenate final PEs to input embedding
        batch.x = torch.cat((h, pos_enc), 1)
        # Keep PE also separate in a variable (e.g. for skip connections to input)
        if self.pass_as_var:
            setattr(batch, f'pe_{self.kernel_type}', pos_enc)
        return batch
