from functools import partial

import torch
import torch.nn as nn
from torch_geometric.graphgym.config import cfg
from torch_geometric.graphgym.register import register_act


class SWISH(nn.Module):
    def __init__(self, inplace=False):
        super().__init__()
        self.inplace = inplace

    def forward(self, x):
        if self.inplace:
            x.mul_(torch.sigmoid(x))
            return x
        else:
            return x * torch.sigmoid(x)


class Sigmoid(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, batch):
        batch.x = torch.sigmoid(batch.x)
        return batch


register_act('swish', partial(SWISH, inplace=cfg.mem.inplace))
register_act('lrelu_03', partial(nn.LeakyReLU, 0.3, inplace=cfg.mem.inplace))

# Add Gaussian Error Linear Unit (GELU).
register_act('gelu', nn.GELU)
register_act('sigmoid', Sigmoid)
