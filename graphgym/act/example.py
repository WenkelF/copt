from functools import partial

import torch
import torch.nn as nn
from torch_geometric.graphgym.config import cfg
from torch_geometric.graphgym.register import register_act#, register_norm
from torch_geometric.utils import unbatch


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


class LinearAct(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x


class Sigmoid(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, batch):
        batch.x = torch.sigmoid(batch.x)
        return batch


def minmax_norm_pyg(data):
    x_list = unbatch(data.x, data.batch)

    p_list = []
    for x in x_list:
        x_min = x.min()
        x_max = x.max()

        p_list.append((x - x_min) / (x_max - x_min + 1e-6))

    data.x = torch.cat(p_list, dim=0)

    return data


register_act('swish', partial(SWISH, inplace=cfg.mem.inplace))
register_act('lrelu_001', partial(nn.LeakyReLU, 0.01, inplace=cfg.mem.inplace))
register_act('lrelu_03', partial(nn.LeakyReLU, 0.3, inplace=cfg.mem.inplace))
register_act('linear', LinearAct)

# Add Gaussian Error Linear Unit (GELU).
register_act('gelu', nn.GELU)
register_act('sigmoid', Sigmoid)


#register_norm('minmax', minmax_norm_pyg)