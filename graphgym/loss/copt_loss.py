from typing import Union, Tuple, List, Dict, Any

import torch

from torch_geometric.data import Batch
from torch_geometric.utils import unbatch, unbatch_edge_index, remove_self_loops
from torch_geometric.graphgym.register import register_loss

from torch_scatter import scatter


def entropy(output, epsilon=1e-8):
    batch_size = output.batch.unique().size(0)
    p = output.x.squeeze()
    entropy = - (p * torch.log(p + epsilon) + (1 - p) * torch.log(1 - p + epsilon)).sum()
    return entropy / batch_size


### MAXCLIQUE ###

@register_loss("maxclique_loss")
def maxclique_loss_pyg(batch, beta=0.1):
    data_list = batch.to_data_list()

    loss = 0.0
    for data in data_list:
        src, dst = data.edge_index[0], data.edge_index[1]

        loss1 = torch.sum(data.x[src] * data.x[dst])
        loss2 = data.x.sum() ** 2 - loss1 - torch.sum(data.x ** 2)
        loss += (- loss1 + beta * loss2) * data.num_nodes

    return loss / batch.size(0)


def maxclique_loss(output, data, beta=0.1):
    adj = data.get('adj')

    loss1 = torch.matmul(output.transpose(-1, -2), torch.matmul(adj, output))
    loss2 = output.sum() ** 2 - loss1 - torch.sum(output ** 2)

    return - loss1.sum() + beta * loss2.sum()


### MAXCUT ###

@register_loss("maxcut_loss")
def maxcut_loss_pyg(data):
    x = (data.x - 0.5) * 2
    src, dst = data.edge_index[0], data.edge_index[1]
    return torch.sum(x[src] * x[dst]) / len(data.batch.unique())


def maxcut_loss(data):
    x = (data['x'] - 0.5) * 2
    adj = data['adj_mat']
    return torch.matmul(x.transpose(-1, -2), torch.matmul(adj, x)).mean()


@register_loss("maxcut_mae")
def maxcut_mae_pyg(data):
    x = (data.x > 0.5).float()
    x = (x - 0.5) * 2
    y = data.cut_binary
    y = (y - 0.5) * 2

    x_list = unbatch(x, data.batch)
    y_list = unbatch(y, data.batch)
    edge_index_list = unbatch_edge_index(data.edge_index, data.batch)

    ae_list = []
    for x, y, edge_index in zip(x_list, y_list, edge_index_list):
        ae_list.append(torch.sum(x[edge_index[0]] * x[edge_index[1]] == -1.0) - torch.sum(y[edge_index[0]] * y[edge_index[1]] == -1.0))

    return 0.5 * torch.Tensor(ae_list).abs().mean()


def maxcut_mae(data):
    output = (data['x'] > 0.5).double()
    target = torch.nan_to_num(data['cut_binary'])

    adj = data['adj_mat']
    adj_weight = adj.sum(-1).sum(-1)
    target_size = adj_weight.clone()
    pred_size = adj_weight.clone()

    target_size -= torch.matmul(target.transpose(-1, -2), torch.matmul(adj, target)).squeeze()
    target = 1 - target
    target_size -= torch.matmul(target.transpose(-1, -2), torch.matmul(adj, target)).squeeze()
    target_size /= 2

    pred_size -= torch.matmul(output.transpose(-1, -2), torch.matmul(adj, output)).squeeze()
    output = 1 - output
    pred_size -= torch.matmul(output.transpose(-1, -2), torch.matmul(adj, output)).squeeze()
    pred_size /= 2

    return torch.mean(torch.abs(pred_size - target_size))


### COLORING ###

def color_loss(output, adj):
    output = (output - 0.5) * 2

    return torch.matmul(output.transpose(-1, -2), torch.matmul(adj, output)).diagonal(dim1=-1, dim2=-2).sum() - 4 * torch.abs(output).sum()


### PLANTEDCLIQUE ###

from torch.nn import BCEWithLogitsLoss
ce_loss = BCEWithLogitsLoss()

@register_loss("plantedclique_loss")
def plantedclique_loss_pyg(data):
    return ce_loss(data.x, data.y.unsqueeze(-1))


### MDS ###

@register_loss("mds_loss")
def mds_loss_pyg(data, beta=1.0):
    batch_size = data.batch.max() + 1.0
    
    p = data.x.squeeze()
    edge_index = remove_self_loops(data.edge_index)[0]
    row, col = edge_index[0], edge_index[1]

    loss = p.sum() + beta * (
        scatter(
            torch.log1p(-p)[row],
            index=col,
            reduce='sum',
        ).exp() * (1 - p)
    ).sum()

    return loss / batch_size


### MIS ###

# @register_loss("mis_loss")
# def mis_loss_pyg(data, beta=1.0, k=2, eps=1e-1):
#     batch_size = data.batch.max() + 1.0

#     edge_index = remove_self_loops(data.edge_index)[0]
#     row, col = edge_index[0], edge_index[1]
#     degree = torch.exp(data.degree)

#     l1 = - torch.sum(data.x ** 2)
#     l2 = + torch.sum((data.x[row] * data.x[col]) ** 2)

#     # l1 = - torch.sum(torch.log(1 - data.x) * degree)
#     # l2 = + torch.log((data.x[row] * data.x[col]) ** 1).sum()

#     # l1 = - data.x.sum()
#     # l2 = + ((data.x[row] * data.x[col]) ** k).sum()

#     loss = l1 + beta * l2

#     return loss #/ batch_size


@register_loss("mis_loss")
def mis_loss_pyg(batch, beta=0.1):
    data_list = batch.to_data_list()

    loss = 0.0
    for data in data_list:
        src, dst = data.edge_index[0], data.edge_index[1]

        loss1 = torch.sum(data.x[src] * data.x[dst])
        loss2 = data.x.sum() ** 2 - loss1 - torch.sum(data.x ** 2)
        loss += (- loss2 + beta * loss1) * data.num_nodes

    return loss / batch.size(0)


### MAXBIPARTITE ###

def maxbipartite_loss(output, adj, beta):
    return maxclique_loss(output, torch.matrix_power(adj, 2), beta)
