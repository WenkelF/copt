from typing import Union, Tuple, List, Dict, Any

import torch

import time

from torch_geometric.data import Batch
from torch_geometric.utils import unbatch, unbatch_edge_index,  add_self_loops, remove_self_loops
from torch_scatter import scatter

from torch_geometric.graphgym.register import register_loss

from copy import deepcopy


def accuracy(output, target):

    return torch.mean((output.argmax(-1) == target).float())


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


def maxclique_size_pyg(batch, dec_length=300, num_seeds=1):

    batch = maxclique_decoder_pyg(batch, dec_length=dec_length, num_seeds=num_seeds)

    data_list = batch.to_data_list()

    size_list = [data.c_size for data in data_list]

    return torch.Tensor(size_list).mean()


def mis_size_pyg(batch, dec_length=300, num_seeds=1):

    batch = mis_decoder_pyg(batch, dec_length=dec_length, num_seeds=num_seeds)

    data_list = batch.to_data_list()

    size_list = [data.is_size for data in data_list]

    return torch.Tensor(size_list).mean()


def maxclique_ratio_pyg(batch, dec_length=300, num_seeds=1):

    batch = maxclique_decoder_pyg(batch, dec_length=dec_length, num_seeds=num_seeds)

    data_list = batch.to_data_list()

    metric_list = []
    for data in data_list:
        metric_list.append(data.c_size / data.y)

    return torch.Tensor(metric_list).mean()


def maxclique_decoder_pyg(batch, dec_length=300, num_seeds=1):

    data_list = batch.to_data_list()

    for data in data_list:
        c_size_list = []

        for seed in range(num_seeds):

            order = torch.argsort(data.x, dim=0, descending=True)
            c = torch.zeros_like(data.x)

            edge_index = remove_self_loops(data.edge_index)[0]
            src, dst = edge_index[0], edge_index[1]
            
            c[order[seed]] = 1
            for idx in range(seed, min(dec_length, data.num_nodes)):
                c[order[idx]] = 1

                cTWc = torch.sum(c[src] * c[dst])
                if c.sum() ** 2 - cTWc - torch.sum(c ** 2) != 0:
                    c[order[idx]] = 0

            c_size_list.append(c.sum())

        data.c_size = max(c_size_list)

    return Batch.from_data_list(data_list)


def mis_decoder_pyg(batch, dec_length=300, num_seeds=1):

    data_list = batch.to_data_list()

    for data in data_list:
        is_size_list = []

        for seed in range(num_seeds):

            order = torch.argsort(data.x, dim=0, descending=True)
            c = torch.zeros_like(data.x)

            edge_index = remove_self_loops(data.edge_index)[0]
            src, dst = edge_index[0], edge_index[1]
            
            c[order[seed]] = 1
            for idx in range(seed, min(dec_length, data.num_nodes)):
                c[order[idx]] = 1

                cTWc = torch.sum(c[src] * c[dst])
                if cTWc != 0:
                    c[order[idx]] = 0

            is_size_list.append(c.sum())

        data.is_size = max(is_size_list)

    return Batch.from_data_list(data_list)


def maxclique_loss(output, data, beta=0.1):

    adj = data.get('adj')

    loss1 = torch.matmul(output.transpose(-1, -2), torch.matmul(adj, output))
    loss2 = output.sum() ** 2 - loss1 - torch.sum(output ** 2)

    return - loss1.sum() + beta * loss2.sum()


def maxclique_ratio(output, data, dec_length=300):

    adj = data.get('adj')
    num_nodes = data.get('num_nodes')
    c = maxclique_decoder(output, adj, num_nodes, dec_length=dec_length)

    target = data.get('mc_size')

    return torch.mean(c.sum(-1) / target)


def maxclique_decoder(output, adj, num_nodes, dec_length=300):

    order = [torch.argsort(output[sample_idx][:num_nodes[sample_idx]], dim=0, descending=True) for sample_idx in range(output.size(0))]
    c = torch.zeros_like(output)

    for sample_idx in range(output.size(0)):
        c[sample_idx][order[sample_idx][0]] = 1

        for i in range(1, min(dec_length, num_nodes[sample_idx])):
            c[sample_idx][order[sample_idx][i]] = 1

            cTWc = torch.matmul(c[sample_idx].transpose(-1, -2), torch.matmul(adj[sample_idx], c[sample_idx]))
            if c[sample_idx].sum() ** 2 - cTWc - torch.sum(c[sample_idx] ** 2) != 0:
                c[sample_idx][order[sample_idx][i]] = 0

    return c.squeeze(-1)


def maxbipartite_loss(output, adj, beta):

    return maxclique_loss(output, torch.matrix_power(adj, 2), beta)


def maxbipartite_decoder(output, adj, dec_length):

    return maxclique_decoder(output, torch.matrix_power(adj, 2), dec_length)


def maxcut_loss_pyg(data):
    x = (data.x - 0.5) * 2
    src, dst = data.edge_index[0], data.edge_index[1]

    return torch.sum(x[src] * x[dst]) / len(data.batch.unique())


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


def maxcut_acc_pyg(data):

    x = (data.x > 0.5).float()
    x = (x - 0.5) * 2
    y = data.cut_binary
    y = (y - 0.5) * 2

    x_list = unbatch(x, data.batch)
    y_list = unbatch(y, data.batch)
    edge_index_list = unbatch_edge_index(data.edge_index, data.batch)

    comparison_list = []
    for x, y, edge_index in zip(x_list, y_list, edge_index_list):
        x_cut = torch.sum(x[edge_index[0]] * x[edge_index[1]] == -1.0)
        y_cut = torch.sum(y[edge_index[0]] * y[edge_index[1]] == -1.0)
        comparison_list.append(x_cut >= y_cut)

    return torch.Tensor(comparison_list).mean()


def maxcut_size_pyg(data):

    x = (data.x > 0.5).float()
    x = (x - 0.5) * 2

    x_list = unbatch(x, data.batch)
    edge_index_list = unbatch_edge_index(data.edge_index, data.batch)

    cut_list = []
    for x, edge_index in zip(x_list, edge_index_list):
        cut_list.append(torch.sum(x[edge_index[0]] * x[edge_index[1]] == -1.0) / 2)

    return torch.Tensor(cut_list).mean()


def maxcut_loss(data):
    x = (data['x'] - 0.5) * 2
    adj = data['adj_mat']
    
    return torch.matmul(x.transpose(-1, -2), torch.matmul(adj, x)).mean()


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


def maxcut_acc(data):

    adj = data['adj']
    adj_weight = adj.sum(-1).sum(-1)
    target_size = adj_weight.clone()
    pred_size = adj_weight.clone()

    target = torch.nan_to_num(data['cut_binary'])
    target_size -= torch.matmul(target.transpose(-1, -2), torch.matmul(adj, target)).squeeze()
    target = 1 - target
    target_size -= torch.matmul(target.transpose(-1, -2), torch.matmul(adj, target)).squeeze()
    target_size /= 2

    output = (data['x'] > 0.5).float()
    pred_size -= torch.matmul(output.transpose(-1, -2), torch.matmul(adj, output)).squeeze()
    output = 1 - output
    pred_size -= torch.matmul(output.transpose(-1, -2), torch.matmul(adj, output)).squeeze()
    pred_size /= 2

    return (pred_size >= target_size).float().mean()


def color_loss(output, adj):

    output = (output - 0.5) * 2

    return torch.matmul(output.transpose(-1, -2), torch.matmul(adj, output)).diagonal(dim1=-1, dim2=-2).sum() - 4 * torch.abs(output).sum()


def color_acc(output, adj, deg_vect):

    output = (output - 0.5) * 2

    one_hot = output > 0
    bin_enc = (one_hot.float() - 0.5) * 2

    return (torch.matmul(bin_enc.transpose(-1, -2), torch.matmul(adj, bin_enc)).diagonal(dim1=-1, dim2=-2).sum(-1) / deg_vect).mean()


def plantedclique_acc_pyg(data):
    pred = torch.sigmoid(data.x) >= 0.5

    return torch.mean((pred.float() == data.y).float())


def mds_size_pyg(data, mum_seeds: int = 3):
    
    data_list = data.to_data_list()

    ds_list = []
    for data in data_list:
        edge_index = add_self_loops(data.edge_index)[0]
        row, col = edge_index[0], edge_index[1]

        mds_size_list = []
        for skip in range(num_seeds):
            ds = torch.zeros_like(data.x).squeeze()
            p = deepcopy(data.x).squeeze()
            
            if skip > 0:
                for _ in range(skip):
                    idx = torch.argmax(p)
                    p[idx] = - torch.inf
        
            t0 = time.time()
            while not is_ds(ds, row, col):
                if torch.max(p) == - torch.inf:
                    break   # break in case skipping top nodes prohibits finding a ds; should prevent infinite loops
                
                idx = torch.argmax(p)
                ds[idx] = True
                p[idx] = - torch.inf

            if is_ds(ds, row, col):
                mds_size_list.append(ds.sum())
            else:
                mds_size_list.append(len(p))    # this case should rarely happen (only if break is triggered above). But let's be conservative just in case and set the ds to the entire node set
        
        ds_list.append(min(mds_size_list))

    return torch.Tensor(ds_list).mean()


def mds_acc_pyg(data):
    data_list = data.to_data_list()

    ds_list = []
    for data in data_list:
        p = deepcopy(data.x).squeeze()
        edge_index = add_self_loops(data.edge_index)[0]
        row, col = edge_index[0], edge_index[1]

        ds = (data.x >= 0.5).squeeze()

        p[ds] = - torch.inf
        
        while not is_ds(ds, row, col):
            idx = torch.argmax(p)
            ds[idx] = True
            p[idx] = - torch.inf

        if is_ds(ds, row, col):
            ds_list.append(True)
        else:
            ds_list.append(False)

    return torch.Tensor(ds_list).mean()


def is_ds(ds, row, col):
    agg = scatter(ds.float()[row], index=col, reduce='sum')
    visited = agg >= 1.0

    return all(visited)


# def mis_size_pyg(data):

#     # eval = False
#     # if not eval:
#     #     return 0.

#     data_list = data.to_data_list()

#     iset_list = []
#     for data in data_list:
#         p = deepcopy(data.x).squeeze()
#         edge_index = remove_self_loops(data.edge_index)[0]
#         row, col = edge_index[0], edge_index[1]

#         iset = (data.x >= 0.5).squeeze()

#         if is_iset(iset, row, col) and any(iset):
#             p[iset] = - torch.inf

#             while True:
#                 idx = torch.argmax(p)
#                 iset[idx] = True
#                 p[idx] = - torch.inf

#                 if not is_iset(iset, row, col):
#                     iset[idx] = False
#                     break

#             iset_list.append(iset.sum())

#         else:
#             iset = torch.zeros_like(iset)
            
#             while True:
#                 idx = torch.argmax(p)
#                 iset[idx] = True
#                 p[idx] = - torch.inf

#                 if not is_iset(iset, row, col):
#                     iset[idx] = False
#                     break

#             iset_list.append(iset.sum())

#     return torch.Tensor(iset_list).mean()


# def is_iset(iset, row, col):

#     edges = iset[row] * iset[col]

#     return all(edges == 0.)