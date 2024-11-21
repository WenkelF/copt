import time
from copy import deepcopy
from functools import partial

import numpy as np
import torch
from torch_geometric.data import Batch
from torch_geometric.graphgym.config import cfg
from torch_geometric.graphgym.register import register_loss
from torch_geometric.utils import unbatch, unbatch_edge_index, add_self_loops, \
    remove_self_loops
from torch_scatter import scatter


def accuracy(output, target):
    return torch.mean((output.argmax(-1) == target).float())


### MAXCLIQUE ###

def maxclique_size_pyg(batch, dec_length=300, num_seeds=1):
    batch = maxclique_decoder_pyg(batch, dec_length=dec_length,
                                  num_seeds=num_seeds)

    data_list = batch.to_data_list()

    size_list = [data.c_size for data in data_list]

    return torch.Tensor(size_list).mean()


def maxclique_ratio_pyg(batch, dec_length=300, num_seeds=1):
    batch = maxclique_decoder_pyg(batch, dec_length=dec_length,
                                  num_seeds=num_seeds)

    data_list = batch.to_data_list()

    metric_list = []
    for data in data_list:
        metric_list.append(data.c_size / data.y)

    return torch.Tensor(metric_list).mean()


def get_csize(seed, data, dec_length):
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

    return c.sum()


def get_csize_np(seed, x, edge_index, num_nodes, dec_length):
    order = np.argsort(-1 * x, axis=0)
    c = np.zeros_like(x)

    edge_index = remove_self_loops(edge_index)[0]
    src, dst = edge_index[0], edge_index[1]

    c[order[seed]] = 1
    for idx in range(seed, min(dec_length, num_nodes)):
        c[order[idx]] = 1

        cTWc = np.sum(c[src] * c[dst])
        if np.sum(c) ** 2 - cTWc - np.sum(c ** 2) != 0:
            c[order[idx]] = 0

    return np.sum(c)


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


def maxclique_decoder_pyg_parallel(batch, dec_length=300, num_seeds=1):
    data_list = batch.to_data_list()

    for data in data_list:
        t0 = time.time()
        with torch.multiprocessing.Pool(processes=cfg.num_workers) as pool:
            c_size_list = pool.map(
                partial(get_csize, data=data, dec_length=dec_length),
                range(num_seeds))
        data.c_size = max(c_size_list)
        t1 = time.time()
        print(t1 - t0)

    return Batch.from_data_list(data_list)


def maxclique_ratio(output, data, dec_length=300):
    adj = data.get('adj')
    num_nodes = data.get('num_nodes')
    c = maxclique_decoder(output, adj, num_nodes, dec_length=dec_length)

    target = data.get('mc_size')

    return torch.mean(c.sum(-1) / target)


def maxclique_decoder(output, adj, num_nodes, dec_length=300):
    order = [torch.argsort(output[sample_idx][:num_nodes[sample_idx]], dim=0,
                           descending=True) for sample_idx in
             range(output.size(0))]
    c = torch.zeros_like(output)

    for sample_idx in range(output.size(0)):
        c[sample_idx][order[sample_idx][0]] = 1

        for i in range(1, min(dec_length, num_nodes[sample_idx])):
            c[sample_idx][order[sample_idx][i]] = 1

            cTWc = torch.matmul(c[sample_idx].transpose(-1, -2),
                                torch.matmul(adj[sample_idx], c[sample_idx]))
            if c[sample_idx].sum() ** 2 - cTWc - torch.sum(
                    c[sample_idx] ** 2) != 0:
                c[sample_idx][order[sample_idx][i]] = 0

    return c.squeeze(-1)


### MAXCUT ###

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
        cut_list.append(
            torch.sum(x[edge_index[0]] * x[edge_index[1]] == -1.0) / 2)

    return torch.Tensor(cut_list).mean()


def maxcut_acc(data):
    adj = data['adj']
    adj_weight = adj.sum(-1).sum(-1)
    target_size = adj_weight.clone()
    pred_size = adj_weight.clone()

    target = torch.nan_to_num(data['cut_binary'])
    target_size -= torch.matmul(target.transpose(-1, -2),
                                torch.matmul(adj, target)).squeeze()
    target = 1 - target
    target_size -= torch.matmul(target.transpose(-1, -2),
                                torch.matmul(adj, target)).squeeze()
    target_size /= 2

    output = (data['x'] > 0.5).float()
    pred_size -= torch.matmul(output.transpose(-1, -2),
                              torch.matmul(adj, output)).squeeze()
    output = 1 - output
    pred_size -= torch.matmul(output.transpose(-1, -2),
                              torch.matmul(adj, output)).squeeze()
    pred_size /= 2

    return (pred_size >= target_size).float().mean()


### COLORING ###

def color_acc(output, adj, deg_vect):
    output = (output - 0.5) * 2

    one_hot = output > 0
    bin_enc = (one_hot.float() - 0.5) * 2

    return (torch.matmul(bin_enc.transpose(-1, -2), torch.matmul(adj, bin_enc)).diagonal(dim1=-1, dim2=-2).sum(-1) / deg_vect).mean()


### PLANTEDCLIQUE ###

def plantedclique_acc_pyg(data):
    pred = torch.sigmoid(data.x) >= 0.5

    return torch.mean((pred.float() == data.y).float())


### MDS ###

def is_ds(ds, row, col):
    agg = scatter(ds.float()[row], index=col, reduce='sum')
    visited = agg >= 1.0

    return all(visited)

def mds_size_pyg(data, num_seeds: int = 1, enable: bool = True):
    if not cfg.test:
        num_seeds = 1
        if not enable:
            return torch.tensor(float('nan'))

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


### MIS ###

def mis_size_pyg(batch, dec_length=300, num_seeds=1):
    batch = mis_decoder_pyg(batch, dec_length=dec_length, num_seeds=num_seeds)

    data_list = batch.to_data_list()

    size_list = [data.is_size for data in data_list]

    return torch.Tensor(size_list).mean()


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


### MAXBIPARTITE ###

def maxbipartite_decoder(output, adj, dec_length):
    return maxclique_decoder(output, torch.matrix_power(adj, 2), dec_length)

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
