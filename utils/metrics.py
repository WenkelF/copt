from typing import Union, Tuple, List, Dict, Any

import torch

from torch_geometric.data import Batch
from torch_geometric.utils import to_dense_adj, unbatch, unbatch_edge_index #, to_torch_sparse_tensor


def accuracy(output, target):

    return torch.mean((output.argmax(-1) == target).float())


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


def maxcut_loss(output, data):
    output = (output - 0.5) * 2

    if isinstance(data, Batch):
        adj = to_dense_adj(data['edge_index']).double()
        return torch.matmul(output.transpose(-1, -2), torch.matmul(adj, output)) / len(data.batch.unique())

    else:
        adj = data['adj_mat']
        return torch.matmul(output.transpose(-1, -2), torch.matmul(adj, output)).mean()


# def maxcut_mae(output, data):

#     target = data['cut_size']
#     num_nodes = data['num_nodes']

#     output = output.squeeze(-1) + data.get('nan_mask')

#     pred = (output > 0.5).float().sum(-1)
#     pred = torch.max(pred, num_nodes - pred)

#     return (target - pred).abs().mean()


def maxcut_mae(output, data):

    output = (output > 0.5).double()
    target = torch.nan_to_num(data['cut_binary'])

    if isinstance(data, Batch):
        edge_index_list = unbatch_edge_index(data.edge_index, data.batch)
        output_list = unbatch(output, data.batch)
        target_list = unbatch(target, data.batch)
        abs_error_list = []
        for edge_index, output, target in zip(edge_index_list, output_list, target_list):
            target = target.double()
            adj = to_dense_adj(edge_index).double()
            adj_weight = adj.sum()
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

            abs_error_list.append(torch.abs(pred_size - target_size))
        
        return torch.mean(torch.Tensor(abs_error_list))
    
    else:
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


def maxcut_p_correct(output, data):

    adj = data['adj']
    adj_weight = adj.sum(-1).sum(-1)
    target_size = adj_weight.clone()
    pred_size = adj_weight.clone()

    target = torch.nan_to_num(data['cut_binary'])
    target_size -= torch.matmul(target.transpose(-1, -2), torch.matmul(adj, target)).squeeze()
    target = 1 - target
    target_size -= torch.matmul(target.transpose(-1, -2), torch.matmul(adj, target)).squeeze()
    target_size /= 2

    output = (output > 0.5).float()
    pred_size -= torch.matmul(output.transpose(-1, -2), torch.matmul(adj, output)).squeeze()
    output = 1 - output
    pred_size -= torch.matmul(output.transpose(-1, -2), torch.matmul(adj, output)).squeeze()
    pred_size /= 2

    return (pred_size >= target_size).float().mean()


def maxcut_acc(output, data):

    target = data['cut_binary'].squeeze(-1)

    output = output.squeeze(-1) + data.get('nan_mask')

    label = (output > 0.5).float()

    return torch.max(1 - torch.nanmean(torch.abs(label - target), dim=-1), 1 - torch.nanmean(torch.abs((1-label) - target), dim=-1)).mean()


def maxcut_p_exact(output, data):

    target = data['cut_binary'].squeeze(-1)

    output = output.squeeze(-1) + data.get('nan_mask')

    label = (output > 0.5).float()

    return torch.mean((torch.nanmean(1 - torch.abs(label - target), dim=-1) == 1).float())


def color_loss(output, adj):

    output = (output - 0.5) * 2

    return torch.matmul(output.transpose(-1, -2), torch.matmul(adj, output)).diagonal(dim1=-1, dim2=-2).sum() - 4 * torch.abs(output).sum()


def color_acc(output, adj, deg_vect):

    output = (output - 0.5) * 2

    one_hot = output > 0
    bin_enc = (one_hot.float() - 0.5) * 2

    return (torch.matmul(bin_enc.transpose(-1, -2), torch.matmul(adj, bin_enc)).diagonal(dim1=-1, dim2=-2).sum(-1) / deg_vect).mean()