from typing import Union, Tuple, List, Dict, Any

import torch


class Loss:
    def __init__(
        self,
        task: str,
        kwargs: Dict[str, Any] = None
    ):

        if task == 'maxclique':
            self.loss_fn = lambda output, data: maxclique_loss(output, data, **kwargs)
            self.decoder = maxclique_decoder
        elif task == 'maxcut':
            self.loss_fn = lambda output, data: maxcut_loss(output, data)
            self.decoder = None
        else:
            raise ValueError(f"Invalid task: {task}")

    def loss(self, output, data):

        return self.loss_fn(output, data)


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

    adj = data['adj']

    output = (output - 0.5) * 2

    return - torch.matmul(output.transpose(-1, -2), torch.matmul(adj, output)).sum()


def maxcut_mae(output, data):

    target = data['target']
    num_nodes = data['num_nodes']

    pred = (output.squeeze(-1) > 0.5).float().sum(-1)
    pred = torch.max(pred, num_nodes - pred)

    return (target - pred).abs().mean()


def maxcut_acc(output, data):

    target = data['cut_binary']

    label = (output > 0.5).float()

    return max(1 - torch.nanmean(torch.abs(label - target)), 1 - torch.nanmean(torch.abs((1-label) - target)))


def color_loss(output, adj):

    output = (output - 0.5) * 2

    return torch.matmul(output.transpose(-1, -2), torch.matmul(adj, output)).diagonal(dim1=-1, dim2=-2).sum() - 4 * torch.abs(output).sum()


def color_acc(output, adj, deg_vect):

    output = (output - 0.5) * 2

    one_hot = output > 0
    bin_enc = (one_hot.float() - 0.5) * 2

    return (torch.matmul(bin_enc.transpose(-1, -2), torch.matmul(adj, bin_enc)).diagonal(dim1=-1, dim2=-2).sum(-1) / deg_vect).mean()