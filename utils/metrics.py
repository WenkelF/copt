import torch


def accuracy(output, target):

    return torch.mean((output.argmax(-1) == target).float())


def maxclique_loss(output, adj, beta):

    loss1 = torch.matmul(output.transpose(-1, -2), torch.matmul(adj, output))
    loss2 = output.sum() ** 2 - loss1 - torch.sum(output ** 2)

    return - loss1 + beta * loss2


def maxclique_decoder(output, adj, dec_length):

    order = torch.argsort(output)
    c = torch.zeros_like(output)
    c[order[0]] = 1

    for i in range(1, dec_length):
        c[order[i]] = 1

        cTWc = torch.matmul(c.transpose(-1, -2), torch.matmul(adj, c))
        if c.sum() ** 2 - cTWc - torch.sum(c ** 2) != 0:
            c[order[i]] = 0

    return c


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

    target = data['cut_onehot']

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