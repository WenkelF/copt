import torch


def get_gcn_matrix(adj, sparse=True):

    adj = adj + torch.eye(adj.size(-1)).to(adj.device)

    deg_vect = adj.sum(-1)
    norm_mat = torch.diag_embed(deg_vect ** (-0.5)).to_sparse()
    
    gcn_mat = torch.spmm(norm_mat, torch.spmm(adj.to_sparse(), norm_mat))

    if sparse:
        return gcn_mat
    else:
        return gcn_mat.to_dense()


def get_sct_matrix(adj, sparse=True):
    
    deg_vect = adj.sum(-1)
    deg_vect[deg_vect == 0] = 1.
    norm_mat = torch.diag_embed(1 / deg_vect).to_sparse()
    sct_mat = torch.spmm(adj.to_sparse(), norm_mat).to_dense()
    sct_mat = 0.5 * (sct_mat + torch.eye(sct_mat.size(-1)).to(adj.device))
    
    if sparse:
        return sct_mat.to_sparse()
    else:
        return sct_mat


def get_res_matrix(adj, alpha=0.5, sparse=True):
    
    deg_vect = adj.sum(-1)
    deg_vect[deg_vect == 0] = 1.
    norm_mat = torch.diag_embed(1 / deg_vect).to_sparse()
    supp_mat = torch.spmm(adj.to_sparse(), norm_mat).to_dense()
    res_mat = (alpha * supp_mat + torch.eye(supp_mat.size(-1)).to(adj.device)) / (alpha + 1)
    
    if sparse:
        return res_mat.to_sparse()
    else:
        return res_mat


def diffusion(x, supp_mat, num_steps):

    for _ in range(num_steps):
        x = torch.matmul(supp_mat, x)

    return x


def get_wav_matrix(supp_mat, scale):
    
    # Highpass
    if scale == 0:
        M_1 = torch.eye(supp_mat.shape[0])
        M_2 = diffusion(supp_mat.to_dense(), supp_mat, 1)
        wav = M_1 - M_2
    
    # Lowpass    
    elif scale < 0:
        num_steps =  - scale - 1
        wav = diffusion(supp_mat.to_dense(), supp_mat, num_steps)
    
    # Bandpass
    else:
        num_steps =  2 ** (scale-1)
        M_1 = diffusion(supp_mat.to_dense(), supp_mat, num_steps - 1)
        M_2 = diffusion(M_1, supp_mat, num_steps)
        
        wav = M_1 - M_2
    
    if scale >= -2 and scale <= 1:
        return wav.to_sparse()
    else:
        return wav
    

def wavelet_diffusion(x, supp_mat, scale):

    # Highpass
    if scale == 0:
        x_1 = x
        x_2 = diffusion(x, supp_mat, 1)
        out = x_1 - x_2
    
    # Lowpass    
    elif scale < 0:
        out = diffusion(x, supp_mat, -scale)

    # Bandpass    
    else:
        num_steps =  2 ** (scale-1)
        x_1 = diffusion(x, supp_mat, num_steps)
        x_2 = diffusion(x_1, supp_mat, num_steps)
        out = x_1 - x_2
    
    return out


def get_supp_matrix(adj, type='gcn', alpha=0.5, sparse=True):

    if type == 'gcn':
        supp_mat = get_gcn_matrix(adj, sparse)
    elif type == 'sct':
        supp_mat = get_sct_matrix(adj, sparse)
    elif type == 'res':
        supp_mat = get_res_matrix(adj, alpha, sparse)

    return supp_mat