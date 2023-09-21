from typing import Union, Tuple, List, Dict, Any

import torch

import numpy as np
import networkx as nx
import dwave_networkx as dnx
import dimod

from utils import get_gcn_matrix


def generate_dataset(
    name: str,
    data_kwargs: Dict[str, Any],
    feat_kwargs: Dict[str, Any]
) -> Tuple[Union[Dict[str, List], List]]:


    samples = []
    for _ in range(data_kwargs["num_samples"]):

        if name == 'maxcut':
            samples.append(generate_maxcut_sample(data_kwargs, feat_kwargs))

        else:
            raise NotImplementedError("Unknown dataset name.")

    return samples


def generate_maxcut_sample(
    data_kwargs: Dict[str, Any],
    feat_kwargs: Dict[str, Any]
) -> Dict[str, Any]:
    
    # Sample graph
    sampler = dimod.SimulatedAnnealingSampler()

    n = np.random.randint(data_kwargs["n_min"], data_kwargs["n_max"]+1)
    g = nx.fast_gnp_random_graph(n, p=data_kwargs["p"])
    while not nx.is_connected(g):
        g = nx.fast_gnp_random_graph(n, p=data_kwargs["p"])

    # Derive adjacency matrix and target (cut)
    adj = torch.from_numpy(nx.to_numpy_array(g))
    num_nodes = adj.size(0)
    cut = dnx.maximum_cut(g, sampler)
    cut_size = max(len(cut), n - len(cut))
    cut_onehot = torch.zeros((num_nodes, 1), dtype=torch.int)
    cut_onehot[torch.tensor(list(cut))] = 1

    sample ={
        "adj": adj,
        "num_nodes": num_nodes,
        "target": cut_size,
        "cut_onehot": cut_onehot,
    }

    # Compute support matrices
    for type in data_kwargs["supp_matrices"]:
        sample.update(generate_supp_matrix(adj, type))

    for this_name, this_kwargs in feat_kwargs.items():
        sample.update(generate_features(this_name, g, adj, this_kwargs))

    return sample


def generate_supp_matrix(
    adj: torch.Tensor,
    type: str
) -> Dict[str, torch.Tensor]:
    
    if type == "gcn":
        supp_matrix = get_gcn_matrix(adj, sparse=False)
    
    else:
        raise NotImplementedError("Unknown support matrix type.")
    
    return {type: supp_matrix}


def generate_features(
    name: str,
    graph: nx.Graph,
    adj: torch.Tensor,
    kwargs: Dict[str, Any]
) -> Dict[str, torch.Tensor]:
    
    type = kwargs["type"]
    out_level = kwargs["level"]

    if type == 'deg':
        feat, in_level = compute_degrees(adj, kwargs['log_transform'])

    elif type == 'const':
        feat, in_level = set_constant_feat(adj, kwargs['norm'])

    else:
        raise NotImplementedError("Unknown node feature type.")
    
    feat, tag = transfer_feat_level(feat, in_level, out_level)

    return {tag + name: feat}


def compute_degrees(
    adj: torch.Tensor,
    log_transform: bool = True
) -> Tuple[List[torch.Tensor], str]:
    """
    Compute node degrees.

    Parameters:
        
    Returns:

    """

    base_level = 'node'

    feat = adj.sum(1).unsqueeze(-1)
    if log_transform:
        feat = torch.log(feat)

    return feat, base_level


def set_constant_feat(
    adj: torch.Tensor,
    norm: bool = True
) -> Tuple[List[torch.Tensor], str]:
    """
    Compute node degrees.

    Parameters:
        
    Returns:

    """

    base_level = 'node'

    feat = torch.ones(adj.size(0)).unsqueeze(-1)
    if norm:
        feat /= adj.size(0)

    return feat, base_level


def transfer_feat_level(
    feat: torch.tensor, in_level: str, out_level: str
) -> List[torch.Tensor]:
    
    if in_level == "node":
        if out_level == "node":
            tag = "node_"
        else:
            raise NotImplementedError()
    
    else:
        raise NotImplementedError()

    return feat, tag