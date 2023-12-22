from typing import Union, Tuple, List, Dict, Any

import torch

import numpy as np
import networkx as nx
import dwave_networkx as dnx
import dimod

from torch_geometric.utils import from_networkx

from utils import get_gcn_matrix, get_sct_matrix, get_res_matrix


def generate_sample(
    task: str,
    data_kwargs: Dict[str, Any],
    feat_kwargs: Dict[str, Any],
    base_graph: nx.Graph = None,
    base_target: Dict[str, Any] = None,
    label: bool = False,
) -> Dict[str, Any]:
    
    g, target = base_graph, base_target
    
    if g is None:
        n = np.random.randint(data_kwargs["n_min"], data_kwargs["n_max"]+1)
        g = nx.fast_gnp_random_graph(n, p=data_kwargs["p"])
        while not nx.is_connected(g):
            g = nx.fast_gnp_random_graph(n, p=data_kwargs["p"])

    if isinstance(g, nx.DiGraph):
        g = g.to_undirected()

    # # Derive adjacency matrix
    adj = torch.from_numpy(nx.to_numpy_array(g))
    num_nodes = adj.size(0)

    sample = {
        # "adj_mat": adj,
        "num_nodes": num_nodes,
    }

    # Compute support matrices
    for type in data_kwargs["supp_matrices"]:
        if type == 'adj':
            sample.update({"adj_mat": adj})
        elif type == 'edge_index':
            sample.update({"edge_index": from_networkx(g).edge_index})
        else:
            sample.update(generate_supp_matrix(adj, type))

    # Compute node features
    for this_name, this_kwargs in feat_kwargs.items():
        sample.update(generate_features(this_name, g, adj, this_kwargs))
    
    # Compute label (if desired)
    if label:
        
        if target is not None:
            pass

        elif task == "maxcut":
            cut_size, cut_binary = compute_maxcut(g)
            target = {
                "cut_size": cut_size,  
                "cut_binary": cut_binary,
            }

        elif task == "maxclique":
            target = {"mc_size": max(len(clique) for clique in nx.find_cliques(g))}

        else:
            raise NotImplementedError("Unknown task name.")
        
        sample.update(target)

    return sample


def generate_supp_matrix(
    adj: torch.Tensor,
    type: str
) -> Dict[str, torch.Tensor]:
    
    if type == "gcn":
        supp_matrix = get_gcn_matrix(adj, sparse=False)
    elif type == "sct":
        supp_matrix = get_sct_matrix(adj, sparse=False)
    elif type == "res":
        supp_matrix = get_res_matrix(adj, sparse=False)
    else:
        raise NotImplementedError("Unknown support matrix type.")
    
    return {"".join([type, "_mat"]): supp_matrix}


def generate_features(
    name: str,
    g: nx.Graph,
    adj: torch.Tensor,
    kwargs: Dict[str, Any]
) -> Dict[str, torch.Tensor]:
    
    type = kwargs["type"]
    out_level = kwargs["level"]

    if type == 'deg':
        feat, in_level = compute_degrees(adj, kwargs['log_transform'])

    elif type == 'ecc':
        feat, in_level = compute_eccentricity(g)

    elif type == 'clu':
        feat, in_level = compute_cluster_coefficient(g)

    elif type == 'tri':
        feat, in_level = compute_triangle_count(g)

    elif type == 'const':
        feat, in_level = set_constant_feat(adj, kwargs['norm'])

    else:
        raise NotImplementedError("Unknown node feature type.")
    
    feat, tag = transfer_feat_level(feat, in_level, out_level)

    return {tag + name: feat}


def compute_maxcut(g):
    adj = torch.from_numpy(nx.to_numpy_array(g))
    num_nodes = adj.size(0)

    cut = dnx.maximum_cut(g, dimod.SimulatedAnnealingSampler())
    cut_size = max(len(cut), g.number_of_nodes() - len(cut))
    cut_binary = torch.zeros((num_nodes, 1), dtype=torch.int)
    cut_binary[torch.tensor(list(cut))] = 1

    return cut_size, cut_binary


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


def compute_eccentricity(
    graph: nx.Graph,
) -> Tuple[List[torch.Tensor], str]:
    """
    Compute node degrees.

    Parameters:
        
    Returns:

    """

    base_level = 'node'

    feat = torch.Tensor(list(nx.eccentricity(graph).values())).unsqueeze(-1)
    
    return feat, base_level


def compute_cluster_coefficient(
    graph: nx.Graph,
) -> Tuple[List[torch.Tensor], str]:
    """
    Compute node degrees.

    Parameters:
        
    Returns:

    """

    base_level = 'node'

    feat = torch.Tensor(list(nx.clustering(graph).values())).unsqueeze(-1)
    
    return feat, base_level


def compute_triangle_count(
    graph: nx.Graph,
) -> Tuple[List[torch.Tensor], str]:
    """
    Compute node degrees.

    Parameters:
        
    Returns:

    """

    base_level = 'node'

    feat = torch.Tensor(list(nx.triangles(graph).values())).unsqueeze(-1)
    
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