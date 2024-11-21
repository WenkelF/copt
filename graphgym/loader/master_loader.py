import logging
import os
import os.path as osp
import pickle
import time
import zipfile
from copy import deepcopy
from functools import partial
from typing import List, Optional
from urllib.parse import urljoin

import numpy as np
import networkx as nx
import dwave_networkx as dnx
import dimod
import requests
import torch
import torch_geometric.transforms as T
from numpy.random import default_rng
from torch_geometric.datasets import (ZINC, GNNBenchmarkDataset, Planetoid,
                                      TUDataset, WikipediaNetwork, SNAPDataset)
from torch_geometric.graphgym.config import cfg, set_cfg
from torch_geometric.graphgym.loader import (load_ogb, load_pyg,
                                             set_dataset_attr)
from torch_geometric.graphgym.model_builder import GraphGymModule
from torch_geometric.graphgym.register import register_loader
from torch_geometric.loader import DataLoader
from torch_geometric.utils import from_smiles, to_networkx
from tqdm import tqdm

from graphgym.loader.dataset.er_dataset import ERDataset
from graphgym.loader.dataset.bp_dataset import BPDataset
from graphgym.loader.dataset.rb_dataset import RBDataset
from graphgym.loader.dataset.pc_dataset import PCDataset
from graphgym.loader.dataset.ba_dataset import BADataset
from graphgym.loader.dataset.synthetic_wl import SyntheticWL
from graphgym.loader.dataset.satlib import SATLIB
from graphgym.loader.dataset.gset import Gset
from graphgym.loader.split_generator import prepare_splits, set_dataset_splits
from graphgym.transform.gnn_hash import GraphNormalizer, RandomGNNHash
from graphgym.transform.posenc_stats import compute_posenc_stats
from graphgym.transform.transforms import (VirtualNodePatchSingleton,
                                           clip_graphs_to_size,
                                           concat_x_and_pos,
                                           pre_transform_in_memory, typecast_x)
from graphgym.utils import get_device
from graphgym.wl_dataset import ToyWLDataset

from modules.data.data_generation import compute_degrees, compute_eccentricity, compute_triangle_count, compute_cluster_coefficient


def log_loaded_dataset(dataset, format, name):
    logging.info(f"[*] Loaded dataset '{name}' from '{format}':")
    logging.info(f"  {dataset.data}")
    logging.info(f"  undirected: {dataset[0].is_undirected()}")
    logging.info(f"  num graphs: {len(dataset)}")

    total_num_nodes = 0
    if hasattr(dataset.data, 'num_nodes'):
        total_num_nodes = dataset.data.num_nodes
    elif hasattr(dataset.data, 'x'):
        total_num_nodes = dataset.data.x.size(0)
    logging.info(f"  avg num_nodes/graph: "
                 f"{total_num_nodes // len(dataset)}")
    logging.info(f"  num node features: {dataset.num_node_features}")
    logging.info(f"  num edge features: {dataset.num_edge_features}")
    if hasattr(dataset, 'num_tasks'):
        logging.info(f"  num tasks: {dataset.num_tasks}")

    if hasattr(dataset.data, 'y') and dataset.data.y is not None:
        if dataset.data.y.numel() == dataset.data.y.size(0) and \
                torch.is_floating_point(dataset.data.y):
            logging.info(f"  num classes: (appears to be a regression task)")
        else:
            logging.info(f"  num classes: {dataset.num_classes}")
    elif hasattr(dataset.data, 'train_edge_label') or hasattr(dataset.data, 'edge_label'):
        # Edge/link prediction task.
        if hasattr(dataset.data, 'train_edge_label'):
            labels = dataset.data.train_edge_label  # Transductive link task
        else:
            labels = dataset.data.edge_label  # Inductive link task
        if labels.numel() == labels.size(0) and \
                torch.is_floating_point(labels):
            logging.info(f"  num edge classes: (probably a regression task)")
        else:
            logging.info(f"  num edge classes: {len(torch.unique(labels))}")

    ## Show distribution of graph sizes.
    # graph_sizes = [d.num_nodes if hasattr(d, 'num_nodes') else d.x.shape[0]
    #                for d in dataset]
    # hist, bin_edges = np.histogram(np.array(graph_sizes), bins=10)
    # logging.info(f'   Graph size distribution:')
    # logging.info(f'     mean: {np.mean(graph_sizes)}')
    # for i, (start, end) in enumerate(zip(bin_edges[:-1], bin_edges[1:])):
    #     logging.info(
    #         f'     bin {i}: [{start:.2f}, {end:.2f}]: '
    #         f'{hist[i]} ({hist[i] / hist.sum() * 100:.2f}%)'
    #     )


@register_loader('custom_master_loader')
def load_dataset_master(format, name, dataset_dir):
    """
    Master loader that controls loading of all datasets, overshadowing execution
    of any default GraphGym dataset loader. Default GraphGym dataset loader are
    instead called from this function, the format keywords `PyG` and `OGB` are
    reserved for these default GraphGym loaders.

    Custom transforms and dataset splitting is applied to each loaded dataset.

    Args:
        format: dataset format name that identifies Dataset class
        name: dataset name to select from the class identified by `format`
        dataset_dir: path where to store the processed dataset

    Returns:
        PyG dataset object with applied perturbation transforms and data splits
    """
    if format.startswith('PyG-'):
        tf_list = []
        if cfg.posenc_GraphStats.enable:
            tf_list.append(compute_graph_stats)
        if cfg.train.task == 'maxcut':
            tf_list.append(set_maxcut)
        elif cfg.train.task == 'maxclique':
            if cfg.dataset.name not in ['IMDB-BINARY', 'COLLAB', 'ego-twitter']:
                tf_list.append(set_maxclique)
        tf_list.append(set_y)

        pyg_dataset_id = format.split('-', 1)[1]
        dataset_dir = osp.join(dataset_dir, pyg_dataset_id)

        if pyg_dataset_id == 'GNNBenchmarkDataset':
            dataset = preformat_GNNBenchmarkDataset(dataset_dir, name)

        elif pyg_dataset_id == 'Planetoid':
            dataset = Planetoid(dataset_dir, name)

        elif pyg_dataset_id == 'TUDataset':
            dataset = preformat_TUDataset(dataset_dir, name)

        elif pyg_dataset_id == 'SNAPDataset':
            dataset = preformat_SNAPDataset(dataset_dir, name)  # "./datasets/snap/twitter", "ego-twitter"

        elif pyg_dataset_id == 'WikipediaNetwork':
            if name == 'crocodile':
                raise NotImplementedError("crocodile not implemented yet")
            dataset = WikipediaNetwork(dataset_dir, name)

        elif pyg_dataset_id == 'ZINC':
            dataset = preformat_ZINC(dataset_dir, name)

        else:
            raise ValueError(f"Unexpected PyG Dataset identifier: {format}")

        print(tf_list)
        pre_transform_in_memory(dataset, T.Compose(tf_list), show_progress=True)

    elif format in ['er', 'bp', 'rb', 'pc', 'ba']:
        # TODO: eccentricity fails on RB because graph may be disconnected
        if not cfg.dataset.label or cfg.train.task == 'plantedclique':
            pre_tf_list = []
        else:
            pre_tf_list = [set_maxcut, set_maxclique]
        tf_list = [T.Constant(), set_y]

        if cfg.posenc_GraphStats.enable:
            pre_tf_list.append(compute_graph_stats)

        if format.startswith('er'):
            dataset = ERDataset(name, dataset_dir, pre_transform=T.Compose(pre_tf_list))
        elif format.startswith('bp'):
            dataset = BPDataset(name, dataset_dir, pre_transform=T.Compose(pre_tf_list))
        elif format.startswith('rb'):
            dataset = RBDataset(name, dataset_dir, pre_transform=T.Compose(pre_tf_list))
        elif format.startswith('pc'):
            dataset = PCDataset(name, dataset_dir, pre_transform=T.Compose(pre_tf_list))
        elif format.startswith('ba'):
            dataset = BADataset(name, dataset_dir, pre_transform=T.Compose(pre_tf_list))

        pre_transform_in_memory(dataset, T.Compose(tf_list), show_progress=True)

    elif format == 'SATLIB' or format == 'Gset':
        if not cfg.dataset.label or cfg.train.task == 'plantedclique':
            pre_tf_list = []
        else:
            pre_tf_list = [set_maxcut, set_maxclique]
        tf_list = [T.Constant(), set_y]

        if cfg.posenc_GraphStats.enable:
            pre_tf_list.append(compute_graph_stats)

        if format == 'SATLIB':
            dataset = SATLIB(dataset_dir, pre_transform=T.Compose(pre_tf_list))
        if format == 'Gset':
            dataset = Gset(name, dataset_dir, pre_transform=T.Compose(pre_tf_list))

        pre_transform_in_memory(dataset, T.Compose(tf_list), show_progress=True)

    # GraphGym default loader for Pytorch Geometric datasets
    elif format == 'PyG':
        dataset = load_pyg(name, dataset_dir)

    elif format == 'SyntheticWL':
        dataset = preformat_SyntheticWL(dataset_dir, name=name)

    elif format == 'ToyWL':
        dataset = preformat_ToyWL(dataset_dir, name=name)

    else:
        raise ValueError(f"Unknown data format: {format}")
    log_loaded_dataset(dataset, format, name)

    # Preprocess for reducing the molecular dataset to unique structured graphs
    if cfg.dataset.unique_mol_graphs:
        dataset = get_unique_mol_graphs_via_smiles(dataset,
                                                   cfg.dataset.umg_train_ratio,
                                                   cfg.dataset.umg_val_ratio,
                                                   cfg.dataset.umg_test_ratio,
                                                   cfg.dataset.umg_random_seed)

    # Precompute necessary statistics for positional encodings.
    pe_enabled_list = []
    for key, pecfg in cfg.items():
        if (key.startswith(('posenc_', 'graphenc_')) and pecfg.enable):
            pe_name = key.split('_', 1)[1]
            pe_enabled_list.append(pe_name)
            if hasattr(pecfg, 'kernel'):
                # Generate kernel times if functional snippet is set.
                if pecfg.kernel.times_func:
                    pecfg.kernel.times = list(eval(pecfg.kernel.times_func))
                logging.info(f"Parsed {pe_name} kernel times / steps: "
                             f"{pecfg.kernel.times}")
    if pe_enabled_list:
        start = time.perf_counter()
        logging.info(f"Precomputing Positional Encoding statistics: "
                     f"{pe_enabled_list} for all graphs...")
        # Estimate directedness based on 10 graphs to save time.
        is_undirected = all(d.is_undirected() for d in dataset[:10])
        logging.info(f"  ...estimated to be undirected: {is_undirected}")
        pre_transform_in_memory(dataset,
                                partial(compute_posenc_stats,
                                        pe_types=pe_enabled_list,
                                        is_undirected=is_undirected,
                                        cfg=cfg),
                                show_progress=True)
        if hasattr(dataset.data, "y") and len(dataset.data.y.shape) == 2:
            cfg.share.num_node_targets = dataset.data.y.shape[1]
        if hasattr(dataset.data, "y_graph"):
            cfg.share.num_graph_targets = dataset.data.y_graph.shape[1]
        elapsed = time.perf_counter() - start
        timestr = time.strftime('%H:%M:%S', time.gmtime(elapsed)) \
                  + f'{elapsed:.2f}'[-3:]
        logging.info(f"Done! Took {timestr}")

    if cfg.hash_feat.enable:  # TODO: Improve handling here
        try:
            pre_transform_in_memory(dataset, RandomGNNHash(), show_progress=True)
        except:
            logging.info("Hashing to be computed later")

    if cfg.graph_norm.enable:
        pre_transform_in_memory(dataset, GraphNormalizer(), show_progress=True)

    dataset.transform_list = None
    randse_enabled_list = []
    for key, pecfg in cfg.items():
        if key.startswith('randenc_') and pecfg.enable:
            pe_name = key.split('_', 1)[1]
            randse_enabled_list.append(pe_name)
    if randse_enabled_list:
        set_random_enc(dataset, randse_enabled_list)

    if cfg.virtual_node:
        set_virtual_node(dataset)

    if dataset.transform_list is not None:
        dataset.transform = T.Compose(dataset.transform_list)

    # Set standard dataset train/val/test splits
    if hasattr(dataset, 'split_idxs'):
        set_dataset_splits(dataset, dataset.split_idxs)
        delattr(dataset, 'split_idxs')

    # Verify or generate dataset train/val/test splits
    prepare_splits(dataset)

    # Precompute GraphLog embeddings if it is enabled
    if cfg.posenc_GraphLog.enable:
        from graphgym.encoder.graphlog_encoder import precompute_graphlog
        precompute_graphlog(cfg, dataset)

    logging.info(f"Finished processing data:\n  {dataset.data}")

    return dataset


def preformat_GNNBenchmarkDataset(dataset_dir, name):
    """Load and preformat datasets from PyG's GNNBenchmarkDataset.

    Args:
        dataset_dir: path where to store the cached dataset
        name: name of the specific dataset in the TUDataset class

    Returns:
        PyG dataset object
    """
    tf_list = []
    if name in ['MNIST', 'CIFAR10']:
        tf_list = [concat_x_and_pos]  # concat pixel value and pos. coordinate
        tf_list.append(partial(typecast_x, type_str='float'))
    elif name == "CSL":
        # CSL does have predefined split. Need to use cv or random splits.
        dataset = GNNBenchmarkDataset(root=dataset_dir, name=name,
                                      split="train")
        pre_transform_in_memory(dataset, T.Constant(cat=False))
        return dataset
    else:
        ValueError(f"Loading dataset '{name}' from "
                   f"GNNBenchmarkDataset is not supported.")

    dataset = join_dataset_splits(
        [GNNBenchmarkDataset(root=dataset_dir, name=name, split=split)
         for split in ['train', 'val', 'test']]
    )
    pre_transform_in_memory(dataset, T.Compose(tf_list))

    return dataset


def preformat_SyntheticWL(dataset_dir, name):
    """Load and preformat synthetic WL graph datasets.

    Args:
        dataset_dir: path where to store the cached dataset.
        name: name of the specific dataset in the SyntheticWL collection.
            Available options are: 'exp', 'cexp', and 'sr25'.

    Returns:
        PyG dataset object

    """
    dataset = SyntheticWL(dataset_dir, name=name)
    if name.lower() == "sr25":
        # Evaluate on training, so train/val/test are the same split
        dataset = join_dataset_splits([deepcopy(dataset) for _ in range(3)])
    return dataset


def preformat_ToyWL(dataset_dir, name=None):
    """Load and preformat toy WL graph datasets.

    Args:
        dataset_dir: path where to store the cached dataset.
        name: name of the specific dataset in the SyntheticWL collection.
            Available options are: 'exp', 'cexp', and 'sr25'.

    Returns:
        PyG dataset object

    """
    dataset = ToyWLDataset(dataset_dir, name)
    dataset = join_dataset_splits([deepcopy(dataset) for _ in range(3)])
    return dataset


def preformat_TUDataset(dataset_dir, name):
    """Load and preformat datasets from PyG's TUDataset.

    Args:
        dataset_dir: path where to store the cached dataset
        name: name of the specific dataset in the TUDataset class

    Returns:
        PyG dataset object
    """
    if name in ['DD', 'NCI1', 'ENZYMES', 'PROTEINS']:
        func = None
    elif name.startswith('IMDB-') or name == "COLLAB":
        func = T.Constant()
    else:
        ValueError(f"Loading dataset '{name}' from TUDataset is not supported.")
    dataset = TUDataset(dataset_dir, name, pre_transform=func)
    if name in ['IMDB-BINARY', 'COLLAB', 'ego-twitter']:
        with open("data/maxclique/" + cfg.dataset.name + "cliqno.txt", "rb") as fp:
            dataset.data.y = torch.Tensor(pickle.load(fp)).to(int)
    return dataset

def set_placeholder_y(data):
    data.y = 1
    return data

def preformat_SNAPDataset(dataset_dir, name):
    """Load and preformat datasets from PyG's SNAPDataset.

    Args:
        dataset_dir: path where to store the cached dataset
        name: name of the specific dataset in the TUDataset class

    Returns:
        PyG dataset object
    """

    dataset = SNAPDataset(dataset_dir, name, pre_transform=set_placeholder_y)
    if name in ['ego-twitter']:
        with open("data/maxclique/" + cfg.dataset.name + "cliqno.txt", "rb") as fp:
            dataset.data.y = torch.Tensor(pickle.load(fp)).to(int)
    return dataset


def preformat_ZINC(dataset_dir, name):
    """Load and preformat ZINC datasets.

    Args:
        dataset_dir: path where to store the cached dataset
        name: select 'subset' or 'full' version of ZINC

    Returns:
        PyG dataset object
    """
    if name not in ['subset', 'full']:
        raise ValueError(f"Unexpected subset choice for ZINC dataset: {name}")
    dataset = join_dataset_splits(
        [ZINC(root=dataset_dir, subset=(name == 'subset'), split=split)
         for split in ['train', 'val', 'test']]
    )
    return dataset

def join_dataset_splits(datasets):
    """Join train, val, test datasets into one dataset object.

    Args:
        datasets: list of 3 PyG datasets to merge

    Returns:
        joint dataset with `split_idxs` property storing the split indices
    """
    assert len(datasets) == 3, "Expecting train, val, test datasets"

    n1, n2, n3 = len(datasets[0]), len(datasets[1]), len(datasets[2])
    data_list = [datasets[0].get(i) for i in range(n1)] + \
                [datasets[1].get(i) for i in range(n2)] + \
                [datasets[2].get(i) for i in range(n3)]

    datasets[0]._indices = None
    datasets[0]._data_list = data_list
    datasets[0].data, datasets[0].slices = datasets[0].collate(data_list)
    split_idxs = [list(range(n1)),
                  list(range(n1, n1 + n2)),
                  list(range(n1 + n2, n1 + n2 + n3))]
    datasets[0].split_idxs = split_idxs

    return datasets[0]


def smiles_from_graph(
    node_list: List[str],
    adjacency_matrix: np.ndarray,
) -> str:
    """Create a SMILES string from a given graph.

    Modified from https://stackoverflow.com/a/51242251/12519564

    """
    try:
        from rdkit import Chem
    except ModuleNotFoundError:
        raise ModuleNotFoundError("rdkit is not installed yet")

    # Create empty editable mol object
    mol = Chem.RWMol()

    # Add atoms to mol and keep track of index
    node_to_idx = {}
    for i in range(len(node_list)):
        a = Chem.Atom(node_list[i])
        molIdx = mol.AddAtom(a)
        node_to_idx[i] = molIdx

    # Add bonds between adjacent atoms
    for i, j in zip(*np.nonzero(adjacency_matrix)):
        # Only traverse half the matrix
        if j <= i:
            continue

        if adjacency_matrix[i, j] >= 1:
            bond_type = Chem.rdchem.BondType.SINGLE
            mol.AddBond(node_to_idx[i], node_to_idx[j], bond_type)

    # Convert RWMol to Mol object
    mol = mol.GetMol()
    smiles = Chem.MolToSmiles(mol)

    return smiles


def get_unique_mol_graphs_via_smiles(
    dataset,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    random_seed: int = 0,
):
    if (sum_ratio := train_ratio + val_ratio + test_ratio) > 1:
        raise ValueError("Total ratio (train + val + test) must be below 1 "
                         f"got {sum_ratio:.2f}")

    old_size = len(dataset)
    all_smiles = []
    for i in tqdm(dataset, total=old_size,
                  desc="Extracting unique graphs (ignoring atom/bond types)"):
        num_nodes = i.num_nodes
        trivial_c_atoms = ["C"] * num_nodes
        adj = torch.sparse_coo_tensor(
            i.edge_index,
            torch.ones(i.edge_index.size(1), dtype=torch.float),
            size=(num_nodes, num_nodes),
        ).to_dense().numpy()
        all_smiles.append(smiles_from_graph(trivial_c_atoms, adj))
    unique_smiles = sorted(set(all_smiles))

    unique_graphs = []
    for smiles in unique_smiles:
        g = from_smiles(smiles)
        if (g.num_nodes > 1) and (g.edge_index.shape[1] > 1):
            delattr(g, "smiles")
            delattr(g, "edge_attr")
            unique_graphs.append(g)

    num_unique = len(unique_graphs)
    split_points = [int(num_unique * train_ratio),
                    int(num_unique * (1 - val_ratio - test_ratio)),
                    int(num_unique * (1 - test_ratio))]
    rng = np.random.default_rng(random_seed)
    new_split_idxs = np.split(rng.permutation(num_unique), split_points)
    new_split_idxs.pop(1)  # pop the fill-in split
    # Reorder graphs into train/val/test (poentially remove the fill-in split)
    unique_graphs = [unique_graphs[i] for i in np.hstack(new_split_idxs)]
    new_size = len(unique_graphs)

    if test_ratio == 1:
        # Evaluation only, pad "training" and "evaluation" set with the first
        # graph
        new_split_idxs[0] = np.array([num_unique])
        new_split_idxs[1] = np.array([num_unique + 1])
        unique_graphs.append(unique_graphs[-1])
        unique_graphs.append(unique_graphs[-1])

    # E.g. [[0, 1], [0, 1, 2], [0]]
    dataset.split_idxs = [torch.arange(idxs.size) for idxs in new_split_idxs]
    if train_ratio != 1:
        # E.g. [[0, 1], [2, 3, 4], [5]]
        for i in range(1, len(dataset.split_idxs)):
            dataset.split_idxs[i] += dataset.split_idxs[i - 1][-1] + 1

    dataset.data, dataset.slices = dataset.collate(unique_graphs)
    # We need to remove _data_list because its presence will bypass the
    # indentded data slicing using the .slices attribute.
    # https://github.com/pyg-team/pytorch_geometric/blob/f0c72186286f257778c1d9293cfd0d35472d30bb/torch_geometric/data/in_memory_dataset.py#L75-L94
    dataset._data_list = [None] * len(dataset)
    dataset._indices = None

    logging.info("[*] Dataset reduced to unique molecular structure graphs\n"
                 f"    Number of graphs before: {old_size:,}\n"
                 f"    Number of graphs after: {new_size:,}\n"
                 f"    Train size: {len(new_split_idxs[0]):,} "
                 f"(first five: {new_split_idxs[0][:5]})\n"
                 f"    Validation size: {len(new_split_idxs[1]):,} "
                 f"(first five: {new_split_idxs[1][:5]})\n"
                 f"    Test size: {len(new_split_idxs[2]):,} "
                 f"(first five: {new_split_idxs[2][:5]})\n"
                 f"    {dataset.data}\n")

    return dataset


def set_random_enc(dataset, pe_types):

    if 'FixedSE' in pe_types:
        def randomSE_Fixed(data):
            N = data.num_nodes
            stat = np.full(shape=(N, cfg.randenc_FixedSE.dim_pe), fill_value=1).astype('float32')
            data.x = torch.from_numpy(stat)
            return data

        dataset.transform_list = [randomSE_Fixed]

    if 'NormalSE' in pe_types:
        def randomSE_Normal(data):
            N = data.num_nodes
            rand = np.random.normal(loc=0, scale=1.0, size=(N, cfg.randenc_NormalSE.dim_pe)).astype('float32')
            data.x = torch.from_numpy(rand)
            return data

        dataset.transform_list = [randomSE_Normal]

    if 'UniformSE' in pe_types:
        def randomSE_Uniform(data):
            N = data.num_nodes
            rand = np.random.uniform(low=0.0, high=1.0, size=(N, cfg.randenc_UniformSE.dim_pe)).astype('float32')
            data.x = torch.from_numpy(rand)
            return data

        dataset.transform_list = [randomSE_Uniform]

    if 'BernoulliSE' in pe_types:
        def randomSE_Bernoulli(data):
            N = data.num_nodes
            rand = np.random.uniform(low=0.0, high=1.0, size=(N, cfg.randenc_BernoulliSE.dim_pe))
            rand = (rand < cfg.randenc_BernoulliSE.threshold).astype('float32')
            data.x = torch.from_numpy(rand)
            return data

        dataset.transform_list = [randomSE_Bernoulli]

    if 'DiracRE' in pe_types:
        def randomRE_Dirac(data):
            N = data.num_nodes
            zeros = torch.zeros(N, cfg.randenc_DiracRE.dim_pe)
            rand_idx = torch.randint(low=0, high=N, size=())
            zeros[rand_idx] = 1.0
            data.x = zeros.float()
            return data

        dataset.transform_list = [randomRE_Dirac]


def set_virtual_node(dataset):
    if dataset.transform_list is None:
        dataset.transform_list = []
    dataset.transform_list.append(VirtualNodePatchSingleton())


def compute_graph_stats(data):
    g = to_networkx(data)
    if isinstance(g, nx.DiGraph):
        g = g.to_undirected()
    # Derive adjacency matrix
    adj = torch.from_numpy(nx.to_numpy_array(g))
    norm_factor = np.sqrt(g.number_of_nodes()) if cfg.gnn.gsn else 1

    if 'degree' in cfg.dataset.graph_stats:
        data.degree = compute_degrees(adj, log_transform=True)[0] / norm_factor
    if 'eccentricity' in cfg.dataset.graph_stats:
        data.eccentricity = compute_eccentricity(g)[0] / norm_factor
    if 'cluster_coefficient' in cfg.dataset.graph_stats:
        data.cluster_coefficient = compute_cluster_coefficient(g)[0] / norm_factor
    if 'triangle_count' in cfg.dataset.graph_stats:
        data.triangle_count = compute_triangle_count(g)[0] / norm_factor

    return data


def compute_maxcut(g):
    adj = torch.from_numpy(nx.to_numpy_array(g))
    num_nodes = adj.size(0)

    cut = dnx.maximum_cut(g, dimod.SimulatedAnnealingSampler())
    cut_size = max(len(cut), g.number_of_nodes() - len(cut))
    cut_binary = torch.zeros((num_nodes, 1), dtype=torch.int)
    cut_binary[torch.tensor(list(cut))] = 1

    return cut_size, cut_binary


def set_maxcut(data):
    g = to_networkx(data)
    if isinstance(g, nx.DiGraph):
        g = g.to_undirected()
    # Derive adjacency matrix
    cut_size, cut_binary = compute_maxcut(g)

    data.cut_size = cut_size
    data.cut_binary = cut_binary
    return data


def set_maxclique(data):
    g = to_networkx(data)
    if isinstance(g, nx.DiGraph):
        g = g.to_undirected()
    # target = {"mc_size": max(len(clique) for clique in nx.find_cliques(g))}
    data.mc_size = max(len(clique) for clique in nx.find_cliques(g))
    return data


def set_plantedclique(data):
    g = to_networkx(data)
    if isinstance(g, nx.DiGraph):
        g = g.to_undirected()
    # target = {"mc_size": max(len(clique) for clique in nx.find_cliques(g))}
    data.mc_size = max(len(clique) for clique in nx.find_cliques(g))
    return data


def set_y(data):
    if not cfg.dataset.label:
        data.y = torch.ones(data.num_nodes, 1)
    elif cfg.train.task == 'maxcut':
        data.y = data.cut_binary
    elif cfg.train.task == 'maxclique':
        if cfg.dataset.name not in ['IMDB-BINARY', 'COLLAB', 'ego-twitter']:
            data.y = data.mc_size
    return data