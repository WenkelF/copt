from typing import Optional, Callable, List

import os.path as osp

import numpy as np
import networkx as nx
import torch
from torch_geometric.data import InMemoryDataset
from torch_geometric.graphgym.config import cfg
from torch_geometric.utils.convert import from_networkx
from networkx.algorithms import bipartite

from graphgym.utils import parallelize_fn


class BPDataset(InMemoryDataset):
    def __init__(self, format, root, transform=None, pre_transform=None):
        self.format = format
        super().__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])
        self.name = ''

    @property
    def processed_dir(self) -> str:
        return osp.join(self.root, 'processed')

    @property
    def processed_file_names(self):
        return ['data.pt']

    def create_graph(self):
        part_sizes = np.random.poisson(cfg[self.format].mean, 2)
        part_sizes = np.maximum(np.minimum(part_sizes, cfg[self.format].n_max), cfg[self.format].n_min)
        g_bp = bipartite.random_graph(*part_sizes, cfg[self.format].p_edge_bp)
        while not nx.is_connected(g_bp):
            g_bp = bipartite.random_graph(*np.random.poisson(cfg[self.format].mean, 2), cfg[self.format].p_edge_bp)
        
        num_nodes = len(g_bp.nodes)
        g_er = nx.erdos_renyi_graph(num_nodes, cfg[self.format].p_edge_er)

        g = nx.compose(g_bp, g_er)

        if isinstance(g, nx.DiGraph):
            g = g.to_undirected()

        g_pyg = from_networkx(g)
        return g_pyg

    def process(self):
        # Read data into huge `Data` list.
        t0 = time.time()
        if self.multiprocessing:
            data_list = parallelize_fn(range(cfg[self.format].num_samples), self.create_graph, num_processes=cfg.dataset.num_workers)
        else:
            data_list = [self.create_graph(idx) for idx in range(cfg[self.format].num_samples)]

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            if self.multiprocessing:
                data_list = parallelize_fn(data_list, self.pre_transform, num_processes=cfg.dataset.num_workers)
            else:
                data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
