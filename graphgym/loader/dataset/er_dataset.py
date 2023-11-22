from typing import Optional, Callable, List

import os.path as osp
from loguru import logger

import numpy as np
import networkx as nx
import torch
from torch_geometric.data import InMemoryDataset
from torch_geometric.graphgym.config import cfg
from torch_geometric.utils.convert import from_networkx

from graphgym.utils import parallelize_fn


class ERDataset(InMemoryDataset):
    def __init__(self, format, root, transform=None, pre_transform=None, multiprocessing=False):
        self.format = format
        self.multiprocessing = multiprocessing
        super().__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])
        self.name = ''

    @property
    def processed_dir(self) -> str:
        return osp.join(self.root, 'processed')

    @property
    def processed_file_names(self):
        return ['data.pt']

    def create_graph(self, idx):
        n = np.random.randint(cfg[self.format].n_min, cfg[self.format].n_max + 1)
        g = nx.fast_gnp_random_graph(n, p=cfg[self.format].p)
        while not nx.is_connected(g):
            g = nx.fast_gnp_random_graph(n, p=cfg[self.format].p)

        if isinstance(g, nx.DiGraph):
            g = g.to_undirected()

        g_pyg = from_networkx(g)
        return g_pyg

    def process(self):
        # Read data into huge `Data` list.
        
        logger.info("Generating graphs...")
        if self.multiprocessing:
            logger.info(f"   num_processes={cfg.dataset.num_workers}")
            data_list = parallelize_fn(range(cfg[self.format].num_samples), self.create_graph, num_processes=cfg.dataset.num_workers)
        else:
            data_list = [self.create_graph(idx) for idx in range(cfg[self.format].num_samples)]

        logger.info("Filtering data...")
        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        logger.info("pre transform data...")
        if self.pre_transform is not None:
            if self.multiprocessing:
                logger.info(f"   num_processes={cfg.dataset.num_workers}")
                data_list = parallelize_fn(data_list, self.pre_transform, num_processes=cfg.dataset.num_workers)
            else:
                data_list = [self.pre_transform(data) for data in data_list]

        logger.info("Saving data...")
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
