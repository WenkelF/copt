from torch.multiprocessing import cpu_count
from typing import Optional, Callable, List

import os.path as osp
from loguru import logger

import torch
from torch_geometric.data import InMemoryDataset
from torch_geometric.graphgym.config import cfg
from tqdm import tqdm

from graphgym.utils import parallelize_fn, parallelize_fn_tqdm, fun_pbar


class SyntheticDataset(InMemoryDataset):
    def __init__(self, format, name, root, transform=None, pre_transform=None):
        self.format_cfg = cfg[format]
        self.name = name
        self.params = getattr(self.format_cfg, self.name)
        self.multiprocessing = cfg.dataset.multiprocessing
        if self.multiprocessing:
            self.num_workers = cfg.num_workers if cfg.num_workers > 0 else cpu_count()
        super().__init__(osp.join(root, format), transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def processed_dir(self) -> str:
        return osp.join(self.root, self.name, 'processed')

    @property
    def processed_file_names(self):
        return ['data.pt']

    def create_graph(self, idx):
        raise NotImplementedError

    def process(self):
        # Read data into huge `Data` list.

        logger.info("Generating graphs...")
        if self.multiprocessing:
            logger.info(f"   num_processes={self.num_workers}")
            data_list = parallelize_fn_tqdm(range(self.format_cfg.num_samples), self.create_graph,
                                            num_processes=self.num_workers)
        else:
            pbar = tqdm(total=self.format_cfg.num_samples)
            pbar.set_description(f'Graph generation')
            data_list = [fun_pbar(self.create_graph, idx, pbar) for idx in range(self.format_cfg.num_samples)]

        logger.info("Filtering data...")
        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        logger.info("pre transform data...")
        if self.pre_transform is not None:
            if self.multiprocessing:
                logger.info(f"   num_processes={self.num_workers}")
                data_list = parallelize_fn_tqdm(data_list, self.pre_transform, num_processes=self.num_workers)
            else:
                pbar_pre = tqdm(total=self.format_cfg.num_samples)
                pbar_pre.set_description('Graph pre-transform')
                data_list = [fun_pbar(self.pre_transform, data, pbar_pre) for data in data_list]

        logger.info("Saving data...")
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
