import os
import os.path as osp
from torch.multiprocessing import cpu_count
from loguru import logger

import torch
from tqdm import tqdm
from torch_geometric.data import InMemoryDataset, download_url
from torch_geometric.utils import add_self_loops, from_networkx
from torch_geometric.graphgym.config import cfg
import networkx as nx

from graphgym.utils import parallelize_fn_tqdm, fun_pbar


class Gset(InMemoryDataset):
    """
    Args:
        root (string): Root directory where the dataset should be saved.
        transform (callable, optional): A function/transform that takes in an
            :obj:`torch_geometric.data.Data` object and returns a transformed
            version. The data object will be transformed before every access.
            (default: :obj:`None`)
        pre_transform (callable, optional): A function/transform that takes in
            an :obj:`torch_geometric.data.Data` object and returns a
            transformed version. The data object will be transformed before
            being saved to disk. (default: :obj:`None`)
        pre_filter (callable, optional): A function that takes in an
            :obj:`torch_geometric.data.Data` object and returns a boolean
            value, indicating whether the data object should be included in the
            final dataset. (default: :obj:`None`)
    """

    def __init__(self, name, root, transform=None, pre_transform=None,
                 pre_filter=None):
        self.idx_dict = {
            'default': [*range(1, 68), 70, 72, 77, 81],
            'small': [*range(1, 21)],
            '1K': [43, 44, 45, 46, 47, 51, 52, 53, 54],
            '2K': [*range(22, 42)],
            'large': [*range(55, 68), 70, 72, 77, 81]
        }
        if name not in self.idx_dict.keys():
            raise ValueError(f"Unrecognized dataset name {name!r}, available "
                             f"options are: {self.idx_dict.keys()}")
        self.name = name
        self.multiprocessing = cfg.dataset.multiprocessing
        if self.multiprocessing:
            self.num_workers = cfg.num_workers if cfg.num_workers > 0 else cpu_count()
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    def download(self):
        for gname in self.idx_dict['default']:
            download_url(f'https://web.stanford.edu/~yyye/yyye/Gset/G{gname}', self.raw_dir)

    @property
    def raw_dir(self) -> str:
        return osp.join(self.root, 'Gset', 'raw')

    @property
    def processed_dir(self) -> str:
        return osp.join(self.root, 'Gset', 'processed')

    @property
    def raw_file_names(self):
        fnames = list()
        for gname in self.idx_dict['default']:
            fnames.append(f'G{gname}')
        return fnames

    @property
    def processed_file_names(self):
        return ['data.pt']

    def build_graph(self, f):
        G = nx.Graph()

        with open(f, 'r') as file:
            next(file)
            for line in file:
                node1, node2, weight = map(int, line.split())
                G.add_edge(node1, node2, weight=weight)

        g_pyg = from_networkx(G)
        return g_pyg

    def process(self):
        logger.info("Processing graphs...")
        path_list = [os.path.join(self.raw_dir, f'G{i}') for i in self.idx_dict[self.name]]
        if self.multiprocessing:
            logger.info(f"   num_processes={self.num_workers}")
            data_list = parallelize_fn_tqdm(path_list, self.build_graph, num_processes=self.num_workers)
        else:
            pbar = tqdm(total=len(list(path_list)))
            pbar.set_description(f'Graph generation')
            data_list = [fun_pbar(self.build_graph, f, pbar) for f in path_list]

        logger.info("pre transform data...")
        if self.pre_transform is not None:
            if self.multiprocessing:
                logger.info(f"   num_processes={self.num_workers}")
                data_list = parallelize_fn_tqdm(data_list, self.pre_transform, num_processes=self.num_workers)
            else:
                pbar_pre = tqdm(total=len(data_list))
                pbar_pre.set_description(f'Graph pre-transform')
                data_list = [fun_pbar(self.pre_transform, data, pbar_pre) for data in data_list]

        logger.info("Saving data...")
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
