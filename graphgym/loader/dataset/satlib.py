import os
import os.path as osp
from torch.multiprocessing import cpu_count
from pathlib import Path
import shutil
from loguru import logger
import csv

import torch
from tqdm import tqdm
from torch_geometric.data import (InMemoryDataset, Data, download_url,
                                  extract_gz)
from torch_geometric.utils import add_self_loops, from_networkx
from torch_geometric.graphgym.config import cfg
import numpy as np
import networkx as nx
from pysat.formula import CNF

from graphgym.utils import parallelize_fn_tqdm, fun_pbar


class SATLIB(InMemoryDataset):
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

    def __init__(self, root, transform=None, pre_transform=None,
                 pre_filter=None):
        self.name = 'default'
        self.multiprocessing = cfg.dataset.multiprocessing
        if self.multiprocessing:
            self.num_workers = cfg.num_workers if cfg.num_workers > 0 else cpu_count()
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    def download(self):
        urls = []
        for clause in [403, 411, 418, 423, 429, 435, 441, 449]:
            for bsize in [10, 30, 50, 70, 90]:
                urls.append(f'https://www.cs.ubc.ca/~hoos/SATLIB/Benchmarks/SAT/CBS/CBS_k3_n100_m{clause}_b{bsize}.tar.gz')

        paths = [download_url(u, self.cbs_dir) for u in urls]
        for p in paths:
            extract_gz(p, self.raw_dir)

    @property
    def cbs_dir(self) -> str:
        return osp.join(self.root, 'SATLIB', 'cbs')

    @property
    def raw_dir(self) -> str:
        return osp.join(self.root, 'SATLIB', 'raw')

    @property
    def processed_dir(self) -> str:
        return osp.join(self.root, 'SATLIB', 'processed')

    @property
    def raw_file_names(self):
        fnames = list()
        with open(osp.join(self.root, 'SATLIB', 'files.csv'), newline='') as f:
            for row in csv.reader(f):
                fnames.append(row[0])
        return fnames

    @property
    def processed_file_names(self):
        return ['data.pt']

    def build_graph(self, cnf_file):
        cnf = CNF(cnf_file)
        nv = cnf.nv
        clauses = list(filter(lambda x: x, cnf.clauses))
        ind = {k: [] for k in np.concatenate([np.arange(1, nv + 1), -np.arange(1, nv + 1)])}
        edges = []
        for i, clause in enumerate(clauses):
            a = clause[0]
            b = clause[1]
            c = clause[2]
            aa = 3 * i + 0
            bb = 3 * i + 1
            cc = 3 * i + 2
            ind[a].append(aa)
            ind[b].append(bb)
            ind[c].append(cc)
            edges.append((aa, bb))
            edges.append((aa, cc))
            edges.append((bb, cc))

        for i in np.arange(1, nv + 1):
            for u in ind[i]:
                for v in ind[-i]:
                    edges.append((u, v))

        G = nx.from_edgelist(edges)

        if cfg.satlib.gen_labels:
            mis = self._call_gurobi_solver(G)
            label_mapping = {vertex: int(vertex in mis) for vertex in G.nodes}
            nx.set_node_attributes(G, values=label_mapping, name='label')

        if cfg.satlib.weighted:
            weight_mapping = {vertex: weight for vertex, weight in
                              zip(G.nodes, self.random_weight(G.number_of_nodes()))}
            nx.set_node_attributes(G, values=weight_mapping, name='weight')

        g_pyg = from_networkx(G)
        return g_pyg

    def process(self):
        logger.info("Generating graphs...")
        path_list = list(Path(self.raw_dir).rglob("*.cnf"))
        if self.multiprocessing:
            logger.info(f"   num_processes={self.num_workers}")
            data_list = parallelize_fn_tqdm(path_list, self.build_graph, num_processes=self.num_workers)
        else:
            pbar = tqdm(total=len(list(path_list)))
            pbar.set_description(f'Graph generation')
            data_list = [fun_pbar(self.build_graph, f, pbar) for f in Path(self.raw_dir).rglob("*.cnf")]

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
