import numpy as np
import networkx as nx
from torch_geometric.utils.convert import from_networkx
from networkx.algorithms import bipartite

from graphgym.loader.dataset.synthetic import SyntheticDataset


class BPDataset(SyntheticDataset):
    def __init__(self, name, root, transform=None, pre_transform=None):
        super().__init__('bp', name, root, transform, pre_transform)

    def create_graph(self, idx):
        part_sizes = np.random.poisson(self.params.mean, 2)
        part_sizes = np.maximum(np.minimum(part_sizes, self.format_cfg.n_max), self.format_cfg.n_min)
        g = bipartite.random_graph(*part_sizes, self.format_cfg.p_edge_bp)
        while not nx.is_connected(g):
            g = bipartite.random_graph(*np.random.poisson(self.params.mean, 2), self.format_cfg.p_edge_bp)
        
        num_nodes = len(g.nodes)
        if self.params.p_edge_er > 0:
            g_er = nx.erdos_renyi_graph(num_nodes, self.params.p_edge_er)
            g = nx.compose(g, g_er)

        if isinstance(g, nx.DiGraph):
            g = g.to_undirected()

        g_pyg = from_networkx(g)
        return g_pyg
