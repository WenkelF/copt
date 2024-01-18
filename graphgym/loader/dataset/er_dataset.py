import numpy as np
import networkx as nx
from torch_geometric.utils.convert import from_networkx

from graphgym.loader.dataset.synthetic import SyntheticDataset


class ERDataset(SyntheticDataset):
    def __init__(self, name, root, transform=None, pre_transform=None):
        super().__init__('er', name, root, transform, pre_transform)

    def create_graph(self, idx):
        n = np.random.randint(self.params.n_min, self.params.n_max + 1)
        g = nx.fast_gnp_random_graph(n, p=self.params.p)
        while not nx.is_connected(g):
            g = nx.fast_gnp_random_graph(n, p=self.params.p)

        if isinstance(g, nx.DiGraph):
            g = g.to_undirected()

        g_pyg = from_networkx(g)
        return g_pyg
