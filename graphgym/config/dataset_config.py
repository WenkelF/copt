from yacs.config import CfgNode as CN
from torch_geometric.graphgym.register import register_config


@register_config('dataset_cfg')
def dataset_cfg(cfg):
    """Dataset-specific config options.
    """

    # Input/output node encoder types (used to construct PE tasks)
    # Use "+" to cnocatenate multiple encoder types, e.g. "LapPE+RWSE"
    cfg.dataset.input_node_encoders = "none"
    cfg.dataset.output_node_encoders = "none"
    cfg.dataset.output_graph_encoders = "none"

    # The number of node types to expect in TypeDictNodeEncoder.
    cfg.dataset.node_encoder_num_types = 0

    # The number of edge types to expect in TypeDictEdgeEncoder.
    cfg.dataset.edge_encoder_num_types = 0

    # VOC/COCO Superpixels dataset version based on SLIC compactness parameter.
    cfg.dataset.slic_compactness = 10

    # Reduce the molecular graph dataset to only contain unique structured
    # graphs (ignoring atom and bond types)
    cfg.dataset.unique_mol_graphs = False
    cfg.dataset.umg_train_ratio = 0.8
    cfg.dataset.umg_val_ratio = 0.1
    cfg.dataset.umg_test_ratio = 0.1
    cfg.dataset.umg_random_seed = 0  # for random indexing

    cfg.dataset.set_graph_stats = False

    cfg.dataset.multiprocessing = False
    cfg.dataset.num_workers = 4

    cfg.dataset.label = True


@register_config('er_test_cfg')
def er_test_cfg(cfg):
    """Configuration options for nx datasets.
    """
    cfg.er_test = CN()
    # features can be one of ['node_const', 'node_onehot', 'node_clustering_coefficient', 'node_pagerank']
    cfg.er_test.num_samples = 100
    cfg.er_test.n_min = 8
    cfg.er_test.n_max = 15
    cfg.er_test.p = 0.4
    cfg.er_test.supp_mtx = ["edge_index"]


@register_config('er_cfg')
def er_cfg(cfg):
    """Configuration options for nx datasets.
    """
    cfg.er = CN()
    cfg.er.num_samples = 10000
    cfg.er.n_min = 8
    cfg.er.n_max = 15
    cfg.er.p = 0.4
    cfg.er.supp_mtx = ["edge_index"]


@register_config('er_50_02_cfg')
def er_50_02_cfg(cfg):
    """Configuration options for nx datasets.
    """
    cfg.er_50_02 = CN()
    cfg.er_50_02.num_samples = 10000
    cfg.er_50_02.n_min = 30
    cfg.er_50_02.n_max = 50
    cfg.er_50_02.p = 0.2
    cfg.er_50_02.supp_mtx = ["edge_index"]


@register_config('er_50_03_cfg')
def er_50_03_cfg(cfg):
    """Configuration options for nx datasets.
    """
    cfg.er_50_03 = CN()
    cfg.er_50_03.num_samples = 10000
    cfg.er_50_03.n_min = 30
    cfg.er_50_03.n_max = 50
    cfg.er_50_03.p = 0.3
    cfg.er_50_03.supp_mtx = ["edge_index"]


@register_config('er_50_04_cfg')
def er_50_04_cfg(cfg):
    """Configuration options for nx datasets.
    """
    cfg.er_50_04 = CN()
    cfg.er_50_04.num_samples = 10000
    cfg.er_50_04.n_min = 30
    cfg.er_50_04.n_max = 50
    cfg.er_50_04.p = 0.4
    cfg.er_50_04.supp_mtx = ["edge_index"]


@register_config('bp_cfg')
def bp_cfg(cfg):
    """Configuration options for nx datasets.
    """
    cfg.bp = CN()
    cfg.bp.num_samples = 10000
    cfg.bp.mean = 10
    cfg.bp.n_min = 4
    cfg.bp.n_max = 20
    cfg.bp.p_edge_bp = 0.4
    cfg.bp.p_edge_er = 0.1
    cfg.bp.supp_mtx = ["edge_index"]


@register_config('bp_20_00_cfg')
def bp_20_00_cfg(cfg):
    """Configuration options for nx datasets.
    """
    cfg.bp_20_00 = CN()
    cfg.bp_20_00.num_samples = 10000
    cfg.bp_20_00.mean = 20
    cfg.bp_20_00.n_min = 10
    cfg.bp_20_00.n_max = 30
    cfg.bp_20_00.p_edge_bp = 0.3
    cfg.bp_20_00.p_edge_er = 0.0
    cfg.bp_20_00.supp_mtx = ["edge_index"]


@register_config('bp_20_01_cfg')
def bp_20_01_cfg(cfg):
    """Configuration options for nx datasets.
    """
    cfg.bp_20_01 = CN()
    cfg.bp_20_01.num_samples = 10000
    cfg.bp_20_01.mean = 20
    cfg.bp_20_01.n_min = 10
    cfg.bp_20_01.n_max = 30
    cfg.bp_20_01.p_edge_bp = 0.3
    cfg.bp_20_01.p_edge_er = 0.1
    cfg.bp_20_01.supp_mtx = ["edge_index"]


@register_config('bp_20_02_cfg')
def bp_20_02_cfg(cfg):
    """Configuration options for nx datasets.
    """
    cfg.bp_20_02 = CN()
    cfg.bp_20_02.num_samples = 10000
    cfg.bp_20_02.mean = 20
    cfg.bp_20_02.n_min = 10
    cfg.bp_20_02.n_max = 30
    cfg.bp_20_02.p_edge_bp = 0.3
    cfg.bp_20_02.p_edge_er = 0.2
    cfg.bp_20_02.supp_mtx = ["edge_index"]


@register_config('bp_20_03_cfg')
def bp_20_03_cfg(cfg):
    """Configuration options for nx datasets.
    """
    cfg.bp_20_03 = CN()
    cfg.bp_20_03.num_samples = 10000
    cfg.bp_20_03.mean = 20
    cfg.bp_20_03.n_min = 10
    cfg.bp_20_03.n_max = 30
    cfg.bp_20_03.p_edge_bp = 0.3
    cfg.bp_20_03.p_edge_er = 0.3
    cfg.bp_20_03.supp_mtx = ["edge_index"]


@register_config('pc_test_cfg')
def pc_test_cfg(cfg):
    """Configuration options for nx datasets.
    """
    cfg.pc_test = CN()
    cfg.pc_test.num_samples = 1000
    cfg.pc_test.graph_size = 500
    cfg.pc_test.clique_size = None
    cfg.pc_test.supp_mtx = ["edge_index"]


@register_config('pc_500_20_cfg')
def pc_500_20_cfg(cfg):
    """Configuration options for nx datasets.
    """
    cfg.pc_500_20 = CN()
    cfg.pc_500_20.num_samples = 10000
    cfg.pc_500_20.graph_size = 500
    cfg.pc_500_20.clique_size = None
    cfg.pc_500_20.supp_mtx = ["edge_index"]


@register_config('pc_100_40_cfg')
def pc_100_40_cfg(cfg):
    """Configuration options for nx datasets.
    """
    cfg.pc_100_40 = CN()
    cfg.pc_100_40.num_samples = 10000
    cfg.pc_100_40.graph_size = 100
    cfg.pc_100_40.clique_size = 40
    cfg.pc_100_40.supp_mtx = ["edge_index"]


@register_config('ba_small_cfg')
def ba_small_cfg(cfg):
    """Configuration options for nx datasets.
    """
    cfg.ba_small = CN()
    cfg.ba_small.num_samples = 5000
    cfg.ba_small.n_min = 200
    cfg.ba_small.n_max = 300
    cfg.ba_small.num_edges = 4
    cfg.ba_small.supp_mtx = ["edge_index"]


@register_config('ba_large_cfg')
def ba_large_cfg(cfg):
    """Configuration options for nx datasets.
    """
    cfg.ba_large = CN()
    cfg.ba_large.num_samples = 5000
    cfg.ba_large.n_min = 800
    cfg.ba_large.n_max = 1200
    cfg.ba_large.num_edges = 4
    cfg.ba_large.supp_mtx = ["edge_index"]