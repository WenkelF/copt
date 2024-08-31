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

    cfg.dataset.append_stats = False
    cfg.dataset.graph_stats = ['degree', 'eccentricity', 'cluster_coefficient', 'triangle_count']

    cfg.dataset.multiprocessing = True
    cfg.dataset.label = False

    cfg.train.batch_size_val = 256


@register_config('satlib_cfg')
def er_test_cfg(cfg):
    cfg.satlib = CN()
    cfg.satlib.gen_labels = False
    cfg.satlib.weighted = False


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
    cfg.er.v50_02 = CN()
    cfg.er.v50_02.n_min = 30
    cfg.er.v50_02.n_max = 50
    cfg.er.v50_02.p = 0.2


@register_config('er_50_03_cfg')
def er_50_03_cfg(cfg):
    cfg.er.v50_03 = CN()
    cfg.er.v50_03.n_min = 30
    cfg.er.v50_03.n_max = 50
    cfg.er.v50_03.p = 0.3


@register_config('er_50_04_cfg')
def er_50_04_cfg(cfg):
    cfg.er.v50_04 = CN()
    cfg.er.v50_04.n_min = 30
    cfg.er.v50_04.n_max = 50
    cfg.er.v50_04.p = 0.4


@register_config('bp_cfg')
def bp_cfg(cfg):
    """Configuration options for nx datasets.
    """
    cfg.bp = CN()
    cfg.bp.num_samples = 10000
    cfg.bp.mean = 10
    cfg.bp.n_min = 10
    cfg.bp.n_max = 30
    cfg.bp.p_edge_bp = 0.3
    cfg.bp.p_edge_er = 0.1
    cfg.bp.supp_mtx = ["edge_index"]


@register_config('bp_20_00_cfg')
def bp_20_00_cfg(cfg):
    cfg.bp.v20_00 = CN()
    cfg.bp.v20_00.mean = 20
    cfg.bp.v20_00.p_edge_er = 0.0


@register_config('bp_20_01_cfg')
def bp_20_01_cfg(cfg):
    cfg.bp.v20_01 = CN()
    cfg.bp.v20_01.mean = 20
    cfg.bp.v20_01.p_edge_er = 0.1


@register_config('bp_20_02_cfg')
def bp_20_02_cfg(cfg):
    cfg.bp.v20_02 = CN()
    cfg.bp.v20_02.mean = 20
    cfg.bp.v20_02.p_edge_er = 0.2


@register_config('bp_20_03_cfg')
def bp_20_03_cfg(cfg):
    cfg.bp.v20_03 = CN()
    cfg.bp.v20_03.mean = 20
    cfg.bp.v20_03.p_edge_er = 0.3


@register_config('rb_cfg')
def rb_cfg(cfg):
    """Configuration options for nx datasets.
    """
    cfg.rb = CN()
    # features can be one of ['node_const', 'node_onehot', 'node_clustering_coefficient', 'node_pagerank']
    cfg.rb.num_samples = 4500


@register_config('rb_small_cfg')
def rb_small_cfg(cfg):
    """Configuration options for nx datasets.
    """
    cfg.rb.small = CN()
    cfg.rb.small.num_samples = 4500
    cfg.rb.small.n = (200, 300)
    cfg.rb.small.na = (20, 25)
    cfg.rb.small.k = (5, 12)


@register_config('rb_large_cfg')
def rb_large_cfg(cfg):
    """Configuration options for nx datasets.
    """
    cfg.rb.large = CN()
    cfg.rb.large.num_samples = 4500
    cfg.rb.large.n = (800, 1200)
    cfg.rb.large.na = (40, 55)
    cfg.rb.large.k = (20, 25)

@register_config('pc_cfg')
def pc_cfg(cfg):
    """Configuration options for nx datasets.
    """
    cfg.pc = CN()
    cfg.pc.num_samples = 1000
    cfg.pc.graph_size = 500
    cfg.pc.clique_size = None
    cfg.pc.supp_mtx = ["edge_index"]


@register_config('pc_500_20_cfg')
def pc_500_20_cfg(cfg):
    """Configuration options for nx datasets.
    """
    cfg.pc.v500_20 = CN()
    cfg.pc.v500_20.graph_size = 500
    cfg.pc.v500_20.clique_size = None


@register_config('pc_100_40_cfg')
def pc_100_40_cfg(cfg):
    """Configuration options for nx datasets.
    """
    cfg.pc.v100_40 = CN()
    cfg.pc.v100_40.graph_size = 100
    cfg.pc.v100_40.clique_size = 40


@register_config('ba_cfg')
def ba_cfg(cfg):
    cfg.ba = CN()
    cfg.ba.num_samples = 5000
    cfg.ba.n_min = 200
    cfg.ba.n_max = 300
    cfg.ba.num_edges = 4
    cfg.ba.supp_mtx = ["edge_index"]


@register_config('ba_small_cfg')
def ba_small_cfg(cfg):
    """Configuration options for nx datasets.
    """
    cfg.ba.small = CN()
    cfg.ba.small.n_min = 200
    cfg.ba.small.n_max = 300


@register_config('ba_large_cfg')
def ba_large_cfg(cfg):
    """Configuration options for nx datasets.
    """
    cfg.ba.large = CN()
    cfg.ba.large.n_min = 800
    cfg.ba.large.n_max = 1200
