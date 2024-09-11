from torch_geometric.graphgym.register import register_config
from yacs.config import CfgNode as CN


@register_config('loss_param_cfg')
def loss_param_cfg(cfg):
    cfg.maxclique_loss = CN()
    cfg.maxclique_loss.beta = 0.1

    cfg.maxcut_loss = CN()

    cfg.mds_loss = CN()
    cfg.mds_loss.beta = 1.0

    cfg.mis_loss = CN()
    cfg.mis_loss.beta = 1.0
    # cfg.mis_loss.k = 2

    cfg.plantedclique_loss = CN()

    cfg.metrics = CN()

    cfg.metrics.maxclique = CN()
    cfg.metrics.maxclique.dec_length = 300
    cfg.metrics.maxclique.num_seeds = 1

    cfg.metrics.mds = CN()
    cfg.metrics.mds.enable = True
    cfg.metrics.mds.num_seeds = 1

    cfg.metrics.mis = CN()
    cfg.metrics.mis.dec_length = 100
    cfg.metrics.maxclique.num_seeds = 1