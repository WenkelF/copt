from torch_geometric.graphgym.register import register_config
from yacs.config import CfgNode as CN


@register_config('extended_optim')
def extended_optim_cfg(cfg):
    """Extend optimizer config group that is first set by GraphGym in
    torch_geometric.graphgym.config.set_cfg
    """

    # Number of batches to accumulate gradients over before updating parameters
    # Requires `custom` training loop, set `train.mode: custom`
    cfg.optim.batch_accumulation = 1

    # ReduceLROnPlateau: Factor by which the learning rate will be reduced
    cfg.optim.reduce_factor = 0.1

    # ReduceLROnPlateau: #epochs without improvement after which LR gets reduced
    cfg.optim.schedule_patience = 10

    # ReduceLROnPlateau: Lower bound on the learning rate
    cfg.optim.min_lr = 0.0

    # For schedulers with warm-up phase, set the warm-up number of epochs
    cfg.optim.num_warmup_epochs = 50

    # Clip gradient norms while training
    cfg.optim.clip_grad_norm = False

    # Ascending steps for FLAG
    cfg.optim.flag_steps = 3

    # Step size for FLAG
    cfg.optim.flag_step_size = 0.001
    
    cfg.optim.entropy = CN()
    cfg.optim.entropy.enable = False
    cfg.optim.entropy.scheduler = "linear-energy"
    cfg.optim.entropy.base_temp = 100.0
    cfg.optim.entropy.min_temp = 0.001

    # cfg.optim.train_mode = cfg.train.mode
    cfg.optim.eval_period = cfg.train.eval_period
