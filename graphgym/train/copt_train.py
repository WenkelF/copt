import time

import torch

from torch_geometric.graphgym.checkpoint import (
    clean_ckpt,
    load_ckpt,
    save_ckpt,
)
from torch_geometric.graphgym.config import cfg
from torch_geometric.graphgym.loss import compute_loss
from torch_geometric.graphgym.register import register_train
from torch_geometric.graphgym.utils.epoch import is_ckpt_epoch, is_eval_epoch

from graphgym.utils import make_wandb_name, cfg_to_dict
from modules.config._loader import load_architecture, load_predictor, load_trainer
from modules.data.datamodule_graphgym import GraphGymDataModule, train


@register_train('copt')
@register_train('copt_test')
def train_copt(cfg, loaders, model):
    if cfg.wandb.use:
        try:
            import wandb
        except ModuleNotFoundError:
            raise ImportError('WandB is not installed. ($ pip install wandb)')
        if cfg.wandb.name == '':
            wandb_name = make_wandb_name(cfg)
        else:
            wandb_name = cfg.wandb.name
        run = wandb.init(entity=cfg.wandb.entity, project=cfg.wandb.project,
                         name=wandb_name)
        run.config.update(cfg_to_dict(cfg))

    # Datamodule
    datamodule = GraphGymDataModule(loaders)
    # datamodule.prepare_data()

    # Model
    # num_feat = datamodule.get_num_features()
    #model, loss_func, eval_func_dict = load_architecture(cfg, num_feat)

    # Predictor
    #predictor = load_predictor(cfg, model, loss_func, eval_func_dict)

    train(model, datamodule)
    # close wandb
    if cfg.wandb.use:
        run.finish()
        run = None
