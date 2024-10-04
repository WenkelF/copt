import logging
import time
from functools import partial
from typing import Any, Dict, Tuple

import torch
from torch_geometric.graphgym import register

from torch_geometric.graphgym.config import cfg
from torch_geometric.graphgym.imports import LightningModule
from torch_geometric.graphgym.loss import compute_loss
from torch_geometric.graphgym.model_builder import GraphGymModule
from torch_geometric.graphgym.optim import create_optimizer, create_scheduler
from torch_geometric.graphgym.register import network_dict

from graphgym.loss.copt_loss import entropy
from modules.utils.spaces import OPTIMIZER_DICT, LOSS_FUNCTION_DICT, EVAL_FUNCTION_DICT, EVAL_FUNCTION_DICT_NOLABEL


class COPTModule(GraphGymModule):
    def __init__(self, dim_in, dim_out, cfg):
        super().__init__(dim_in, dim_out, cfg)

        # Loss function
        loss_func = register.loss_dict[cfg.model.loss_fun]
        loss_params = cfg[cfg.model.loss_fun]
        self.loss_func = partial(loss_func, **loss_params)
        if cfg.optim.entropy.scheduler == "linear-energy":
            self.alpha = (cfg.optim.entropy.base_temp / cfg.optim.entropy.min_temp - 1) / cfg.optim.max_epoch
        elif cfg.optim.entropy.scheduler == "linear-entropy":
            self.alpha = (cfg.optim.entropy.base_temp - cfg.optim.entropy.min_temp) / cfg.optim.max_epoch

        # Eval function
        if not cfg.dataset.label:
            self.eval_func_dict = EVAL_FUNCTION_DICT_NOLABEL[cfg.train.task]
        else:
            self.eval_func_dict = EVAL_FUNCTION_DICT[cfg.train.task]
        for key, eval_func in self.eval_func_dict.items():
            if cfg.train.task in cfg.metrics:
                eval_func = partial(eval_func, **cfg.metrics[cfg.train.task])
            self.eval_func_dict[key] = eval_func

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def configure_optimizers(self) -> Tuple[Any, Any]:
        optimizer = create_optimizer(self.model.parameters(), self.cfg.optim)
        scheduler = create_scheduler(optimizer, self.cfg.optim)
        return [optimizer], [scheduler]

    def training_step(self, batch, *args, **kwargs):
        batch.split = "train"
        out = self.forward(batch)
        loss = self.loss_func(batch)
        self.log("loss/train", loss, batch_size=batch.batch_size, on_step=True, prog_bar=True, logger=True)

        if cfg.optim.entropy.enable:
            if cfg.optim.entropy.scheduler == "linear-energy":
                tau = cfg.optim.entropy.base_temp / (1.0 + self.alpha * self.current_epoch)
            elif cfg.optim.entropy.scheduler == "linear-entropy":
                tau = cfg.optim.entropy.base_temp - self.alpha * self.current_epoch
                
            H = tau * entropy(out)
            self.log("loss/train-entropy", H, batch_size=batch.batch_size, on_step=True, prog_bar=True, logger=True)

            loss -= H
            self.log("loss/train-anneal-loss", loss, batch_size=batch.batch_size, on_step=True, prog_bar=True, logger=True)

        step_end_time = time.time()
        return dict(loss=loss, step_end_time=step_end_time)

    def validation_step(self, batch, *args, **kwargs):
        batch.split = "val"
        out = self.forward(batch)
        loss = self.loss_func(batch)
        step_end_time = time.time()
        eval_dict = dict(loss=loss, step_end_time=step_end_time)
        self.log("loss/valid", loss, batch_size=batch.batch_size, on_epoch=True, prog_bar=True, logger=True)
        for eval_type, eval_func in self.eval_func_dict.items():
            eval = eval_func(batch)
            eval_dict.update({eval_type: eval})
            self.log("".join([eval_type, "/valid"]), eval, batch_size=batch.batch_size, on_epoch=True, prog_bar=True, logger=True)
        return eval_dict

    def test_step(self, batch, *args, **kwargs):
        cfg.test = True
        out = self.forward(batch)
        loss = self.loss_func(batch)
        step_end_time = time.time()
        eval_dict = dict(loss=loss, step_end_time=step_end_time)
        self.log("loss/test", loss, batch_size=batch.batch_size, on_epoch=True, prog_bar=True, logger=True)
        for eval_type, eval_func in self.eval_func_dict.items():
            eval = eval_func(batch)
            eval_dict.update({eval_type: eval})
            self.log("".join([eval_type, "/test"]), eval, batch_size=batch.batch_size, on_epoch=True, prog_bar=True, logger=True)
        return eval_dict

    @property
    def encoder(self) -> torch.nn.Module:
        return self.model.encoder

    @property
    def mp(self) -> torch.nn.Module:
        return self.model.mp

    @property
    def post_mp(self) -> torch.nn.Module:
        return self.model.post_mp

    @property
    def pre_mp(self) -> torch.nn.Module:
        return self.model.pre_mp


def create_model(to_device=True, dim_in=None, dim_out=None) -> GraphGymModule:
    r"""Create model for graph machine learning.

    Args:
        to_device (bool, optional): Whether to transfer the model to the
            specified device. (default: :obj:`True`)
        dim_in (int, optional): Input dimension to the model
        dim_out (int, optional): Output dimension to the model
    """
    dim_in = cfg.share.dim_in if dim_in is None else dim_in
    dim_out = cfg.share.dim_out if dim_out is None else dim_out
    # binary classification, output dim = 1
    if 'classification' in cfg.dataset.task_type and dim_out == 2:
        dim_out = 1

    if cfg.pretrained.dir:
        logging.info(f'Loading pretrained model from {cfg.pretrained.dir}')
        model = COPTModule.load_from_checkpoint(cfg.pretrained.dir, dim_in=dim_in, dim_out=dim_out, cfg=cfg)
    else:
        model = COPTModule(dim_in, dim_out, cfg)
    if to_device:
        model.to(torch.device(cfg.accelerator))
    return model
