import time
from functools import partial
from typing import Any, Dict, Tuple

import torch
from torch_geometric.graphgym import register

from torch_geometric.graphgym.config import cfg
from torch_geometric.graphgym.imports import LightningModule
from torch_geometric.graphgym.loss import compute_loss
from torch_geometric.graphgym.model_builder import GraphGymModule
from torch_geometric.graphgym.models.gnn import GNN
from torch_geometric.graphgym.optim import create_optimizer, create_scheduler
from torch_geometric.graphgym.register import network_dict

from modules.utils.spaces import OPTIMIZER_DICT, LOSS_FUNCTION_DICT, EVAL_FUNCTION_DICT


class COPTModule(GraphGymModule):
    def __init__(self, dim_in, dim_out, cfg):
        super().__init__(dim_in, dim_out, cfg)

        # Loss function
        self.loss_func = register.loss_dict[cfg.model.loss_fun]

        # Eval function
        self.eval_func_dict = EVAL_FUNCTION_DICT[cfg.train.task]
        for key, eval_func in self.eval_func_dict.items():
            self.eval_func_dict[key] = eval_func

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def configure_optimizers(self) -> Tuple[Any, Any]:
        optimizer = create_optimizer(self.model.parameters(), self.cfg.optim)
        scheduler = create_scheduler(optimizer, self.cfg.optim)
        return [optimizer], [scheduler]

    # def _shared_step(self, batch, split: str) -> Dict:
    #     batch.split = split
    #     pred, true = self(batch)
    #     loss, pred_score = compute_loss(pred, true)
    #     step_end_time = time.time()
    #     return dict(loss=loss, true=true, pred_score=pred_score.detach(),
    #                 step_end_time=step_end_time)

    def training_step(self, batch, *args, **kwargs):
        batch.split = "train"
        batch = self.forward(batch)
        loss = self.loss_func(batch)
        step_end_time = time.time()
        return dict(loss=loss, step_end_time=step_end_time)

    def validation_step(self, batch, *args, **kwargs):
        batch.split = "val"
        out = self.forward(batch)
        loss = self.loss_func(batch)
        step_end_time = time.time()
        eval_dict = dict(loss=loss, step_end_time=step_end_time)
        for eval_type, eval_func in self.eval_func_dict.items():
            eval = eval_func(batch)
            eval_dict.update({eval_type: eval})
        return eval_dict

    def test_step(self, batch, *args, **kwargs):
        out = self.forward(batch)
        loss = self.loss_func(batch)
        step_end_time = time.time()
        eval_dict = dict(loss=loss, step_end_time=step_end_time)
        for eval_type, eval_func in self.eval_func_dict.items():
            eval = eval_func(batch)
            eval_dict.update({eval_type: eval})
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

    model = COPTModule(dim_in, dim_out, cfg)
    if to_device:
        model.to(torch.device(cfg.accelerator))
    return model
