from typing import Any, Sequence, Union, Dict
import time
from loguru import logger

import torch

from pytorch_lightning import LightningModule
from pytorch_lightning.callbacks.callback import Callback

from torch.optim.lr_scheduler import StepLR

from modules.utils.spaces import OPTIMIZER_DICT


class PredictorModule(LightningModule):
    def __init__(
        self,
        model,
        loss_func,
        eval_func_dict: Dict,
        optimizer_kwargs: Dict,
        scheduler_kwargs: Dict,
    ):
        super().__init__()
        self.model = model
        self.loss_func = loss_func
        self.eval_func_dict = eval_func_dict
        self.optimizer_kwargs = optimizer_kwargs
        self.scheduler_kwargs = scheduler_kwargs

    def forward(self, data):
        return self.model.forward(data)
    
    def training_step(self, batch, batch_idx):
        data = batch
        out = self.forward(data)

        loss = self.loss_func(out, data)
        self.log("loss/train", loss, on_epoch=True, prog_bar=True, logger=True)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        data = batch
        out = self.forward(data)

        loss = self.loss_func(out, data)
        self.log("loss/valid", loss, on_epoch=True, prog_bar=True, logger=True)
        
        for eval_type, eval_func in self.eval_func_dict.items():
            eval = eval_func(out, data)
            self.log("".join([eval_type, "/valid"]), eval, on_epoch=True, prog_bar=True, logger=True)
        
        return loss
    
    def test_step(self, batch, batch_idx):
        data = batch
        out = self.forward(data)

        loss = self.loss_func(out, data)
        self.log("loss/test", loss, on_epoch=True, prog_bar=True, logger=True)
        
        for eval_type, eval_func in self.eval_func_dict.items():
            eval = eval_func(out, data)
            self.log("".join([eval_type, "/test"]), eval, on_epoch=True, prog_bar=True, logger=True)
        
        return loss
    
    def configure_optimizers(self):
        
        optimizer_type = self.optimizer_kwargs.pop("type")
        optimizer_type = OPTIMIZER_DICT[optimizer_type]
        optimizer = optimizer_type(self.model.parameters(), **self.optimizer_kwargs)

        scheduler = StepLR(optimizer, **self.scheduler_kwargs)

        return [optimizer], [scheduler]
    
    def on_train_epoch_start(self) -> None:
        self.epoch_start_time = time.time()

    def on_train_epoch_end(self) -> None:
        if self.epoch_start_time is None:
            logger.warning("Epoch timer not initialized")
        else:
            epoch_time = time.time() - self.epoch_start_time
            self.epoch_start_time = None
            self.log("epoch_time", torch.tensor(epoch_time))
