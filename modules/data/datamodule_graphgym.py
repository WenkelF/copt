import logging
import time
import warnings
from typing import Optional

import torch
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader

from torch_geometric.data.lightning.datamodule import LightningDataModule
from torch_geometric.graphgym import create_loader
from torch_geometric.graphgym.checkpoint import get_ckpt_dir
from torch_geometric.graphgym.config import cfg
from torch_geometric.graphgym.imports import pl
from pytorch_lightning.callbacks import LearningRateMonitor
from torch_geometric.graphgym.model_builder import GraphGymModule


class GraphGymDataModule(LightningDataModule):
    def __init__(self, loaders):
        self.loaders = loaders
        super().__init__(has_val=True, has_test=True)

    def train_dataloader(self) -> DataLoader:
        return self.loaders[0]

    def val_dataloader(self) -> DataLoader:
        # better way would be to test after fit.
        # First call trainer.fit(...) then trainer.test(...)
        return self.loaders[1]

    def test_dataloader(self) -> DataLoader:
        return self.loaders[2]


def train(model: GraphGymModule, datamodule, logger: bool = True,
          trainer_config: Optional[dict] = None):
    warnings.filterwarnings('ignore', '.*use `CSVLogger` as the default.*')

    callbacks = []
    if cfg.train.enable_ckpt:
        ckpt_cbk = pl.callbacks.ModelCheckpoint(dirpath=get_ckpt_dir(),
                                                monitor=cfg.train.ckpt_monitor,
                                                filename='epoch{epoch:02d}-val_size{size/valid:.2f}')
        callbacks.append(ckpt_cbk)

    # Monitor learning rate
    lr_monitor = LearningRateMonitor(logging_interval='epoch')
    callbacks.append(lr_monitor)

    trainer_config = trainer_config or {}
    trainer = pl.Trainer(
        **trainer_config,
        enable_checkpointing=cfg.train.enable_ckpt,
        callbacks=callbacks,
        default_root_dir=cfg.out_dir,
        max_epochs=cfg.optim.max_epoch,
        accelerator=cfg.accelerator,
        devices='auto' if not torch.cuda.is_available() else cfg.devices,
        check_val_every_n_epoch=cfg.train.val_period,
        accumulate_grad_batches=cfg.optim.batch_accumulation
    )

    if cfg.wandb.use:
        trainer.logger = WandbLogger(**cfg.wandb)

    if not cfg.train.mode == 'copt_test':
        trainer.fit(model, datamodule=datamodule)
    elif not cfg.pretrained.dir:
        logging.warning(f'You are running inference on a model that has not been trained. Either train a model first, or provide a checkpoint using "cfg.pretrained.dir".')
    t1 = time.time()
    if not cfg.train.mode == 'copt_test' and cfg.train.enable_ckpt:
        trainer.test(model, datamodule=datamodule, ckpt_path="best")
    else:
        trainer.test(model, datamodule=datamodule)
    t2 = time.time()
    print(f'Test time: {t2-t1}')
