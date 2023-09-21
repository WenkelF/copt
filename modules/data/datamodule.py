from typing import Tuple, Union, Optional, Dict, Any, OrderedDict

import os

from loguru import logger

import torch

from torch.utils.data import DataLoader, Dataset
from pytorch_lightning import LightningDataModule

from modules.data.data_generation import generate_dataset
from modules.data.collate import collate_fn


class DataModule(LightningDataModule):
    def __init__(
        self,
        task: str,
        data_kwargs: Dict[str, Any],
        feat_kwargs: Dict[str, Any],
        data_dir: str,
        train_ds_kwargs: Dict[str, Any],
        valid_ds_kwargs: Dict[str, Any],
        batch_size_train: int = 16,
        batch_size_valid: int = 16,
        regenerate: bool = False,

    ):
        super().__init__()
        self.task = task
        self.data_kwargs = data_kwargs
        self.feat_kwargs = feat_kwargs
        self.data_dir = data_dir
        self.regenerate = regenerate

        self.train_ds_kwargs = train_ds_kwargs
        self.valid_ds_kwargs = valid_ds_kwargs
        self.batch_size_train = batch_size_train
        self.batch_size_valid = batch_size_valid

        self.samples = None

    def prepare_data(self) -> None:

        logger.info("Preparing data...")
        
        if self.samples is not None:
            logger.info("Data already prepared.")
            return

        # Load (or generate) data
        path = "".join([self.data_dir, self.task, "/"])
        
        if not os.path.exists(path) or self.regenerate:
            logger.info("Generating data...")
            try:
                os.mkdir(path)
            except:
                pass
            
            samples = generate_dataset(
                name = self.task,
                data_kwargs = self.data_kwargs,
                feat_kwargs = self.feat_kwargs,
            )

            logger.info("Saving data...")
            torch.save(samples, path + 'samples.pt')

        samples = torch.load(path + 'samples.pt')

        # Shuffle data
        perm = torch.randperm(len(samples))
        self.samples = [samples[idx] for idx in perm]

        logger.info("Done.")

    def setup(self, stage: str):

        if stage == "fit":
            self.train_ds = self.get_dataset(**self.train_ds_kwargs)
            self.valid_ds = self.get_dataset(**self.valid_ds_kwargs)

        # if stage == "test":
        #     self.test_ds = self.get_dataset(**test_loader_args)

    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size=self.batch_size_train, shuffle=True, collate_fn=collate_fn)

    def val_dataloader(self):
        return DataLoader(self.valid_ds, batch_size=self.batch_size_valid, shuffle=False, collate_fn=collate_fn)

    def test_dataloader(self):
        return DataLoader(self.train_ds, batch_size=self.batch_size_valid, shuffle=False, collate_fn=collate_fn)

    def get_num_features(self):
        sample = self.samples[0]
        feat_keys = [key for key in sample.keys() if key.startswith("node_")]

        return sum([sample[key].size(-1) for key in feat_keys])

    def get_dataset(self, start, end):
        return Dataset(self.samples[start:end])


class Dataset(Dataset):
    def __init__(
        self,
        samples: Dict[str, torch.Tensor]
    ) -> None:
        super().__init__()
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]