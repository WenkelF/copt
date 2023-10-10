from typing import Tuple, Union, List, Optional, Dict, Any, OrderedDict, Literal

import os

from loguru import logger

import torch
import pickle

from torch_geometric.datasets import TUDataset
from torch_geometric.utils import to_networkx
from torch_geometric.data import Dataset as PygDataset
from torch_geometric.loader import DataLoader as PygDataLoader
from torch_geometric.data import Data
from torch.utils.data import DataLoader, Dataset
from pytorch_lightning import LightningDataModule

from modules.data.data_generation import generate_sample
from modules.data.collate import collate_fn, collate_fn_pyg


class DataModule(LightningDataModule):
    def __init__(
        self,
        task: str,
        data_kwargs: Dict[str, Any],
        feat_kwargs: Dict[str, Any],
        data_dir: str,
        train_ds_kwargs: Dict[str, Any],
        valid_ds_kwargs: Dict[str, Any] = None,
        test_ds_kwargs: Dict[str, Any] = None,
        batch_size_train: int = 16,
        batch_size_valid: int = 16,
        regenerate: bool = False,
        backend: Literal["base", "pyg"] = "base",

    ):
        super().__init__()
        self.task = task
        self.data_kwargs = data_kwargs
        self.feat_kwargs = feat_kwargs
        self.data_dir = data_dir
        self.regenerate = regenerate
        self.backend = backend

        self.train_ds_kwargs = train_ds_kwargs
        self.valid_ds_kwargs = valid_ds_kwargs
        self.test_ds_kwargs = test_ds_kwargs
        self.batch_size_train = batch_size_train
        self.batch_size_valid = batch_size_valid

        self.base_dataset = None
        self.base_targets = None

        self.datasets = None

        self.dataset = PygDataset if backend == "pyg" else Dataset
        self.dataloader = PygDataLoader if backend == "pyg" else DataLoader
        self.collate_fn = None if backend == "pyg" else collate_fn


    def prepare_data(self) -> None:

        logger.info("Preparing data...")
        
        if self.datasets is not None:
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

            self.generate_datasets(
                self.train_ds_kwargs,
                self.valid_ds_kwargs,
                self.test_ds_kwargs,
            )
            
            logger.info("Saving data...")
            torch.save(self.datasets, path + 'datasets.pt')

        if self.datasets is None:
            self.datasets = torch.load(path + 'datasets.pt')

        logger.info("Done.")

    def setup(self, stage: str):

        if stage == "fit":
            self.train_ds = self.get_dataset(step="train")
            self.valid_ds = self.get_dataset(step="valid")

        if stage == "test":
            self.test_ds = self.get_dataset(step="test")


    def generate_datasets(
        self,
        train_ds_kwargs: Dict[str, Any],
        valid_ds_kwargs: Dict[str, Any] = None,
        test_ds_kwargs: Dict[str, Any] = None,
    ) -> None:
        
        from_existing = self.data_kwargs.pop("from_existing", None)
        if from_existing is not None:
            self.get_base_dataset(from_existing)
  
            # Shuffle data
            perm = torch.randperm(len(self.base_dataset))
            self.base_dataset = [self.base_dataset[idx] for idx in perm]
            if self.base_targets is not None:
                for key in self.base_targets.keys():
                    self.base_targets[key] = [self.base_targets[key][idx] for idx in perm]

        self.num_samples = self.data_kwargs.pop("num_samples", None)

        if self.num_samples is not None:
            if from_existing is not None:
                self.num_samples = min(self.num_samples, len(self.base_dataset))    
        else:
            if from_existing is not None:
                self.num_samples = len(self.base_dataset)
            else:
                raise ValueError("Number of samples not specified.")

        self.tmp_sample_idx = 0

        train_ds = self.get_samples(**train_ds_kwargs) if train_ds_kwargs is not None else None
        valid_ds = self.get_samples(**valid_ds_kwargs) if valid_ds_kwargs is not None else None
        test_ds = self.get_samples(**test_ds_kwargs) if test_ds_kwargs is not None else None

        self.datasets = {
            "train": train_ds,
            "valid": valid_ds,
            "test": test_ds,
        }


    def get_samples(self, frac: float, label: bool = False) -> List[Dict[str, Any]]:

        ds = []

        last_sample_idx = self.tmp_sample_idx + int(frac * self.num_samples)

        for idx in range(self.tmp_sample_idx, last_sample_idx):
            base_graph = None
            base_target = None
            if self.base_dataset is not None:
                base_graph = to_networkx(self.base_dataset[idx])
            if self.base_targets is not None:
                base_target = {key: targets[idx] for key, targets in self.base_targets.items()}
                
            ds.append(
                generate_sample(
                    self.task,
                    self.data_kwargs,
                    self.feat_kwargs,
                    base_graph=base_graph,
                    base_target=base_target,
                    label=label,
                )
            )
        
        self.tmp_sample_idx = last_sample_idx
        return ds


    def get_base_dataset(
        self,
        dataset_name: str
    ) -> List[Dict[str, Any]]:
        
        self.base_dataset = None
        self.base_targets = None

        if dataset_name == "imdb":
            self.base_dataset = TUDataset(root='/tmp/IMDB-BINARY', name='IMDB-BINARY')
            if self.task == "maxclique":
                self.base_targets = {}
                with open("data/maxclique/IMDB-BINARYcliqno.txt","rb") as file:
                    self.base_targets["mc_size"] = pickle.load(file)
        elif dataset_name == "collab":
            self.base_dataset = TUDataset(root='/tmp/COLLAB', name='COLLAB')
            if self.task == "maxclique":
                self.base_targets = {}
                with open("data/maxclique/COLLABcliqno.txt","rb") as file:
                    self.base_targets["mc_size"] = pickle.load(file)
        else:
            raise NotImplementedError("Unsupported dataset name.")


    def train_dataloader(self):
        return self.dataloader(self.train_ds, batch_size=self.batch_size_train, shuffle=True, collate_fn=self.collate_fn)

    def val_dataloader(self):
        return self.dataloader(self.valid_ds, batch_size=self.batch_size_valid, shuffle=False, collate_fn=self.collate_fn)

    def test_dataloader(self):
        return self.dataloader(self.test_ds, batch_size=self.batch_size_valid, shuffle=False, collate_fn=self.collate_fn)

    def get_num_features(self):
        sample = self.datasets["train"][0]
        feat_keys = [key for key in sample.keys() if key.startswith("node_")]

        return sum([sample[key].size(-1) for key in feat_keys])

    def get_dataset(self, step: str):
        dataset = self.datasets[step]

        if self.backend == "pyg":
            for sample in self.datasets[step]:
                sample_keys = list(sample.keys()).copy()
                feat = [sample.pop(key) for key in sample_keys if key.startswith("node_")]
                sample["x"] = torch.cat(feat, dim=-1)
            dataset = [Data.from_dict(sample) for sample in self.datasets[step]]

        # if self.backend == "pyg":
        #     adj_list = []
        #     data_list = []
        #     for sample in self.datasets[step]:
        #         adj_list.append(sample.pop("adj_mat"))
        #         data_list.append(Data.from_dict(sample))

        #     dataset = (adj_list, data_list)

        return self.dataset(dataset)


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


class PygDataset(PygDataset):
    def __init__(
        self,
        samples: Dict[str, torch.Tensor]
    ) -> None:
        super().__init__()
        self.samples = samples

    def len(self):
        return len(self.samples)

    def get(self, idx):
        return self.samples[idx]