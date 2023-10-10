from modules.data.datamodule import DataModule

from modules.utils.spaces import OPTIMIZER_DICT, LOSS_FUNCTION_DICT, EVAL_FUNCTION_DICT
from modules.architecture.global_architecture import FullGraphNetwork
from modules.trainer.predictor import PredictorModule

from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger

from functools import partial


def load_datamodule(cfg):

    cfg_data = cfg["datamodule"]

    datamodule = DataModule(**cfg_data)

    return datamodule

def load_architecture(cfg, input_dim):

    cfg_arch = cfg["architecture"]
    loss_kwargs = cfg_arch.pop("loss_kwargs", {})
    eval_kwargs = cfg_arch.pop("eval_kwargs", {})

    task = cfg_arch.get("task")

    # Loss function
    loss_func = LOSS_FUNCTION_DICT[task]
    loss_func = partial(loss_func, **loss_kwargs)

    # Eval function
    eval_func_dict = EVAL_FUNCTION_DICT[task]
    for key, eval_func in eval_func_dict.items():
        eval_func_dict[key] = partial(eval_func, **eval_kwargs.get(key, {}))
    
    model = FullGraphNetwork(input_dim, **cfg_arch)

    return model, loss_func, eval_func_dict

def load_predictor(cfg, model, loss_func, eval_func_dict):

    optimizer_kwargs = cfg["training"].pop("optimizer_kwargs")
    scheduler_kwargs = cfg["training"].pop("scheduler_kwargs")

    predictor = PredictorModule(model, loss_func, eval_func_dict, optimizer_kwargs=optimizer_kwargs, scheduler_kwargs=scheduler_kwargs, accelerator=cfg["constants"]["device"])

    return predictor

def load_trainer(cfg):

    train_kwargs = cfg["training"].pop("train_kwargs")
    wandb_kwargs = cfg["constants"].get("wandb", None)

    trainer = Trainer(**train_kwargs)

    if wandb_kwargs is not None:
        trainer.logger = WandbLogger(**wandb_kwargs)

    return trainer