import os
from os.path import dirname, abspath
import yaml
from copy import deepcopy
from loguru import logger

import wandb

from modules.config._loader import (
        load_datamodule,
        load_architecture,
        load_predictor,
        load_trainer,
)

from utils import set_seed, maxcut_loss, maxcut_mae


def main(cfg):

        cfg = deepcopy(cfg)
        
        wandb_cfg = cfg["constants"].get("wandb", None)
        if wandb_cfg is not None:
                wandb.init(config=cfg, **wandb_cfg)   

        set_seed(cfg["constants"]["seed"])

        # Datamodule
        datamodule = load_datamodule(cfg)
        datamodule.prepare_data()

        # Model
        num_feat = datamodule.get_num_features()

        # Model
        model, loss_func_dict, eval_func = load_architecture(cfg, num_feat)

        # Predictor
        predictor = load_predictor(cfg, model, loss_func_dict, eval_func)

        trainer = load_trainer(cfg)

        logger.info("Starting training...")
        trainer.fit(model=predictor, datamodule=datamodule)
        logger.info("Done.")

        logger.info("Testing...")
        trainer.test(model=predictor, datamodule=datamodule)
        logger.info("Done.")


if __name__ == "__main__":
        MAIN_DIR = os.getcwd()
        CONFIG_FILE = "expts/configs/config.yaml"

        os.chdir(MAIN_DIR)

        with open(os.path.join(MAIN_DIR, CONFIG_FILE), "r") as f:
                cfg = yaml.safe_load(f)

        main(cfg)