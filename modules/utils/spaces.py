
import torch.nn as nn

from torch.optim import SGD, Adam

from modules.architecture.models import GCN, GAT, ScGCN
from modules.architecture.models_pyg import PygGCN

from utils.norms import min_max_norm, min_max_norm_pyg

from utils.metrics import (
    maxcut_loss,
    maxcut_mae,
    maxcut_acc,
    maxcut_loss_pyg,
    maxcut_mae_pyg,
    maxcut_acc_pyg,
    maxclique_loss,
    maxclique_ratio,
    maxclique_loss_pyg,
    maxclique_ratio_pyg,
)

OPTIMIZER_DICT = {
    "sgd": SGD,
    "adam": Adam,
}

GNN_MODEL_DICT = {
    "pyg:gcn": PygGCN,
    "gcn": GCN,
    "gat": GAT,
    "scgcn": ScGCN,
}

LAST_ACTIVATION_DICT = {
    "maxcut": nn.Sigmoid(),
    "maxclique": nn.Sigmoid(),
    # "maxclique": None,
}

LAST_NORMALIZATION_DICT = {
    "maxcut": None,
    "maxclique": None,
    # "maxclique": min_max_norm_pyg,
}

LOSS_FUNCTION_DICT = {
    "maxcut": maxcut_loss_pyg,
    "maxclique": maxclique_loss_pyg,
}
    
EVAL_FUNCTION_DICT = {
    "maxcut": {"mae": maxcut_mae_pyg, "acc": maxcut_acc_pyg},
    "maxclique": {"approx_ratio": maxclique_ratio_pyg},
}