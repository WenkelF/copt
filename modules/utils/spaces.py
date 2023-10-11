
import torch.nn as nn

from torch.optim import SGD, Adam

from modules.architecture.models import GCN, GAT, ScGCN
from modules.architecture.models_pyg import PygGCN

from utils.norms import min_max_norm    

from utils.metrics import (
    maxcut_loss,
    maxcut_mae,
    maxcut_acc,
    maxcut_loss_pyg,
    maxcut_mae_pyg,
    maxcut_acc_pyg,
    maxclique_loss,
    maxclique_ratio,
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

# POOLING_OPERATION_DICT = {
#     "maxcut": node_to_graph
# }

LAST_ACTIVATION_DICT = {
    "maxcut": nn.Sigmoid(),
    "maxclique": None,
}

LAST_NORMALIZATION_DICT = {
    "maxcut": None,
    "maxclique": min_max_norm,
}

LOSS_FUNCTION_DICT = {
    "maxcut": maxcut_loss_pyg,
    "maxclique": maxclique_loss,
}
    
EVAL_FUNCTION_DICT = {
    "maxcut": {"mae": maxcut_mae_pyg, "acc": maxcut_acc_pyg},
    "maxclique": {"mc_ratio": maxclique_ratio},
}