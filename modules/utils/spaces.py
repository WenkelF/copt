
import torch.nn as nn

from torch.optim import SGD, Adam

from modules.architecture.models import GAT

from utils.metrics import (
    maxcut_loss,
    maxcut_mae,
    maxcut_acc,
)

OPTIMIZER_DICT = {
    "sgd": SGD,
    "adam": Adam,
}

GNN_MODEL_DICT = {
    "gat": GAT,
}

# POOLING_OPERATION_DICT = {
#     "maxcut": node_to_graph
# }

LAST_ACTIVATION_DICT = {
    "maxcut": nn.Sigmoid(),
}

LOSS_FUNCTION_DICT = {
    "maxcut": maxcut_loss,
}
    
EVAL_FUNCTION_DICT = {
    "maxcut": {"mae": maxcut_mae, "acc": maxcut_acc},
}