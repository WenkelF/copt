
import torch.nn as nn

from torch.optim import SGD, Adam

from modules.architecture.models import GCN, GAT, ScGCN

from utils.norms import min_max_norm    

from utils.metrics import (
    maxcut_loss,
    maxcut_mae,
    maxcut_acc,
    maxcut_p_exact,
    maxclique_loss,
    maxclique_ratio,
)

OPTIMIZER_DICT = {
    "sgd": SGD,
    "adam": Adam,
}

GNN_MODEL_DICT = {
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
    "maxcut": maxcut_loss,
    "maxclique": maxclique_loss,
}
    
EVAL_FUNCTION_DICT = {
    "maxcut": {"mae": maxcut_mae, "acc": maxcut_acc, "p_exact": maxcut_p_exact},
    "maxclique": {"mc_ratio": maxclique_ratio},
}