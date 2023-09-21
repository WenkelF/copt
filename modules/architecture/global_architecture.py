from typing import Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from modules.architecture.layers import GNN_layer, GAT_layer, hybrid_layer, hybrid_layer_pre, reweighting_layer

from modules.utils.spaces import GNN_MODEL_DICT, LAST_ACTIVATION_DICT


class FullGraphNetwork(nn.Module):
    def __init__(
        self,
        input_dim: int,
        task: str,
        gnn_kwargs: Dict[str, Any],
        head_kwargs: Dict[str, Any],
    ) -> nn.Module():
        super().__init__()   

        self.gnn = nn.ModuleDict()

        for module_name, module_kwargs in gnn_kwargs.items():
            gnn_type = module_kwargs.pop("type")
            self.gnn.update({
                module_name: GNN_MODEL_DICT[gnn_type](**module_kwargs, input_dim=input_dim)
            })
            # define new input_dim as last output_dim

        self.head = None

        self.pooling_operation = None

        self.last_activation = LAST_ACTIVATION_DICT[task]

    def forward(self, data):

        for module in self.gnn.values():
            data = module(data)

        out = data['x']

        if self.pooling_operation is not None:
            out = self.pooling_operation(out, data)

        out = self.last_activation(out)

        return out