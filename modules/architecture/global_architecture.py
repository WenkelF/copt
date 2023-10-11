from typing import Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from modules.utils.spaces import GNN_MODEL_DICT, LAST_ACTIVATION_DICT, LAST_NORMALIZATION_DICT
from modules.architecture.models import MLP 


class FullGraphNetwork(nn.Module):
    def __init__(
        self,
        input_dim: int,
        task: str,
        gnn_kwargs: Dict[str, Any],
        head_kwargs: Dict[str, Any],
    ) -> nn.Module():
        super().__init__()   

        self.norm = nn.LayerNorm(input_dim)

        self.gnn = nn.ModuleDict()

        for module_name, module_kwargs in gnn_kwargs.items():
            gnn_type = module_kwargs.pop("type")
            self.gnn.update({
                module_name: GNN_MODEL_DICT[gnn_type](**module_kwargs, input_dim=input_dim)
            })
            # define new input_dim as last output_dim

        self.head = None
        if head_kwargs is not None:
           self.head = MLP(**head_kwargs, input_dim=list(gnn_kwargs.values())[-1]['output_dim'])

        self.pooling_operation = None

        self.last_activation = LAST_ACTIVATION_DICT[task]

        self.last_norm = LAST_NORMALIZATION_DICT[task]

    def forward(self, data):

        data['x'] = self.norm(data['x'])

        for module in self.gnn.values():
            data = module(data)

        x = data['x']

        if self.pooling_operation is not None:
            x = self.pooling_operation(x, data)

        if self.head is not None:
            x = self.head(x)

        if self.last_activation is not None:
            x = self.last_activation(x)
        
        if self.last_norm is not None:
            x = self.last_norm(x, data)

        data['x'] = x

        return data