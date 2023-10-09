from typing import Dict, Any

import torch

def min_max_norm(x: torch.Tensor, data: Dict[str, Any]):

    # for sample_idx in range(x.size(0)):
    #     x_min = torch.min(x[sample_idx, :data['num_nodes'][sample_idx]], dim=0)[0].detach()
    #     x_max = torch.max(x[sample_idx, :data['num_nodes'][sample_idx]], dim=0)[0].detach()
    #     x[sample_idx, :data['num_nodes'][sample_idx]] = (x[sample_idx, :data['num_nodes'][sample_idx]] - x_min) / (x_max - x_min)
    
    # for idx, num_nodes in enumerate(data['num_nodes']):
    #     x[idx, num_nodes:] = float('inf')
    # x_min = torch.zeros(x.size(0))
    # x_max = torch.zeros(x.size(0))
    
    x_min = torch.min(x, axis=1, keepdims=True)[0]

    # for idx, num_nodes in enumerate(data['num_nodes']):
    #     x[idx, num_nodes:] = - float('inf')
    
    x_max = torch.max(x, axis=1, keepdims=True)[0]
    
    x = (x - x_min) / (x_max - x_min)

    # for idx, num_nodes in enumerate(data['num_nodes']):
    #     x[idx, num_nodes:] = 0.

    return x