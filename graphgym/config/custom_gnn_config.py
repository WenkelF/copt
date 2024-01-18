from yacs.config import CfgNode as CN
from torch_geometric.graphgym.register import register_config


@register_config('custom_gnn')
def custom_gnn_cfg(cfg):
    """Extending config group of GraphGym's built-in GNN for purposes of our
    CustomGNN network model.
    """

    # Use residual connections between the GNN layers.
    cfg.gnn.residual = False

    # Attention dropout ratios
    cfg.gnn.att_dropout = 0.0

    # Concatenate embeddings from multihead attentions, followed by a lin proj
    cfg.gnn.att_concat_proj = False

    cfg.gnn.last_act = None
    cfg.gnn.last_norm = None


    cfg.gnn.hybrid = CN()
    cfg.gnn.hybrid.channel_list = [[1], [2], [4], [0, 1], [1, 2], [2, 4]]
    cfg.gnn.hybrid.combine_fn = 'cat'
    cfg.gnn.hybrid.residual = False
    cfg.gnn.hybrid.activation_channel = 'relu'
    cfg.gnn.hybrid.num_heads = 1
    cfg.gnn.hybrid.add_self_loops = True
    
    cfg.gnn.hybrid_v2 = CN()
    cfg.gnn.hybrid_v2.channel_list = [[0], [1], [2], [4], [0, 1], [1, 2], [2, 4]]
    cfg.gnn.hybrid_v2.activation_att1 = 'relu'
    cfg.gnn.hybrid_v2.activation_att2 = 'relu'
    cfg.gnn.hybrid_v2.activation = 'leaky_relu'
    cfg.gnn.hybrid_v2.num_heads = 1
    cfg.gnn.hybrid_v2.depth_mlp = 1
    cfg.gnn.hybrid_v2.skip = False