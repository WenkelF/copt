import torch


def collate_fn(batch):

    batch_size = len(batch)
    keys = batch[0].keys()
    feat_keys = [key for key in keys if key.startswith("node_")]
    num_nodes_list = [sample['num_nodes'] for sample in batch]
    max_num_nodes = max(num_nodes_list)

    padded_batch = {}
    for key in keys:
        if key.startswith("node_"):
            shape = (batch_size, max_num_nodes, batch[0][key].size(-1))
            padded_batch[key] = torch.zeros(shape)
            for idx, num_nodes in enumerate(num_nodes_list):
                padded_batch[key][idx, :num_nodes] = batch[idx][key]

        if key == 'cut_onehot':
            shape = (batch_size, max_num_nodes, batch[0][key].size(-1))
            padded_batch[key] = torch.full(shape, float('nan'))
            for idx, num_nodes in enumerate(num_nodes_list):
                padded_batch[key][idx, :num_nodes] = batch[idx][key]

        
        elif key in ['adj', 'gcn']:
            shape = (batch_size, max_num_nodes, max_num_nodes)
            padded_batch[key] = torch.zeros(shape)
            for idx, num_nodes in enumerate(num_nodes_list):
                padded_batch[key][idx,:num_nodes,:num_nodes] = batch[idx][key]
        
        elif key in ['target', 'num_nodes']:
            padded_batch[key] = torch.Tensor([batch[idx][key] for idx in range(batch_size)])
        
        else:
            pass

    padded_batch['x'] = torch.cat([padded_batch[key] for key in feat_keys], dim=-1)
    for key in feat_keys:
        padded_batch.pop(key)
   
    return padded_batch