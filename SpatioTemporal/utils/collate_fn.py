import torch

def custom_collate_fn(batch):
    """
    Args:
        batch: 每个样本是包含4个邻居字典的列表
        
    Returns:
        dict: 堆叠后的张量
    """
    anchors = torch.stack([
        torch.stack([torch.stack(neighbor['anchors']) for neighbor in sample])
        for sample in batch
    ])
    
    temporal_negatives = torch.stack([
        torch.stack([torch.stack(neighbor['temporal_negatives']) for neighbor in sample])
        for sample in batch
    ])
    
    temporal_negatives_ratios = torch.stack([
        torch.stack([torch.stack(neighbor['temporal_negatives_ratios']) for neighbor in sample])
        for sample in batch
    ])
    
    return {
        'anchors': anchors,    # [batch_size, 4, n_views, C, H, W]
        'temporal_negatives': temporal_negatives,  # [batch_size, 4, num_neg, C, H, W]
        'temporal_negatives_ratios': temporal_negatives_ratios  # [batch_size, 4, num_neg, label_num]
    }
