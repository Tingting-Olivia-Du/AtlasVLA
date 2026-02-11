"""
Evaluation metrics
"""
import numpy as np


def compute_metrics(actions_pred, actions_gt):
    pred = actions_pred.detach().cpu().numpy()
    gt = actions_gt.detach().cpu().numpy()
    
    mse = np.mean((pred - gt) ** 2)
    mae = np.mean(np.abs(pred - gt))
    
    if pred.shape[-1] == 7:
        pos_error = np.mean(np.linalg.norm(pred[..., :3] - gt[..., :3], axis=-1))
        ori_error = np.mean(np.linalg.norm(pred[..., 3:7] - gt[..., 3:7], axis=-1))
        
        metrics = {
            'mse': mse,
            'mae': mae,
            'pos_error': pos_error,
            'ori_error': ori_error
        }
    else:
        metrics = {
            'mse': mse,
            'mae': mae
        }
    
    return metrics
