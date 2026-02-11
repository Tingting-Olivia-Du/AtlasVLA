"""
Loss functions
"""
import torch
import torch.nn.functional as F


def action_loss_fn(actions_pred, actions_gt, config):
    loss = F.mse_loss(actions_pred, actions_gt)
    return loss
