"""
Loss functions
"""
import torch
import torch.nn.functional as F

#loss = F.mse_loss(actions_pred, actions_gt)
def action_loss_fn(actions_pred, actions_gt, config):
    # Supports mixed loss:
    #   MSE + Huber + temporal smoothness + gripper weighting.
    cfg = getattr(config, "action_head", config)
    mse_w = float(getattr(cfg, "loss_mse_weight", 0.5))
    huber_w = float(getattr(cfg, "loss_huber_weight", 0.5))
    huber_delta = float(getattr(cfg, "loss_huber_delta", 1.0))
    smooth_w = float(getattr(cfg, "loss_smooth_weight", 0.02))
    gripper_w = float(getattr(cfg, "loss_gripper_weight", 2.0))
    smooth_exclude_gripper = bool(getattr(cfg, "loss_smooth_exclude_gripper", True))

    # Ensure shape [B, H, D] for unified handling.
    if actions_pred.dim() == 2:
        actions_pred = actions_pred.unsqueeze(1)
    if actions_gt.dim() == 2:
        actions_gt = actions_gt.unsqueeze(1)

    # Weighted regression by action dimension (up-weight gripper: last dim).
    dim_weight = torch.ones(actions_pred.size(-1), device=actions_pred.device, dtype=actions_pred.dtype)
    if actions_pred.size(-1) >= 1:
        dim_weight[-1] = gripper_w
    dim_weight = dim_weight.view(1, 1, -1)

    diff = actions_pred - actions_gt
    mse_term = (diff.pow(2) * dim_weight).mean()
    huber_term = F.huber_loss(actions_pred, actions_gt, delta=huber_delta, reduction="none")
    huber_term = (huber_term * dim_weight).mean()

    # Temporal smoothness on prediction chunk.
    smooth_term = actions_pred.new_tensor(0.0)
    if actions_pred.size(1) > 1:
        pred_for_smooth = actions_pred[..., :-1] if smooth_exclude_gripper and actions_pred.size(-1) > 1 else actions_pred
        smooth_term = (pred_for_smooth[:, 1:, :] - pred_for_smooth[:, :-1, :]).abs().mean()

    loss = mse_w * mse_term + huber_w * huber_term + smooth_w * smooth_term
    return loss
