"""
Loss functions for VLA training
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict


class VLALoss(nn.Module):
    """
    Loss function for Vision-Language-Action model
    
    Computes losses for:
    - End-effector pose (6-DOF): L1 + L2 combined
    - Gripper action: Binary cross-entropy or L1
    
    Args:
        pose_weight: Weight for pose loss (default 1.0)
        gripper_weight: Weight for gripper loss (default 0.5)
        pose_loss_type: 'l1', 'l2', or 'smooth_l1' (default 'smooth_l1')
        gripper_loss_type: 'bce' or 'l1' (default 'l1')
    """
    
    def __init__(
        self,
        pose_weight: float = 1.0,
        gripper_weight: float = 0.5,
        pose_loss_type: str = "smooth_l1",
        gripper_loss_type: str = "l1",
    ):
        super().__init__()
        
        self.pose_weight = pose_weight
        self.gripper_weight = gripper_weight
        self.pose_loss_type = pose_loss_type
        self.gripper_loss_type = gripper_loss_type
        
        # Pose loss
        if pose_loss_type == "l1":
            self.pose_loss_fn = nn.L1Loss()
        elif pose_loss_type == "l2":
            self.pose_loss_fn = nn.MSELoss()
        elif pose_loss_type == "smooth_l1":
            self.pose_loss_fn = nn.SmoothL1Loss()
        else:
            raise ValueError(f"Unknown pose loss type: {pose_loss_type}")
            
        # Gripper loss
        if gripper_loss_type == "bce":
            self.gripper_loss_fn = nn.BCELoss()
        elif gripper_loss_type == "l1":
            self.gripper_loss_fn = nn.L1Loss()
        else:
            raise ValueError(f"Unknown gripper loss type: {gripper_loss_type}")
            
    def forward(
        self,
        predictions: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        Compute loss
        
        Args:
            predictions: Dict with keys:
                - "pose": [B, 6] - Predicted 6-DOF pose
                - "gripper": [B, 1] - Predicted gripper action
            targets: Dict with keys:
                - "pose": [B, 6] - Ground truth 6-DOF pose
                - "gripper": [B, 1] - Ground truth gripper action
                
        Returns:
            Dict with keys:
                - "total_loss": Total weighted loss
                - "pose_loss": Pose loss value
                - "gripper_loss": Gripper loss value
        """
        pred_pose = predictions["pose"]  # [B, 6]
        pred_gripper = predictions["gripper"]  # [B, 1]
        
        gt_pose = targets["pose"]  # [B, 6]
        gt_gripper = targets["gripper"]  # [B, 1]
        
        # Compute pose loss
        pose_loss = self.pose_loss_fn(pred_pose, gt_pose)
        
        # Compute gripper loss
        if self.gripper_loss_type == "bce":
            # Ensure gripper predictions are in [0, 1] for BCE
            pred_gripper_clamped = torch.clamp(pred_gripper, 0, 1)
            gt_gripper_clamped = torch.clamp(gt_gripper, 0, 1)
            gripper_loss = self.gripper_loss_fn(pred_gripper_clamped, gt_gripper_clamped)
        else:
            gripper_loss = self.gripper_loss_fn(pred_gripper, gt_gripper)
            
        # Compute total loss
        total_loss = (
            self.pose_weight * pose_loss +
            self.gripper_weight * gripper_loss
        )
        
        return {
            "total_loss": total_loss,
            "pose_loss": pose_loss,
            "gripper_loss": gripper_loss,
        }
    
    def compute_metrics(
        self,
        predictions: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor]
    ) -> Dict[str, float]:
        """
        Compute evaluation metrics (without gradients)
        
        Returns:
            Dict with metric names and values
        """
        with torch.no_grad():
            pred_pose = predictions["pose"]
            pred_gripper = predictions["gripper"]
            
            gt_pose = targets["pose"]
            gt_gripper = targets["gripper"]
            
            # Pose metrics
            pose_error = torch.norm(pred_pose - gt_pose, dim=-1)  # [B]
            pose_l1_error = torch.abs(pred_pose - gt_pose).mean(dim=-1)  # [B]
            
            # Gripper metrics
            gripper_error = torch.abs(pred_gripper - gt_gripper).squeeze(-1)  # [B]
            
            # Gripper accuracy (for binary classification)
            if self.gripper_loss_type == "bce":
                pred_gripper_binary = (pred_gripper > 0.5).float()
                gt_gripper_binary = (gt_gripper > 0.5).float()
                gripper_accuracy = (pred_gripper_binary == gt_gripper_binary).float().mean()
            else:
                gripper_accuracy = None
                
            metrics = {
                "pose_l2_error": pose_error.mean().item(),
                "pose_l1_error": pose_l1_error.mean().item(),
                "gripper_error": gripper_error.mean().item(),
            }
            
            if gripper_accuracy is not None:
                metrics["gripper_accuracy"] = gripper_accuracy.item()
                
            return metrics
