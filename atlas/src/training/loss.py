"""
VLA训练损失函数

改进6: 添加辅助损失
- 几何一致性损失：鼓励几何特征的一致性
- 特征正则化损失：防止特征过度拟合
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional


class VLALoss(nn.Module):
    """
    Vision-Language-Action模型损失函数
    
    计算损失：
    - 末端执行器姿态（6-DOF）：L1/L2/SmoothL1
    - 夹爪动作：L1或BCE
    - 改进6: 辅助损失（可选）
        - 几何一致性损失
        - 特征正则化损失
    
    Args:
        pose_weight: 姿态损失权重（默认1.0）
        gripper_weight: 夹爪损失权重（默认0.5）
        pose_loss_type: 'l1', 'l2', 或 'smooth_l1'（默认'smooth_l1'）
        gripper_loss_type: 'bce' 或 'l1'（默认'l1'）
        use_auxiliary_loss: 是否使用辅助损失（改进6）
        geom_consistency_weight: 几何一致性损失权重（默认0.1）
        feature_reg_weight: 特征正则化损失权重（默认0.01）
    """
    
    def __init__(
        self,
        pose_weight: float = 1.0,
        gripper_weight: float = 0.5,
        pose_loss_type: str = "smooth_l1",
        gripper_loss_type: str = "l1",
        # 改进6: 辅助损失参数
        use_auxiliary_loss: bool = False,
        geom_consistency_weight: float = 0.1,
        feature_reg_weight: float = 0.01,
    ):
        super().__init__()
        
        self.pose_weight = pose_weight
        self.gripper_weight = gripper_weight
        self.pose_loss_type = pose_loss_type
        self.gripper_loss_type = gripper_loss_type
        
        # 改进6: 辅助损失开关和权重
        self.use_auxiliary_loss = use_auxiliary_loss
        self.geom_consistency_weight = geom_consistency_weight
        self.feature_reg_weight = feature_reg_weight
        
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
        targets: Dict[str, torch.Tensor],
        intermediates: Optional[Dict[str, torch.Tensor]] = None
    ) -> Dict[str, torch.Tensor]:
        """
        计算损失
        
        Args:
            predictions: 包含预测值的字典
                - "pose": [B, 6] - 预测的6-DOF姿态
                - "gripper": [B, 1] - 预测的夹爪动作
            targets: 包含真实值的字典
                - "pose": [B, 6] - 真实6-DOF姿态
                - "gripper": [B, 1] - 真实夹爪动作
            intermediates: 可选的中间特征（改进6）
                - "geometry_features": [B, S, D] - 几何特征
                - "language_features": [B, L, D] - 语言特征
                - "fused_features": [B, D] - 融合特征
                
        Returns:
            包含损失值的字典：
                - "total_loss": 总加权损失
                - "pose_loss": 姿态损失值
                - "gripper_loss": 夹爪损失值
                - "auxiliary_loss": 辅助损失值（如果启用）
        """
        pred_pose = predictions["pose"]  # [B, 6]
        pred_gripper = predictions["gripper"]  # [B, 1]
        
        gt_pose = targets["pose"]  # [B, 6]
        gt_gripper = targets["gripper"]  # [B, 1]
        
        # 计算姿态损失
        pose_loss = self.pose_loss_fn(pred_pose, gt_pose)
        
        # 计算夹爪损失
        if self.gripper_loss_type == "bce":
            # 确保夹爪预测在[0, 1]范围内用于BCE
            pred_gripper_clamped = torch.clamp(pred_gripper, 0, 1)
            gt_gripper_clamped = torch.clamp(gt_gripper, 0, 1)
            gripper_loss = self.gripper_loss_fn(pred_gripper_clamped, gt_gripper_clamped)
        else:
            gripper_loss = self.gripper_loss_fn(pred_gripper, gt_gripper)
        
        # 改进6: 计算辅助损失
        auxiliary_loss = 0.0
        if self.use_auxiliary_loss and intermediates is not None:
            # 几何一致性损失：鼓励几何特征的一致性
            if "geometry_features" in intermediates:
                geom_feat = intermediates["geometry_features"]  # [B, S, D]
                if geom_feat.shape[1] > 1:
                    # 计算不同帧之间的特征差异（鼓励一致性）
                    geom_mean = geom_feat.mean(dim=1, keepdim=True)  # [B, 1, D]
                    geom_consistency = F.mse_loss(geom_feat, geom_mean.expand_as(geom_feat))
                    auxiliary_loss += self.geom_consistency_weight * geom_consistency
            
            # 特征正则化损失：防止特征过度拟合
            if "fused_features" in intermediates:
                fused_feat = intermediates["fused_features"]  # [B, D]
                # L2正则化
                feature_reg = fused_feat.norm(dim=-1).mean()
                auxiliary_loss += self.feature_reg_weight * feature_reg
        
        # 计算总损失
        total_loss = (
            self.pose_weight * pose_loss +
            self.gripper_weight * gripper_loss +
            auxiliary_loss
        )
        
        result = {
            "total_loss": total_loss,
            "pose_loss": pose_loss,
            "gripper_loss": gripper_loss,
        }
        
        if self.use_auxiliary_loss:
            result["auxiliary_loss"] = auxiliary_loss
        
        return result
    
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
