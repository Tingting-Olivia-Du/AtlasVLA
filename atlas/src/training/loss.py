"""
VLA训练损失函数 (Enhanced for Flow Matching)

支持:
1. Flow Matching Action Head (主要)
2. 传统 Action Head (兼容)
3. Diffusion Action Head (兼容)

改进:
- 自动检测 action head 类型
- 支持 action chunking loss
- 辅助损失（几何一致性、特征正则化）
- 详细的评估指标
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Literal
import logging


class VLALoss(nn.Module):
    """
    Vision-Language-Action模型损失函数
    
    支持多种 action head:
    1. Flow Matching: loss 在 action head 内部计算
    2. MLP/Regression: 使用此 loss module
    3. Diffusion: loss 在 action head 内部计算
    
    Args:
        action_head_type: "flow_matching", "mlp", "diffusion"
        pose_weight: 姿态损失权重
        gripper_weight: 夹爪损失权重
        pose_loss_type: 'l1', 'l2', 'smooth_l1'
        gripper_loss_type: 'l1', 'bce'
        use_auxiliary_loss: 是否使用辅助损失
        geom_consistency_weight: 几何一致性损失权重
        feature_reg_weight: 特征正则化损失权重
        temporal_consistency_weight: 时序一致性损失权重（for action chunking）
    """
    
    def __init__(
        self,
        action_head_type: Literal["flow_matching", "mlp", "diffusion"] = "flow_matching",
        # 基础损失权重
        pose_weight: float = 1.0,
        gripper_weight: float = 0.5,
        pose_loss_type: str = "smooth_l1",
        gripper_loss_type: str = "l1",
        # 辅助损失
        use_auxiliary_loss: bool = False,
        geom_consistency_weight: float = 0.1,
        feature_reg_weight: float = 0.01,
        temporal_consistency_weight: float = 0.05,  # 新增：时序一致性
    ):
        super().__init__()
        
        self.action_head_type = action_head_type
        self.pose_weight = pose_weight
        self.gripper_weight = gripper_weight
        self.pose_loss_type = pose_loss_type
        self.gripper_loss_type = gripper_loss_type
        
        # 辅助损失
        self.use_auxiliary_loss = use_auxiliary_loss
        self.geom_consistency_weight = geom_consistency_weight
        self.feature_reg_weight = feature_reg_weight
        self.temporal_consistency_weight = temporal_consistency_weight
        
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
                Flow Matching / Diffusion:
                    - "loss": scalar - action head 内部计算的 loss
                    - "action_chunk": [B, T, 7] (可选，用于计算额外指标)
                MLP:
                    - "pose": [B, 6] or [B, T, 6]
                    - "gripper": [B, 1] or [B, T, 1]
                    - "action": [B, 7] or [B, T, 7]
                    
            targets: 包含真实值的字典
                - "pose": [B, 6] or [B, T, 6]
                - "gripper": [B, 1] or [B, T, 1]
                - "action": [B, 7] or [B, T, 7] (可选)
                
            intermediates: 可选的中间特征
                - "geometry_features": [B, S, D]
                - "fused_features": [B, D]
                - "velocity_pred": [B, T, A] (Flow Matching)
                - "velocity_target": [B, T, A] (Flow Matching)
                
        Returns:
            包含损失值的字典
        """
        # ============================================================
        # 1. Action Head Loss
        # ============================================================
        if self.action_head_type in ["flow_matching", "diffusion"]:
            # Flow Matching / Diffusion: loss 已在 action head 内部计算
            if "loss" in predictions:
                action_loss = predictions["loss"]
            else:
                raise ValueError(
                    f"{self.action_head_type} action head should return 'loss' in predictions"
                )
            pose_loss = torch.tensor(0.0, device=action_loss.device)
            gripper_loss = torch.tensor(0.0, device=action_loss.device)
            
        else:
            # MLP / Traditional: 计算 pose + gripper loss
            pred_pose = predictions["pose"]
            pred_gripper = predictions["gripper"]
            gt_pose = targets["pose"]
            gt_gripper = targets["gripper"]
            
            # 处理 action chunking: [B, T, D] → 只用第一步 [B, D]
            if pred_pose.dim() == 3:
                pred_pose = pred_pose[:, 0, :]
            if pred_gripper.dim() == 3:
                pred_gripper = pred_gripper[:, 0, :]
            if gt_pose.dim() == 3:
                gt_pose = gt_pose[:, 0, :]
            if gt_gripper.dim() == 3:
                gt_gripper = gt_gripper[:, 0, :]
            
            # 计算损失
            pose_loss = self.pose_loss_fn(pred_pose, gt_pose)
            
            if self.gripper_loss_type == "bce":
                pred_gripper_clamped = torch.clamp(pred_gripper, 0, 1)
                gt_gripper_clamped = torch.clamp(gt_gripper, 0, 1)
                gripper_loss = self.gripper_loss_fn(pred_gripper_clamped, gt_gripper_clamped)
            else:
                gripper_loss = self.gripper_loss_fn(pred_gripper, gt_gripper)
            
            action_loss = self.pose_weight * pose_loss + self.gripper_weight * gripper_loss
        
        # ============================================================
        # 2. Auxiliary Losses
        # ============================================================
        auxiliary_loss = torch.tensor(0.0, device=action_loss.device)
        
        if self.use_auxiliary_loss and intermediates is not None:
            # 2.1 几何一致性损失
            if "geometry_features" in intermediates:
                geom_feat = intermediates["geometry_features"]  # [B, S, D]
                if geom_feat.dim() == 3 and geom_feat.shape[1] > 1:
                    # 鼓励不同帧的几何特征一致
                    geom_mean = geom_feat.mean(dim=1, keepdim=True)
                    geom_consistency = F.mse_loss(geom_feat, geom_mean.expand_as(geom_feat))
                    auxiliary_loss += self.geom_consistency_weight * geom_consistency
            
            # 2.2 特征正则化损失
            if "fused_features" in intermediates:
                fused_feat = intermediates["fused_features"]  # [B, D]
                feature_reg = fused_feat.norm(dim=-1).mean()
                auxiliary_loss += self.feature_reg_weight * feature_reg
            
            # 2.3 时序一致性损失（for action chunking）
            if "action_chunk" in predictions:
                action_chunk = predictions["action_chunk"]  # [B, T, A]
                if action_chunk.dim() == 3 and action_chunk.shape[1] > 1:
                    # 鼓励连续动作平滑
                    action_diff = action_chunk[:, 1:, :] - action_chunk[:, :-1, :]
                    temporal_consistency = action_diff.pow(2).mean()
                    auxiliary_loss += self.temporal_consistency_weight * temporal_consistency
        
        # ============================================================
        # 3. Total Loss
        # ============================================================
        total_loss = action_loss + auxiliary_loss
        
        # ============================================================
        # 4. Return
        # ============================================================
        result = {
            "total_loss": total_loss,
            "action_loss": action_loss,
        }
        
        # 添加详细的 loss 分解
        if self.action_head_type == "mlp":
            result["pose_loss"] = pose_loss
            result["gripper_loss"] = gripper_loss
        
        if self.use_auxiliary_loss:
            result["auxiliary_loss"] = auxiliary_loss
        
        return result
    
    def compute_metrics(
        self,
        predictions: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor]
    ) -> Dict[str, float]:
        """
        计算评估指标 (without gradients)
        
        Returns:
            Dict with metric names and values
        """
        with torch.no_grad():
            metrics = {}
            
            # 获取预测和真实动作
            if "action" in predictions:
                pred_action = predictions["action"]
            elif "action_chunk" in predictions:
                pred_action = predictions["action_chunk"][:, 0, :]  # 只用第一步
            else:
                # 手动拼接 pose + gripper
                pred_pose = predictions.get("pose", None)
                pred_gripper = predictions.get("gripper", None)
                if pred_pose is not None and pred_gripper is not None:
                    if pred_pose.dim() == 3:
                        pred_pose = pred_pose[:, 0, :]
                    if pred_gripper.dim() == 3:
                        pred_gripper = pred_gripper[:, 0, :]
                    pred_action = torch.cat([pred_pose, pred_gripper], dim=-1)
                else:
                    return metrics
            
            if "action" in targets:
                gt_action = targets["action"]
            else:
                gt_pose = targets.get("pose", None)
                gt_gripper = targets.get("gripper", None)
                if gt_pose is not None and gt_gripper is not None:
                    if gt_pose.dim() == 3:
                        gt_pose = gt_pose[:, 0, :]
                    if gt_gripper.dim() == 3:
                        gt_gripper = gt_gripper[:, 0, :]
                    gt_action = torch.cat([gt_pose, gt_gripper], dim=-1)
                else:
                    return metrics
            
            # Split into pose and gripper
            pred_pose = pred_action[:, :6]
            pred_gripper = pred_action[:, 6:7]
            gt_pose = gt_action[:, :6]
            gt_gripper = gt_action[:, 6:7]
            
            # Pose metrics
            pose_error_l2 = torch.norm(pred_pose - gt_pose, dim=-1).mean()
            pose_error_l1 = torch.abs(pred_pose - gt_pose).mean()
            
            # Position error (前3维)
            position_error = torch.norm(pred_pose[:, :3] - gt_pose[:, :3], dim=-1).mean()
            
            # Orientation error (后3维，假设是欧拉角)
            orientation_error = torch.norm(pred_pose[:, 3:6] - gt_pose[:, 3:6], dim=-1).mean()
            
            # Gripper metrics
            gripper_error = torch.abs(pred_gripper - gt_gripper).mean()
            
            # Gripper accuracy (binary)
            pred_gripper_binary = (pred_gripper > 0.5).float()
            gt_gripper_binary = (gt_gripper > 0.5).float()
            gripper_accuracy = (pred_gripper_binary == gt_gripper_binary).float().mean()
            
            metrics.update({
                "pose_l2_error": pose_error_l2.item(),
                "pose_l1_error": pose_error_l1.item(),
                "position_error": position_error.item(),
                "orientation_error": orientation_error.item(),
                "gripper_error": gripper_error.item(),
                "gripper_accuracy": gripper_accuracy.item(),
            })
            
            # Action chunking metrics (如果有完整序列)
            if "action_chunk" in predictions:
                action_chunk = predictions["action_chunk"]
                if action_chunk.shape[1] > 1:
                    # 计算时序平滑度
                    action_diff = action_chunk[:, 1:, :] - action_chunk[:, :-1, :]
                    smoothness = action_diff.pow(2).mean()
                    metrics["action_smoothness"] = smoothness.item()
            
            return metrics


class FlowMatchingLossWrapper(nn.Module):
    """
    Flow Matching Loss Wrapper
    
    当使用 Flow Matching action head 时，loss 主要在 action head 内部计算
    这个 wrapper 只负责添加辅助损失和计算 metrics
    """
    
    def __init__(
        self,
        use_auxiliary_loss: bool = False,
        geom_consistency_weight: float = 0.1,
        feature_reg_weight: float = 0.01,
        temporal_consistency_weight: float = 0.05,
    ):
        super().__init__()
        self.use_auxiliary_loss = use_auxiliary_loss
        self.geom_consistency_weight = geom_consistency_weight
        self.feature_reg_weight = feature_reg_weight
        self.temporal_consistency_weight = temporal_consistency_weight
    
    def forward(
        self,
        flow_matching_loss: torch.Tensor,  # 来自 action head 的 loss
        predictions: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
        intermediates: Optional[Dict[str, torch.Tensor]] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            flow_matching_loss: action head 返回的 flow matching loss
            predictions: {"action_chunk": [B, T, 7], ...}
            targets: {"action": [B, T, 7], ...}
            intermediates: {"geometry_features": [...], ...}
        """
        total_loss = flow_matching_loss
        auxiliary_loss = torch.tensor(0.0, device=total_loss.device)
        
        if self.use_auxiliary_loss and intermediates is not None:
            # 几何一致性
            if "geometry_features" in intermediates:
                geom_feat = intermediates["geometry_features"]
                if geom_feat.dim() == 3 and geom_feat.shape[1] > 1:
                    geom_mean = geom_feat.mean(dim=1, keepdim=True)
                    geom_consistency = F.mse_loss(geom_feat, geom_mean.expand_as(geom_feat))
                    auxiliary_loss += self.geom_consistency_weight * geom_consistency
            
            # 特征正则化
            if "fused_features" in intermediates:
                fused_feat = intermediates["fused_features"]
                feature_reg = fused_feat.norm(dim=-1).mean()
                auxiliary_loss += self.feature_reg_weight * feature_reg
            
            # 时序一致性
            if "action_chunk" in predictions:
                action_chunk = predictions["action_chunk"]
                if action_chunk.dim() == 3 and action_chunk.shape[1] > 1:
                    action_diff = action_chunk[:, 1:, :] - action_chunk[:, :-1, :]
                    temporal_consistency = action_diff.pow(2).mean()
                    auxiliary_loss += self.temporal_consistency_weight * temporal_consistency
            
            total_loss = total_loss + auxiliary_loss
        
        return {
            "total_loss": total_loss,
            "flow_matching_loss": flow_matching_loss,
            "auxiliary_loss": auxiliary_loss,
        }
    
    def compute_metrics(
        self,
        predictions: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor]
    ) -> Dict[str, float]:
        """计算评估指标"""
        with torch.no_grad():
            metrics = {}
            
            # 获取预测动作 (第一步)
            if "action_chunk" in predictions:
                pred_action = predictions["action_chunk"][:, 0, :]  # [B, 7]
            elif "action" in predictions:
                pred_action = predictions["action"]
            else:
                return metrics
            
            # 获取真实动作
            if "action" in targets:
                gt_action = targets["action"]
                if gt_action.dim() == 3:
                    gt_action = gt_action[:, 0, :]
            else:
                return metrics
            
            # Split
            pred_pose = pred_action[:, :6]
            pred_gripper = pred_action[:, 6:7]
            gt_pose = gt_action[:, :6]
            gt_gripper = gt_action[:, 6:7]
            
            # Metrics
            metrics["pose_l2_error"] = torch.norm(pred_pose - gt_pose, dim=-1).mean().item()
            metrics["pose_l1_error"] = torch.abs(pred_pose - gt_pose).mean().item()
            metrics["position_error"] = torch.norm(pred_pose[:, :3] - gt_pose[:, :3], dim=-1).mean().item()
            metrics["orientation_error"] = torch.norm(pred_pose[:, 3:] - gt_pose[:, 3:], dim=-1).mean().item()
            metrics["gripper_error"] = torch.abs(pred_gripper - gt_gripper).mean().item()
            
            pred_gripper_binary = (pred_gripper > 0.5).float()
            gt_gripper_binary = (gt_gripper > 0.5).float()
            metrics["gripper_accuracy"] = (pred_gripper_binary == gt_gripper_binary).float().mean().item()
            
            # Smoothness
            if "action_chunk" in predictions:
                action_chunk = predictions["action_chunk"]
                if action_chunk.shape[1] > 1:
                    action_diff = action_chunk[:, 1:, :] - action_chunk[:, :-1, :]
                    metrics["action_smoothness"] = action_diff.pow(2).mean().item()
            
            return metrics