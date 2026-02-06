"""
Action Prediction Head
Predicts robot actions (end-effector pose + gripper) from fused features

改进：
- 支持四元数表示旋转（避免欧拉角的奇异性问题）
- 支持传统的欧拉角表示（向后兼容）
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ActionHead(nn.Module):
    """
    动作预测头
    
    功能：
    - 预测末端执行器姿态（6-DOF或7-DOF with quaternion）
    - 预测夹爪动作
    
    输出格式：
    - 欧拉角模式: [x, y, z, roll, pitch, yaw, gripper] (7维)
    - 四元数模式: [x, y, z, qw, qx, qy, qz, gripper] (8维)
    
    Args:
        input_dim: 输入特征维度
        action_dim: 动作维度（7 for 欧拉角，8 for 四元数）
        hidden_dim: 隐藏层维度
        use_discrete_gripper: 是否使用离散夹爪动作
        use_quaternion: 是否使用四元数表示旋转（推荐）
    """
    
    def __init__(self, input_dim=1024, action_dim=7, hidden_dim=512, 
                 use_discrete_gripper=False, use_quaternion=False):
        super().__init__()
        
        self.action_dim = action_dim
        self.use_discrete_gripper = use_discrete_gripper
        self.use_quaternion = use_quaternion
        
        # 根据旋转表示方式调整输出维度
        if use_quaternion:
            # 四元数模式：位置(3) + 四元数(4) + 夹爪(1) = 8维
            self.pose_output_dim = 7  # 3 pos + 4 quat
            self.expected_action_dim = 8
        else:
            # 欧拉角模式：位置(3) + 欧拉角(3) + 夹爪(1) = 7维
            self.pose_output_dim = 6  # 3 pos + 3 euler
            self.expected_action_dim = 7
        
        if action_dim != self.expected_action_dim:
            print(f"警告: action_dim={action_dim} 与 use_quaternion={use_quaternion} 不匹配")
            print(f"  期望维度: {self.expected_action_dim}")
        
        # Shared feature extraction
        self.shared_layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU()
        )
        
        # 末端执行器姿态预测头
        self.pose_head = nn.Sequential(
            nn.Linear(hidden_dim // 2, 256),
            nn.ReLU(),
            nn.Linear(256, self.pose_output_dim)  # 6 for 欧拉角，7 for 四元数
        )
        
        # Gripper head
        if use_discrete_gripper:
            # Discrete: binary classification
            self.gripper_head = nn.Sequential(
                nn.Linear(hidden_dim // 2, 256),
                nn.ReLU(),
                nn.Linear(256, 2),  # Binary classification
                nn.LogSoftmax(dim=-1)
            )
        else:
            # Continuous: regression with sigmoid
            self.gripper_head = nn.Sequential(
                nn.Linear(hidden_dim // 2, 256),
                nn.ReLU(),
                nn.Linear(256, 1),
                nn.Sigmoid()  # Output in [0, 1]
            )
        
    def forward(self, fused_features):
        """
        前向传播
        
        Args:
            fused_features: [B, input_dim] - 融合后的多模态特征
            
        Returns:
            dict包含:
                - action: [B, action_dim] - 预测的完整动作
                - pose: [B, 6或7] - 末端执行器姿态
                - gripper: [B, 1] - 夹爪动作
                - quaternion: [B, 4] - 四元数（如果use_quaternion=True）
        """
        # 共享特征提取
        shared_feat = self.shared_layers(fused_features)  # [B, hidden_dim // 2]
        
        # 预测姿态
        pose_raw = self.pose_head(shared_feat)  # [B, pose_output_dim]
        
        if self.use_quaternion:
            # 四元数模式
            pos = pose_raw[:, :3]  # [B, 3] - 位置
            quat_raw = pose_raw[:, 3:7]  # [B, 4] - 未归一化的四元数
            
            # 归一化四元数（确保是单位四元数）
            quat = F.normalize(quat_raw, p=2, dim=-1)  # [B, 4]
            
            pose = torch.cat([pos, quat], dim=-1)  # [B, 7]
        else:
            # 欧拉角模式
            pose = pose_raw  # [B, 6]
            quat = None
        
        # 预测夹爪
        if self.use_discrete_gripper:
            # 离散模式：二分类
            gripper_logits = self.gripper_head(shared_feat)  # [B, 2]
            gripper = F.softmax(gripper_logits, dim=-1)[:, 1:2]  # [B, 1] (打开的概率)
            action = torch.cat([pose, gripper], dim=-1)  # [B, action_dim]
        else:
            # 连续模式：回归（sigmoid确保在[0,1]）
            gripper = self.gripper_head(shared_feat)  # [B, 1]
            action = torch.cat([pose, gripper], dim=-1)  # [B, action_dim]
        
        result = {
            "action": action,
            "pose": pose,
            "gripper": gripper
        }
        
        if self.use_quaternion and quat is not None:
            result["quaternion"] = quat
        
        return result
    
    def quaternion_to_euler(self, quaternion: torch.Tensor) -> torch.Tensor:
        """
        将四元数转换为欧拉角（用于兼容性）
        
        Args:
            quaternion: [B, 4] - 四元数 [qw, qx, qy, qz]
            
        Returns:
            euler: [B, 3] - 欧拉角 [roll, pitch, yaw]
        """
        qw, qx, qy, qz = quaternion[:, 0], quaternion[:, 1], quaternion[:, 2], quaternion[:, 3]
        
        # Roll (x-axis rotation)
        sinr_cosp = 2 * (qw * qx + qy * qz)
        cosr_cosp = 1 - 2 * (qx * qx + qy * qy)
        roll = torch.atan2(sinr_cosp, cosr_cosp)
        
        # Pitch (y-axis rotation)
        sinp = 2 * (qw * qy - qz * qx)
        sinp = torch.clamp(sinp, -1.0, 1.0)
        pitch = torch.asin(sinp)
        
        # Yaw (z-axis rotation)
        siny_cosp = 2 * (qw * qz + qx * qy)
        cosy_cosp = 1 - 2 * (qy * qy + qz * qz)
        yaw = torch.atan2(siny_cosp, cosy_cosp)
        
        return torch.stack([roll, pitch, yaw], dim=-1)
