"""
动作归一化模块
用于归一化机器人动作，解决不同动作维度尺度差异大的问题
"""

import torch
import torch.nn as nn
import numpy as np
import os
from typing import Dict, Optional, Tuple


class ActionNormalizer:
    """
    动作归一化器
    
    功能：
    - 分别归一化位置(x,y,z)和旋转(roll,pitch,yaw)
    - 支持从统计数据文件加载或自动计算统计信息
    - 提供归一化和反归一化功能
    """
    
    def __init__(
        self,
        stats_path: Optional[str] = None,
        pos_mean: Optional[torch.Tensor] = None,
        pos_std: Optional[torch.Tensor] = None,
        rot_mean: Optional[torch.Tensor] = None,
        rot_std: Optional[torch.Tensor] = None,
        use_quaternion: bool = False,
    ):
        """
        Args:
            stats_path: 统计信息文件路径（.pt格式）
            pos_mean: 位置均值 [3]
            pos_std: 位置标准差 [3]
            rot_mean: 旋转均值 [3或4] (欧拉角或四元数)
            rot_std: 旋转标准差 [3或4] (欧拉角或四元数)
            use_quaternion: 是否使用四元数表示旋转
        """
        self.use_quaternion = use_quaternion
        rot_dim = 4 if use_quaternion else 3
        if stats_path and os.path.exists(stats_path):
            # 从文件加载统计信息
            stats = torch.load(stats_path)
            self.pos_mean = stats['pos_mean']
            self.pos_std = stats['pos_std']
            self.rot_mean = stats.get('rot_mean', torch.zeros(rot_dim))
            self.rot_std = stats.get('rot_std', torch.ones(rot_dim))
        elif pos_mean is not None and pos_std is not None:
            # 使用提供的统计信息
            self.pos_mean = pos_mean
            self.pos_std = pos_std
            self.rot_mean = rot_mean if rot_mean is not None else torch.zeros(rot_dim)
            self.rot_std = rot_std if rot_std is not None else torch.ones(rot_dim)
        else:
            # 使用默认值（不归一化）
            self.pos_mean = torch.zeros(3)
            self.pos_std = torch.ones(3)
            self.rot_mean = torch.zeros(rot_dim)
            self.rot_std = torch.ones(rot_dim)
    
    def normalize(self, action: torch.Tensor) -> torch.Tensor:
        """
        归一化动作
        
        Args:
            action: [B, 7或8] 或 [7或8] - 原始动作
                    欧拉角: [x, y, z, roll, pitch, yaw, gripper]
                    四元数: [x, y, z, qw, qx, qy, qz, gripper]
            
        Returns:
            normalized_action: [B, 7或8] 或 [7或8] - 归一化后的动作
        """
        device = action.device
        pos_mean = self.pos_mean.to(device)
        pos_std = self.pos_std.to(device)
        rot_mean = self.rot_mean.to(device)
        rot_std = self.rot_std.to(device)
        
        # 分离各个部分
        if self.use_quaternion:
            # 四元数模式: [x, y, z, qw, qx, qy, qz, gripper]
            pos = action[..., :3]       # 位置
            rot = action[..., 3:7]      # 四元数
            gripper = action[..., 7:8]  # 夹爪
        else:
            # 欧拉角模式: [x, y, z, roll, pitch, yaw, gripper]
            pos = action[..., :3]       # 位置
            rot = action[..., 3:6]      # 欧拉角
            gripper = action[..., 6:7]  # 夹爪
        
        # 归一化位置和旋转
        pos_norm = (pos - pos_mean) / (pos_std + 1e-8)
        rot_norm = (rot - rot_mean) / (rot_std + 1e-8)
        
        # 拼接
        normalized = torch.cat([pos_norm, rot_norm, gripper], dim=-1)
        return normalized
    
    def denormalize(self, action: torch.Tensor) -> torch.Tensor:
        """
        反归一化动作
        
        Args:
            action: [B, 7或8] 或 [7或8] - 归一化后的动作
            
        Returns:
            denormalized_action: [B, 7或8] 或 [7或8] - 原始尺度的动作
        """
        device = action.device
        pos_mean = self.pos_mean.to(device)
        pos_std = self.pos_std.to(device)
        rot_mean = self.rot_mean.to(device)
        rot_std = self.rot_std.to(device)
        
        # 分离各个部分
        if self.use_quaternion:
            # 四元数模式
            pos_norm = action[..., :3]
            rot_norm = action[..., 3:7]
            gripper = action[..., 7:8]
        else:
            # 欧拉角模式
            pos_norm = action[..., :3]
            rot_norm = action[..., 3:6]
            gripper = action[..., 6:7]
        
        # 反归一化
        pos = pos_norm * pos_std + pos_mean
        rot = rot_norm * rot_std + rot_mean
        
        # 拼接
        denormalized = torch.cat([pos, rot, gripper], dim=-1)
        return denormalized
    
    def compute_stats(self, actions: np.ndarray) -> Dict[str, torch.Tensor]:
        """
        从动作数据计算统计信息
        
        Args:
            actions: [N, 7] - 动作数组
            
        Returns:
            stats: 包含统计信息的字典
        """
        actions_tensor = torch.from_numpy(actions).float()
        
        pos = actions_tensor[:, :3]
        rot = actions_tensor[:, 3:6]
        
        stats = {
            'pos_mean': pos.mean(dim=0),
            'pos_std': pos.std(dim=0),
            'rot_mean': rot.mean(dim=0),
            'rot_std': rot.std(dim=0),
        }
        
        # 避免标准差为0
        stats['pos_std'] = torch.clamp(stats['pos_std'], min=1e-6)
        stats['rot_std'] = torch.clamp(stats['rot_std'], min=1e-6)
        
        return stats
    
    def save_stats(self, save_path: str):
        """保存统计信息到文件"""
        stats = {
            'pos_mean': self.pos_mean,
            'pos_std': self.pos_std,
            'rot_mean': self.rot_mean,
            'rot_std': self.rot_std,
        }
        torch.save(stats, save_path)
        print(f"动作统计信息已保存到: {save_path}")
