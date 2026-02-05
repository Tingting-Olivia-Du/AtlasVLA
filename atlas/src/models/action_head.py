"""
Action Prediction Head
Predicts robot actions (end-effector pose + gripper) from fused features
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ActionHead(nn.Module):
    """
    Predict robot actions from fused multimodal features
    
    Output format:
    - End-effector pose: 6-DOF [x, y, z, roll, pitch, yaw]
    - Gripper action: continuous [0, 1] or discrete {0, 1}
    
    Total action dimension: 7 (6 pose + 1 gripper)
    """
    
    def __init__(self, input_dim=1024, action_dim=7, hidden_dim=512, 
                 use_discrete_gripper=False):
        super().__init__()
        
        self.action_dim = action_dim
        self.use_discrete_gripper = use_discrete_gripper
        
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
        
        # End-effector pose head (6-DOF)
        self.pose_head = nn.Sequential(
            nn.Linear(hidden_dim // 2, 256),
            nn.ReLU(),
            nn.Linear(256, 6)  # [x, y, z, roll, pitch, yaw]
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
        Args:
            fused_features: [B, input_dim] - Fused multimodal features
            
        Returns:
            action: [B, action_dim] - Predicted action
            pose: [B, 6] - End-effector pose
            gripper: [B, 1] or [B, 2] - Gripper action
        """
        # Shared feature extraction
        shared_feat = self.shared_layers(fused_features)  # [B, hidden_dim // 2]
        
        # Predict pose
        pose = self.pose_head(shared_feat)  # [B, 6]
        
        # Predict gripper
        if self.use_discrete_gripper:
            gripper_logits = self.gripper_head(shared_feat)  # [B, 2]
            gripper = F.softmax(gripper_logits, dim=-1)[:, 1:2]  # [B, 1] (probability of open)
            action = torch.cat([pose, gripper], dim=-1)  # [B, 7]
        else:
            gripper = self.gripper_head(shared_feat)  # [B, 1]
            action = torch.cat([pose, gripper], dim=-1)  # [B, 7]
        
        return {
            "action": action,
            "pose": pose,
            "gripper": gripper
        }
