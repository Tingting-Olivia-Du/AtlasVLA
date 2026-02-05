"""
PointNet encoder for 3D point cloud processing
Simple implementation for extracting features from point clouds
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class PointNetEncoder(nn.Module):
    """
    Simple PointNet encoder for 3D point cloud feature extraction
    
    Processes point clouds to extract global features
    """
    
    def __init__(self, output_dim=256):
        super().__init__()
        self.conv1 = nn.Conv1d(3, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, output_dim, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(output_dim)
        
    def forward(self, points):
        """
        Args:
            points: [B, S, H, W, 3] or [B, N, 3] - Point cloud coordinates
        Returns:
            features: [B, S, output_dim] or [B, output_dim] - Global features
        """
        original_shape = points.shape
        
        # Handle different input shapes
        if len(points.shape) == 5:
            # [B, S, H, W, 3] -> [B*S, H*W, 3]
            B, S, H, W, C = points.shape
            points = points.view(B * S, H * W, C)
            need_reshape = True
        elif len(points.shape) == 3:
            # [B, N, 3] - already in correct format
            need_reshape = False
        else:
            raise ValueError(f"Unexpected point cloud shape: {points.shape}")
            
        # Transpose for conv1d: [B, N, 3] -> [B, 3, N]
        points = points.transpose(1, 2)
        
        # PointNet layers
        x = F.relu(self.bn1(self.conv1(points)))  # [B, 64, N]
        x = F.relu(self.bn2(self.conv2(x)))       # [B, 128, N]
        x = self.bn3(self.conv3(x))               # [B, output_dim, N]
        
        # Global max pooling: [B, output_dim, N] -> [B, output_dim]
        x = torch.max(x, dim=2)[0]
        
        # Reshape back if needed
        if need_reshape:
            # [B*S, output_dim] -> [B, S, output_dim]
            x = x.view(B, S, -1)
            
        return x
