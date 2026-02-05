"""
3D Geometry Feature Extractor
Extracts geometric features from VGGT outputs
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .pointnet import PointNetEncoder




class GeometryFeatureExtractor(nn.Module):
    """
    Extract 3D geometric features from VGGT outputs
    
    Inputs:
    - aggregated_tokens_list: List of token tensors from VGGT aggregator
    - world_points: [B, S, H, W, 3] - 3D point cloud (optional)
    - depth: [B, S, H, W, 1] - Depth maps (optional)
    - pose_enc: [B, S, 9] - Camera pose encoding (optional)
    
    Output:
    - geometry_features: [B, S, output_dim] - Geometric feature vectors
    """
    
    def __init__(self, token_dim=2048, output_dim=512, use_pointnet=True, use_pose=True):
        super().__init__()
        self.use_pointnet = use_pointnet
        self.use_pose = use_pose
        
        # Token feature extraction (from aggregated tokens)
        # Use spatial pooling or learnable query
        self.token_pooler = nn.Sequential(
            nn.Linear(token_dim, output_dim // 2),
            nn.LayerNorm(output_dim // 2),
            nn.GELU()
        )
        
        # Learnable query for token aggregation (alternative to mean pooling)
        self.token_query = nn.Parameter(torch.randn(1, 1, token_dim))
        
        # 3D point cloud feature extraction (optional)
        if use_pointnet:
            self.pointnet = PointNetEncoder(output_dim // 2)
        else:
            self.pointnet = None
            
        # Camera pose encoding (optional)
        if use_pose:
            self.pose_encoder = nn.Sequential(
                nn.Linear(9, output_dim // 4),
                nn.LayerNorm(output_dim // 4),
                nn.GELU()
            )
        else:
            self.pose_encoder = None
            
        # Final feature dimension calculation
        feat_dims = [output_dim // 2]  # token features
        if use_pointnet:
            feat_dims.append(output_dim // 2)  # pointnet features
        if use_pose:
            feat_dims.append(output_dim // 4)  # pose features
            
        # Projection to final output dimension
        total_dim = sum(feat_dims)
        if total_dim != output_dim:
            self.final_proj = nn.Linear(total_dim, output_dim)
        else:
            self.final_proj = nn.Identity()
            
    def forward(self, aggregated_tokens_list, world_points=None, 
                depth=None, pose_enc=None):
        """
        Args:
            aggregated_tokens_list: List of tensors, each [B, S, P, token_dim]
            world_points: [B, S, H, W, 3] or None
            depth: [B, S, H, W, 1] or None
            pose_enc: [B, S, 9] or None
            
        Returns:
            geometry_features: [B, S, output_dim]
        """
        B, S = aggregated_tokens_list[-1].shape[:2]
        device = aggregated_tokens_list[-1].device
        
        # 1. Extract token features from the last few layers
        # Use the last 2 layers for richer features
        num_layers = min(2, len(aggregated_tokens_list))
        tokens = aggregated_tokens_list[-num_layers:]  # Last 2 layers
        
        # Average over layers: [B, S, P, token_dim]
        tokens = torch.stack(tokens, dim=0).mean(dim=0)
        
        # Spatial pooling: [B, S, P, token_dim] -> [B, S, token_dim]
        # Option 1: Mean pooling (simple and effective)
        token_feat = tokens.mean(dim=2)  # [B, S, token_dim]
        
        # Option 2: Learnable query attention (commented out, can be enabled)
        # token_query = self.token_query.expand(B * S, -1, -1)  # [B*S, 1, token_dim]
        # tokens_flat = tokens.view(B * S, tokens.shape[2], tokens.shape[3])  # [B*S, P, token_dim]
        # attn_weights = F.softmax(torch.bmm(token_query, tokens_flat.transpose(1, 2)), dim=-1)  # [B*S, 1, P]
        # token_feat = torch.bmm(attn_weights, tokens_flat).squeeze(1)  # [B*S, token_dim]
        # token_feat = token_feat.view(B, S, -1)  # [B, S, token_dim]
        
        token_feat = self.token_pooler(token_feat)  # [B, S, output_dim // 2]
        
        # 2. Extract 3D point cloud features (if provided)
        features = [token_feat]
        
        if self.use_pointnet and world_points is not None:
            point_feat = self.pointnet(world_points)  # [B, S, output_dim // 2]
            features.append(point_feat)
        elif self.use_pointnet:
            # If pointnet is enabled but world_points not provided, use zeros
            point_feat = torch.zeros(B, S, self.pointnet.conv3.out_channels, 
                                    device=device, dtype=token_feat.dtype)
            features.append(point_feat)
            
        # 3. Encode camera pose (if provided)
        if self.use_pose and pose_enc is not None:
            pose_feat = self.pose_encoder(pose_enc)  # [B, S, output_dim // 4]
            features.append(pose_feat)
        elif self.use_pose:
            # If pose encoder is enabled but pose_enc not provided, use zeros
            pose_feat = torch.zeros(B, S, 9, device=device, dtype=token_feat.dtype)
            pose_feat = self.pose_encoder(pose_feat)  # [B, S, output_dim // 4]
            features.append(pose_feat)
            
        # 4. Concatenate and project to final dimension
        geometry_features = torch.cat(features, dim=-1)  # [B, S, total_dim]
        geometry_features = self.final_proj(geometry_features)  # [B, S, output_dim]
        
        return geometry_features
