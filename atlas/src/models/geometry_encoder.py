"""
Enhanced 3D Geometry Feature Extractor
Extracts geometric features from VGGT outputs with temporal modeling and spatial attention

改进点：
1. 多层特征融合（类似FPN）
2. 时序Transformer建模多帧依赖
3. 空间注意力机制关注任务相关区域
4. 可选的3D点云处理
5. 相机位姿编码
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional
import math

from .pointnet import PointNetEncoder


class PositionalEncoding3D(nn.Module):
    """3D位置编码，用于空间token"""
    def __init__(self, dim: int, max_len: int = 5000):
        super().__init__()
        self.dim = dim
        
        # 预计算位置编码
        pe = torch.zeros(max_len, dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        
        div_term = torch.exp(torch.arange(0, dim, 2).float() * (-math.log(10000.0) / dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        """
        Args:
            x: [B, N, D] where N is sequence length
        Returns:
            [B, N, D] with positional encoding added
        """
        return x + self.pe[:x.size(1), :].unsqueeze(0)


class SpatialAttentionPooling(nn.Module):
    """
    空间注意力池化
    用learnable queries聚合空间token，而不是简单的mean pooling
    """
    def __init__(self, dim: int, num_queries: int = 8, num_heads: int = 8):
        super().__init__()
        self.num_queries = num_queries
        
        # Learnable queries用于聚合空间信息
        self.queries = nn.Parameter(torch.randn(1, num_queries, dim))
        
        # Multi-head attention
        self.attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            batch_first=True,
            dropout=0.1
        )
        
        # Layer norm
        self.norm = nn.LayerNorm(dim)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, N, D] spatial tokens
        Returns:
            [B, num_queries, D] aggregated features
        """
        B = x.size(0)
        
        # Expand queries
        queries = self.queries.expand(B, -1, -1)  # [B, num_queries, D]
        
        # Cross-attention: queries attend to spatial tokens
        out, attn_weights = self.attn(queries, x, x)  # [B, num_queries, D]
        
        # Residual + norm
        out = self.norm(out + queries)
        
        return out


class TemporalTransformerEncoder(nn.Module):
    """
    时序Transformer编码器
    处理多帧之间的时序依赖
    """
    def __init__(self, dim: int, num_layers: int = 2, num_heads: int = 8, mlp_ratio: float = 4.0):
        super().__init__()
        
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=dim,
                nhead=num_heads,
                dim_feedforward=int(dim * mlp_ratio),
                dropout=0.1,
                activation='gelu',
                batch_first=True,
                norm_first=True  # Pre-norm，更稳定
            )
            for _ in range(num_layers)
        ])
        
        # Positional encoding for temporal sequence
        self.pos_encoding = PositionalEncoding3D(dim)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, S, D] where S is number of frames
        Returns:
            [B, S, D] temporally encoded features
        """
        # Add positional encoding
        x = self.pos_encoding(x)
        
        # Apply transformer layers
        for layer in self.layers:
            x = layer(x)
            
        return x


class MultiLayerFeatureFusion(nn.Module):
    """
    多层特征融合模块
    融合VGGT不同层的特征（类似FPN的思想）
    """
    def __init__(self, token_dim: int, output_dim: int, num_layers: int = 3):
        super().__init__()
        self.num_layers = num_layers
        
        # 每层的投影
        self.layer_projs = nn.ModuleList([
            nn.Sequential(
                nn.Linear(token_dim, output_dim),
                nn.LayerNorm(output_dim),
                nn.GELU()
            )
            for _ in range(num_layers)
        ])
        
        # 融合权重（可学习）
        self.fusion_weights = nn.Parameter(torch.ones(num_layers) / num_layers)
        
    def forward(self, layer_features: List[torch.Tensor]) -> torch.Tensor:
        """
        Args:
            layer_features: List of [B, N, token_dim] from different VGGT layers
        Returns:
            [B, output_dim] fused features
        """
        assert len(layer_features) == self.num_layers, \
            f"Expected {self.num_layers} layers, got {len(layer_features)}"
        
        # 投影每层特征
        projected = []
        for feat, proj in zip(layer_features, self.layer_projs):
            # [B, N, token_dim] -> [B, N, output_dim]
            p = proj(feat)
            # Global average pooling: [B, N, output_dim] -> [B, output_dim]
            p = p.mean(dim=1)
            projected.append(p)
        
        # Stack: [num_layers, B, output_dim]
        stacked = torch.stack(projected, dim=0)
        
        # 加权融合: [B, output_dim]
        weights = F.softmax(self.fusion_weights, dim=0)  # 归一化权重
        fused = torch.sum(stacked * weights.view(-1, 1, 1), dim=0)
        
        return fused


class EnhancedGeometryFeatureExtractor(nn.Module):
    """
    Enhanced 3D Geometry Feature Extractor
    
    功能：
    1. 从VGGT aggregated tokens提取特征
    2. 多层特征融合（浅层→细节，深层→语义）
    3. 时序建模（多帧之间的依赖）
    4. 空间注意力（关注任务相关区域）
    5. 可选的点云和位姿编码
    
    Args:
        token_dim: VGGT token维度（默认2048 = 2 * embed_dim）
        output_dim: 输出特征维度
        num_fusion_layers: 使用多少层VGGT特征（默认3）
        use_temporal: 是否使用时序建模
        use_spatial_attn: 是否使用空间注意力
        use_pointnet: 是否使用点云特征
        use_pose: 是否使用相机位姿
        temporal_layers: 时序Transformer层数
        spatial_queries: 空间注意力query数量
    """
    
    def __init__(
        self,
        token_dim: int = 2048,
        output_dim: int = 512,
        num_fusion_layers: int = 3,
        use_temporal: bool = True,
        use_spatial_attn: bool = True,
        use_pointnet: bool = True,
        use_pose: bool = True,
        temporal_layers: int = 2,
        spatial_queries: int = 8,
    ):
        super().__init__()
        
        self.token_dim = token_dim
        self.output_dim = output_dim
        self.num_fusion_layers = num_fusion_layers
        self.use_temporal = use_temporal
        self.use_spatial_attn = use_spatial_attn
        self.use_pointnet = use_pointnet
        self.use_pose = use_pose
        
        # ============================================================
        # 1. 多层特征融合
        # ============================================================
        self.layer_fusion = MultiLayerFeatureFusion(
            token_dim=token_dim,
            output_dim=output_dim // 2,  # 一半维度给token特征
            num_layers=num_fusion_layers
        )
        
        # ============================================================
        # 2. 空间注意力池化（可选）
        # ============================================================
        if use_spatial_attn:
            self.spatial_pooling = SpatialAttentionPooling(
                dim=token_dim,
                num_queries=spatial_queries,
                num_heads=8
            )
            # 将spatial queries投影到output_dim
            self.spatial_proj = nn.Sequential(
                nn.Linear(token_dim * spatial_queries, output_dim // 2),
                nn.LayerNorm(output_dim // 2),
                nn.GELU()
            )
        else:
            self.spatial_pooling = None
            self.spatial_proj = None
            
        # ============================================================
        # 3. 时序建模（可选）
        # ============================================================
        if use_temporal:
            # 时序transformer的输入维度
            temporal_input_dim = output_dim // 2
            if use_spatial_attn:
                temporal_input_dim = output_dim  # token特征 + spatial特征
                
            self.temporal_encoder = TemporalTransformerEncoder(
                dim=temporal_input_dim,
                num_layers=temporal_layers,
                num_heads=8,
                mlp_ratio=4.0
            )
            
            # 时序聚合：加权pooling（学习每帧的重要性）
            self.temporal_weights = nn.Sequential(
                nn.Linear(temporal_input_dim, 1),
                nn.Softmax(dim=1)
            )
        else:
            self.temporal_encoder = None
            self.temporal_weights = None
            
        # ============================================================
        # 4. 点云特征提取（可选）
        # ============================================================
        if use_pointnet:
            self.pointnet = PointNetEncoder(output_dim=output_dim // 4)
        else:
            self.pointnet = None
            
        # ============================================================
        # 5. 相机位姿编码（可选）
        # ============================================================
        if use_pose:
            self.pose_encoder = nn.Sequential(
                nn.Linear(9, output_dim // 4),  # 9D: 3x3旋转矩阵展平
                nn.LayerNorm(output_dim // 4),
                nn.GELU(),
                nn.Linear(output_dim // 4, output_dim // 4)
            )
        else:
            self.pose_encoder = None
            
        # ============================================================
        # 6. 最终投影层
        # ============================================================
        # 计算总维度
        final_input_dim = 0
        
        # Token特征
        if use_temporal:
            if use_spatial_attn:
                final_input_dim += output_dim  # temporal后的特征
            else:
                final_input_dim += output_dim // 2
        else:
            if use_spatial_attn:
                final_input_dim += output_dim
            else:
                final_input_dim += output_dim // 2
                
        # 点云特征
        if use_pointnet:
            final_input_dim += output_dim // 4
            
        # 位姿特征
        if use_pose:
            final_input_dim += output_dim // 4
            
        # 最终投影到output_dim
        if final_input_dim != output_dim:
            self.final_proj = nn.Sequential(
                nn.Linear(final_input_dim, output_dim),
                nn.LayerNorm(output_dim),
                nn.GELU(),
                nn.Dropout(0.1)
            )
        else:
            self.final_proj = nn.Identity()
            
    def forward(
        self,
        aggregated_tokens_list: List[torch.Tensor],
        world_points: Optional[torch.Tensor] = None,
        depth: Optional[torch.Tensor] = None,
        pose_enc: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            aggregated_tokens_list: List of [B, S, P, token_dim] from VGGT
                S = number of frames
                P = number of patches per frame
            world_points: [B, S, H, W, 3] - 3D point cloud (optional)
            depth: [B, S, H, W, 1] - Depth maps (optional, not used currently)
            pose_enc: [B, S, 9] - Camera pose encoding (optional)
            
        Returns:
            geometry_features: [B, S, output_dim] - Per-frame geometric features
        """
        # 获取形状
        last_layer = aggregated_tokens_list[-1]
        B, S, P, D = last_layer.shape
        device = last_layer.device
        
        # ============================================================
        # 1. 多层Token特征融合
        # ============================================================
        # 选择最后num_fusion_layers层
        selected_layers = aggregated_tokens_list[-self.num_fusion_layers:]
        
        # 处理每一帧
        frame_features = []
        for frame_idx in range(S):
            # 提取当前帧的所有层特征
            frame_layer_feats = [
                layer[:, frame_idx, :, :]  # [B, P, token_dim]
                for layer in selected_layers
            ]
            
            # 如果使用空间注意力
            if self.use_spatial_attn and self.spatial_pooling is not None:
                # 对每层应用空间注意力
                spatial_feats = []
                for layer_feat in frame_layer_feats:
                    # [B, P, token_dim] -> [B, num_queries, token_dim]
                    spatial_feat = self.spatial_pooling(layer_feat)
                    spatial_feats.append(spatial_feat)
                
                # 多层spatial features融合
                # Stack and average: [num_layers, B, num_queries, token_dim] -> [B, num_queries, token_dim]
                spatial_stacked = torch.stack(spatial_feats, dim=0).mean(dim=0)
                
                # Flatten queries: [B, num_queries * token_dim]
                spatial_flat = spatial_stacked.flatten(1)
                
                # 投影: [B, num_queries * token_dim] -> [B, output_dim // 2]
                spatial_out = self.spatial_proj(spatial_flat)  # [B, output_dim // 2]
                
                # 同时也保留多层融合的全局特征
                layer_fused = self.layer_fusion(frame_layer_feats)  # [B, output_dim // 2]
                
                # 拼接: [B, output_dim]
                frame_feat = torch.cat([layer_fused, spatial_out], dim=-1)
            else:
                # 只用多层融合
                frame_feat = self.layer_fusion(frame_layer_feats)  # [B, output_dim // 2]
                
            frame_features.append(frame_feat)
        
        # Stack所有帧: [B, S, output_dim] or [B, S, output_dim // 2]
        token_features = torch.stack(frame_features, dim=1)
        
        # ============================================================
        # 2. 时序建模（如果启用）
        # ============================================================
        if self.use_temporal and self.temporal_encoder is not None:
            # 时序transformer: [B, S, D] -> [B, S, D]
            temporal_features = self.temporal_encoder(token_features)
            
            # 学习每帧的权重: [B, S, 1]
            weights = self.temporal_weights(temporal_features)
            
            # 加权聚合: [B, S, D] -> [B, D]
            # 然后再expand回每一帧（保持[B, S, D]格式）
            # 这里我们保留时序信息，不做聚合
            token_features = temporal_features
        
        # 现在token_features: [B, S, output_dim] or [B, S, output_dim // 2]
        
        # ============================================================
        # 3. 点云特征（如果启用）
        # ============================================================
        if self.use_pointnet and self.pointnet is not None and world_points is not None:
            # PointNet: [B, S, H, W, 3] -> [B, S, output_dim // 4]
            point_features = self.pointnet(world_points)
        elif self.use_pointnet:
            # 如果启用但未提供点云，用零填充
            point_features = torch.zeros(
                B, S, self.output_dim // 4,
                device=device,
                dtype=token_features.dtype
            )
        else:
            point_features = None
            
        # ============================================================
        # 4. 相机位姿特征（如果启用）
        # ============================================================
        if self.use_pose and self.pose_encoder is not None and pose_enc is not None:
            # Pose encoder: [B, S, 9] -> [B, S, output_dim // 4]
            pose_features = self.pose_encoder(pose_enc)
        elif self.use_pose:
            # 如果启用但未提供位姿，用零填充
            pose_features = torch.zeros(
                B, S, self.output_dim // 4,
                device=device,
                dtype=token_features.dtype
            )
        else:
            pose_features = None
            
        # ============================================================
        # 5. 拼接所有特征
        # ============================================================
        features_to_concat = [token_features]
        
        if point_features is not None:
            features_to_concat.append(point_features)
            
        if pose_features is not None:
            features_to_concat.append(pose_features)
            
        # Concatenate: [B, S, total_dim]
        combined_features = torch.cat(features_to_concat, dim=-1)
        
        # ============================================================
        # 6. 最终投影
        # ============================================================
        geometry_features = self.final_proj(combined_features)  # [B, S, output_dim]
        
        return geometry_features
    
    def get_feature_dims(self) -> dict:
        """返回各个组件的特征维度（调试用）"""
        dims = {
            "token_features": self.output_dim // 2 if not self.use_spatial_attn else self.output_dim,
        }
        
        if self.use_pointnet:
            dims["point_features"] = self.output_dim // 4
            
        if self.use_pose:
            dims["pose_features"] = self.output_dim // 4
            
        dims["total"] = sum(dims.values())
        dims["output"] = self.output_dim
        
        return dims


# ============================================================
# 便捷构造函数
# ============================================================

def create_geometry_encoder(
    config: str = "full",
    token_dim: int = 2048,
    output_dim: int = 512,
    **kwargs
) -> EnhancedGeometryFeatureExtractor:
    """
    便捷构造函数，提供几种预设配置
    
    Args:
        config: "full", "temporal_only", "spatial_only", "minimal"
        token_dim: VGGT token维度
        output_dim: 输出维度
        **kwargs: 其他参数覆盖
        
    Returns:
        EnhancedGeometryFeatureExtractor instance
    """
    configs = {
        "full": {
            "use_temporal": True,
            "use_spatial_attn": True,
            "use_pointnet": True,
            "use_pose": True,
            "num_fusion_layers": 3,
            "temporal_layers": 2,
            "spatial_queries": 8,
        },
        "temporal_only": {
            "use_temporal": True,
            "use_spatial_attn": False,
            "use_pointnet": True,
            "use_pose": True,
            "num_fusion_layers": 3,
            "temporal_layers": 3,
        },
        "spatial_only": {
            "use_temporal": False,
            "use_spatial_attn": True,
            "use_pointnet": True,
            "use_pose": True,
            "num_fusion_layers": 3,
            "spatial_queries": 16,
        },
        "minimal": {
            "use_temporal": False,
            "use_spatial_attn": False,
            "use_pointnet": False,
            "use_pose": False,
            "num_fusion_layers": 2,
        },
    }
    
    if config not in configs:
        raise ValueError(f"Unknown config: {config}. Choose from {list(configs.keys())}")
    
    # 合并配置
    final_config = configs[config].copy()
    final_config.update(kwargs)
    
    return EnhancedGeometryFeatureExtractor(
        token_dim=token_dim,
        output_dim=output_dim,
        **final_config
    )