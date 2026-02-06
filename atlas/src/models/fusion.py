"""
多模态融合模块
融合语言和3D几何特征，使用交叉注意力机制

改进5: 使用attention pooling替代简单的mean pooling
- 添加learnable query token用于聚合语言特征
- 保留细粒度的语言token信息
- 提高对关键指令词的敏感度
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultimodalFusion(nn.Module):
    """
    多模态融合模块
    
    架构：
    - 语言特征作为query，几何特征作为key/value
    - 多层交叉注意力实现深度融合
    - 改进5: 使用attention pooling聚合语言特征（替代mean pooling）
    
    Args:
        lang_dim: 语言特征维度
        geom_dim: 几何特征维度
        hidden_dim: 隐藏层维度
        num_layers: 交叉注意力层数
        num_heads: 注意力头数
        dropout: Dropout比率
        use_attention_pooling: 是否使用attention pooling（改进5）
    """
    
    def __init__(self, lang_dim=4096, geom_dim=512, hidden_dim=1024, 
                 num_layers=4, num_heads=16, dropout=0.1,
                 use_attention_pooling=True):
        super().__init__()
        
        self.lang_dim = lang_dim
        self.geom_dim = geom_dim
        self.hidden_dim = hidden_dim
        self.use_attention_pooling = use_attention_pooling
        
        # 投影语言和几何特征到统一维度
        self.lang_proj = nn.Sequential(
            nn.Linear(lang_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout)
        )
        
        self.geom_proj = nn.Sequential(
            nn.Linear(geom_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout)
        )
        
        # 交叉注意力层
        # 语言特征作为query，几何特征作为key/value
        self.cross_attn_layers = nn.ModuleList([
            nn.TransformerDecoderLayer(
                d_model=hidden_dim,
                nhead=num_heads,
                dim_feedforward=hidden_dim * 4,
                dropout=dropout,
                activation='gelu',
                batch_first=True,
                norm_first=True  # Pre-norm提高训练稳定性
            ) for _ in range(num_layers)
        ])
        
        # 改进5: Attention pooling用于聚合语言特征
        if use_attention_pooling:
            # Learnable query token用于聚合语言token
            self.lang_query = nn.Parameter(torch.randn(1, 1, hidden_dim))
            self.lang_attn_pool = nn.MultiheadAttention(
                embed_dim=hidden_dim,
                num_heads=num_heads,
                dropout=dropout,
                batch_first=True
            )
        else:
            self.lang_query = None
            self.lang_attn_pool = None
        
        # 输出投影
        self.output_norm = nn.LayerNorm(hidden_dim)
        self.output_proj = nn.Linear(hidden_dim, hidden_dim)
        
    def forward(self, lang_features, geom_features):
        """
        前向传播
        
        Args:
            lang_features: [B, L, lang_dim] - 语言token特征
            geom_features: [B, S, geom_dim] - 几何特征（可以是多帧）
            
        Returns:
            fused_features: [B, hidden_dim] - 融合后的表示
        """
        # 投影到统一维度
        lang = self.lang_proj(lang_features)  # [B, L, hidden_dim]
        geom = self.geom_proj(geom_features)  # [B, S, hidden_dim]
        
        # 交叉注意力：语言query关注几何key/value
        # 这允许语言特征关注相关的3D几何信息
        fused = lang
        for layer in self.cross_attn_layers:
            # TransformerDecoderLayer中：
            # - tgt (query) = lang_features
            # - memory (key/value) = geom_features
            fused = layer(fused, geom)  # [B, L, hidden_dim]
        
        # 改进5: 聚合语言token
        if self.use_attention_pooling and self.lang_query is not None:
            # 使用attention pooling替代mean pooling
            # 这可以保留细粒度的语言信息，提高对关键指令词的敏感度
            query = self.lang_query.expand(fused.size(0), -1, -1)  # [B, 1, hidden_dim]
            fused_pooled, attn_weights = self.lang_attn_pool(
                query, fused, fused  # query, key, value
            )
            fused = fused_pooled.squeeze(1)  # [B, hidden_dim]
        else:
            # 原始方法：mean pooling
            # 如果语言编码器提供[CLS] token，可以使用它；否则使用mean
            fused = fused.mean(dim=1)  # [B, hidden_dim]
        
        # 最终投影
        fused = self.output_norm(fused)
        fused = self.output_proj(fused)  # [B, hidden_dim]
        
        return fused
