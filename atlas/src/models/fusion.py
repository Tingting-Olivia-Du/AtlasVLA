"""
多模态融合模块
融合语言和3D几何特征

包含两种融合策略：
1. MultimodalFusion - 单向cross-attention（原始版本）
2. BidirectionalFusion - 双向交互融合（改进版本）
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultimodalFusion(nn.Module):
    """
    多模态融合模块（原始版本）
    
    架构：
    - 语言特征作为query，几何特征作为key/value
    - 多层交叉注意力实现深度融合
    - 使用attention pooling聚合语言特征（替代mean pooling）
    
    Args:
        lang_dim: 语言特征维度
        geom_dim: 几何特征维度
        hidden_dim: 隐藏层维度
        num_layers: 交叉注意力层数
        num_heads: 注意力头数
        dropout: Dropout比率
        use_attention_pooling: 是否使用attention pooling
    """
    
    def __init__(
        self,
        lang_dim: int = 4096,
        geom_dim: int = 512,
        hidden_dim: int = 1024,
        num_layers: int = 4,
        num_heads: int = 16,
        dropout: float = 0.1,
        use_attention_pooling: bool = True
    ):
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
        
        # Attention pooling用于聚合语言特征
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
        
        # 聚合语言token
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


class BidirectionalFusion(nn.Module):
    """
    双向多模态融合模块（改进版本）
    
    改进点：
    1. 双向交互：language ↔ geometry 互相关注
    2. 早期融合：concat后联合编码，而不是单向cross-attention
    3. Perceiver-style aggregation：用少量queries高效聚合信息
    
    Args:
        lang_dim: 语言特征维度
        geom_dim: 几何特征维度
        hidden_dim: 隐藏层维度
        num_layers: Transformer层数
        num_heads: 注意力头数
        num_perceiver_queries: Perceiver queries数量
        dropout: Dropout比率
    """
    
    def __init__(
        self,
        lang_dim: int = 4096,
        geom_dim: int = 512,
        hidden_dim: int = 1024,
        num_layers: int = 4,
        num_heads: int = 16,
        num_perceiver_queries: int = 32,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.num_perceiver_queries = num_perceiver_queries
        
        # 投影到统一空间
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
        
        # 位置编码（区分language和geometry tokens）
        self.lang_type_embed = nn.Parameter(torch.randn(1, 1, hidden_dim))
        self.geom_type_embed = nn.Parameter(torch.randn(1, 1, hidden_dim))
        
        # 双向transformer（类似BERT的双向性）
        self.fusion_blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=num_heads,
                dim_feedforward=hidden_dim * 4,
                dropout=dropout,
                activation='gelu',
                batch_first=True,
                norm_first=True
            )
            for _ in range(num_layers)
        ])
        
        # Perceiver-style cross-attention（更高效的聚合）
        self.perceiver_queries = nn.Parameter(
            torch.randn(1, num_perceiver_queries, hidden_dim)
        )
        self.perceiver_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # 最终投影
        self.final_norm = nn.LayerNorm(hidden_dim)
        self.final_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
    def forward(self, lang_features, geom_features):
        """
        前向传播
        
        Args:
            lang_features: [B, L, lang_dim] - 语言token特征
            geom_features: [B, S, geom_dim] - 几何特征序列
            
        Returns:
            fused_features: [B, hidden_dim] - 融合后的表示
        """
        B = lang_features.size(0)
        
        # 投影到统一空间
        lang = self.lang_proj(lang_features)  # [B, L, hidden_dim]
        geom = self.geom_proj(geom_features)  # [B, S, hidden_dim]
        
        # 添加type embeddings（类似BERT的segment embeddings）
        lang = lang + self.lang_type_embed.expand(B, lang.size(1), -1)
        geom = geom + self.geom_type_embed.expand(B, geom.size(1), -1)
        
        # 拼接进行联合编码（早期融合）
        combined = torch.cat([lang, geom], dim=1)  # [B, L+S, hidden_dim]
        
        # 多层双向attention
        # language tokens和geometry tokens互相关注
        for block in self.fusion_blocks:
            combined = block(combined)  # [B, L+S, hidden_dim]
        
        # Perceiver: 用少量queries聚合信息
        # 这比直接mean pooling更灵活，可以学习聚焦到重要信息
        queries = self.perceiver_queries.expand(B, -1, -1)  # [B, num_queries, hidden_dim]
        
        aggregated, _ = self.perceiver_attn(
            queries,    # query
            combined,   # key
            combined    # value
        )  # [B, num_queries, hidden_dim]
        
        # 聚合所有queries（可以用mean或者只取第一个）
        fused = aggregated.mean(dim=1)  # [B, hidden_dim]
        
        # 最终投影
        fused = self.final_norm(fused)
        fused = self.final_proj(fused)  # [B, hidden_dim]
        
        return fused


class PerceiverResampler(nn.Module):
    """
    Perceiver Resampler (from Flamingo/IDEFICS)
    
    用固定数量的learnable queries来聚合可变长度的输入
    特别适合处理多帧视觉特征
    """
    
    def __init__(
        self,
        dim: int,
        depth: int = 2,
        num_latents: int = 64,
        num_heads: int = 8,
        ff_mult: int = 4,
    ):
        super().__init__()
        
        self.latents = nn.Parameter(torch.randn(num_latents, dim))
        
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                nn.LayerNorm(dim),
                nn.MultiheadAttention(dim, num_heads, batch_first=True),
                nn.LayerNorm(dim),
                nn.Sequential(
                    nn.Linear(dim, dim * ff_mult),
                    nn.GELU(),
                    nn.Linear(dim * ff_mult, dim)
                )
            ]))
            
    def forward(self, x):
        """
        Args:
            x: [B, N, dim] - 输入特征序列（可变长度）
        Returns:
            [B, num_latents, dim] - 固定长度的输出
        """
        B = x.size(0)
        
        # Expand latents for batch
        latents = self.latents.unsqueeze(0).expand(B, -1, -1)  # [B, num_latents, dim]
        
        for norm1, attn, norm2, ff in self.layers:
            # Cross attention: latents attend to input
            normed_latents = norm1(latents)
            attn_out, _ = attn(normed_latents, x, x)
            latents = latents + attn_out
            
            # Feedforward
            latents = latents + ff(norm2(latents))
            
        return latents


class AdaptiveFusion(nn.Module):
    """
    自适应融合模块
    
    根据任务动态调整language和geometry的融合权重
    """
    
    def __init__(self, lang_dim: int, geom_dim: int, hidden_dim: int):
        super().__init__()
        
        self.lang_proj = nn.Linear(lang_dim, hidden_dim)
        self.geom_proj = nn.Linear(geom_dim, hidden_dim)
        
        # 学习融合权重的网络
        self.weight_net = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2),
            nn.Softmax(dim=-1)
        )
        
    def forward(self, lang_features, geom_features):
        """
        Args:
            lang_features: [B, lang_dim]
            geom_features: [B, geom_dim]
        Returns:
            [B, hidden_dim]
        """
        lang_proj = self.lang_proj(lang_features)  # [B, hidden_dim]
        geom_proj = self.geom_proj(geom_features)  # [B, hidden_dim]
        
        # 计算融合权重
        concat = torch.cat([lang_proj, geom_proj], dim=-1)  # [B, 2*hidden_dim]
        weights = self.weight_net(concat)  # [B, 2]
        
        # 加权融合
        w_lang = weights[:, 0:1]  # [B, 1]
        w_geom = weights[:, 1:2]  # [B, 1]
        
        fused = w_lang * lang_proj + w_geom * geom_proj  # [B, hidden_dim]
        
        return fused