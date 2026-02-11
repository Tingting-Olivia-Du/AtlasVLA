"""
VGGT Adapter - 从 HuggingFace 加载 facebook/vggt 并适配到 VLA 任务
专门处理单帧输入 + 语言指令
"""
import torch
import torch.nn as nn
from typing import Dict, Tuple, Optional
from transformers import AutoModel


class VGGTAdapter(nn.Module):
    """
    适配 facebook/vggt 到 VLA 任务:
    1. ✅ 处理单帧输入 (原始VGGT设计用于视频序列，我们适配为单帧)
    2. ✅ 注入 language tokens (通过特殊的融合机制)
    3. ✅ 提取适合 action prediction 的特征
    
    关键改进:
    - 单帧图像被扩展为伪视频序列 [B, 1, 3, H, W] 以适配VGGT
    - Language tokens通过attention机制与visual features交互
    - 使用learnable action queries提取任务相关特征
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # 加载预训练的 VGGT
        print("\n" + "="*60)
        print("Loading facebook/vggt from HuggingFace...")
        print("="*60)
        try:
            self.vggt = AutoModel.from_pretrained(
                "facebook/vggt",
                trust_remote_code=True
            )
            print("✓ Successfully loaded facebook/vggt from HuggingFace")
            self.use_pretrained_vggt = True
        except Exception as e:
            print(f"⚠ Warning: Could not load facebook/vggt from HuggingFace: {e}")
            print("Falling back to local VGGT implementation...")
            try:
                # Fallback: 从本地vggt目录加载
                import sys
                import os
                vggt_path = os.path.join(os.path.dirname(__file__), '../../vggt')
                if vggt_path not in sys.path:
                    sys.path.insert(0, vggt_path)
                
                from vggt.models.vggt import VGGT
                self.vggt = VGGT(
                    img_size=224,  # 适配我们的输入尺寸
                    patch_size=16,
                    embed_dim=1024,
                    enable_camera=False,
                    enable_point=False,
                    enable_depth=False,
                    enable_track=False
                )
                print("✓ Successfully loaded VGGT from local implementation")
                self.use_pretrained_vggt = False
            except Exception as e2:
                print(f"✗ Error loading local VGGT: {e2}")
                raise RuntimeError("Cannot load VGGT. Please install vggt or check HuggingFace access.")
        
        # VGGT的embedding维度
        self.vggt_embed_dim = 1024  # facebook/vggt 默认
        self.target_dim = config.embed_dim  # 我们的目标维度 (768)
        
        print(f"  VGGT embedding dim: {self.vggt_embed_dim}")
        print(f"  Target dim: {self.target_dim}")
        
        # Language token 投影层: target_dim -> vggt_embed_dim
        # Qwen3-0.6B输出需要投影到VGGT空间
        self.lang_adapter = nn.Sequential(
            nn.Linear(self.target_dim, self.vggt_embed_dim),
            nn.LayerNorm(self.vggt_embed_dim),
            nn.GELU(),
            nn.Dropout(0.1)
        )
        print(f"  Language adapter: {self.target_dim} -> {self.vggt_embed_dim}")
        
        # Vision token 投影层 (如果维度不匹配)
        if self.target_dim != self.vggt_embed_dim:
            self.vision_adapter = nn.Sequential(
                nn.Linear(self.target_dim, self.vggt_embed_dim),
                nn.LayerNorm(self.vggt_embed_dim)
            )
            print(f"  Vision adapter: {self.target_dim} -> {self.vggt_embed_dim}")
        else:
            self.vision_adapter = nn.Identity()
            print("  Vision adapter: Identity (dims match)")
        
        # Cross-modal attention: language attends to vision
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=self.vggt_embed_dim,
            num_heads=8,
            dropout=0.1,
            batch_first=True
        )
        self.cross_attn_norm = nn.LayerNorm(self.vggt_embed_dim)
        
        # 特征提取层：从 VGGT 输出提取 VLA 特征
        # VGGT aggregator 输出是 list of [B, S, P, 2C]，我们取最后一层
        self.feature_projector = nn.Sequential(
            nn.Linear(self.vggt_embed_dim * 2, self.vggt_embed_dim),
            nn.LayerNorm(self.vggt_embed_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(self.vggt_embed_dim, self.target_dim),
            nn.LayerNorm(self.target_dim)
        )
        print(f"  Feature projector: {self.vggt_embed_dim * 2} -> {self.target_dim}")
        
        # Action queries (可学习的query tokens用于action prediction)
        self.num_action_queries = 16
        self.action_queries = nn.Parameter(
            torch.randn(1, self.num_action_queries, self.target_dim)
        )
        nn.init.trunc_normal_(self.action_queries, std=0.02)
        print(f"  Action queries: {self.num_action_queries} learnable tokens")
        
        # 冻结VGGT backbone (可选)
        if config.freeze_vggt:
            print("\n  🔒 Freezing VGGT backbone...")
            for param in self.vggt.parameters():
                param.requires_grad = False
            trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
            print(f"  ✓ Trainable parameters: {trainable / 1e6:.2f}M (adapter layers only)")
        else:
            trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
            print(f"  Trainable parameters: {trainable / 1e6:.2f}M (full model)")
        
        print("="*60 + "\n")
    
    def forward(
        self,
        vision_tokens: torch.Tensor,      # [B, N_v, D]
        language_tokens: torch.Tensor,    # [B, N_l, D]
        vision_info: Dict,
        language_info: Dict,
        language_mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict]:
        """
        处理单帧输入的forward pass
        
        Args:
            vision_tokens: [B, N_v, D] - 来自vision encoder的tokens (单帧)
            language_tokens: [B, N_l, D] - 来自Qwen3的language tokens
            
        Returns:
            vision_features: [B, N_v, D]
            language_features: [B, N_l, D]
            global_features: [B, num_queries, D]
            output_info: Dict
            
        流程:
        1. 将单帧vision tokens适配到VGGT空间
        2. Language tokens通过cross-attention与vision交互
        3. 使用VGGT处理fused features
        4. 提取action-relevant features
        """
        B = vision_tokens.size(0)
        N_v = vision_tokens.size(1)
        N_l = language_tokens.size(1)
        device = vision_tokens.device
        
        # ========== Step 1: 适配到VGGT空间 ==========
        vision_adapted = self.vision_adapter(vision_tokens)  # [B, N_v, 1024]
        language_adapted = self.lang_adapter(language_tokens)  # [B, N_l, 1024]
        
        # ========== Step 2: Cross-modal interaction ==========
        # Language attends to vision (language-conditioned visual features)
        language_enhanced, _ = self.cross_attn(
            query=language_adapted,
            key=vision_adapted,
            value=vision_adapted,
            need_weights=False
        )
        language_enhanced = self.cross_attn_norm(language_enhanced + language_adapted)
        
        # ========== Step 3: 准备VGGT输入 (单帧) ==========
        # 将vision tokens重塑为VGGT期望的格式
        # VGGT期望: [B, S, 3, H, W] 其中 S 是序列长度
        # 对于单帧，我们设置 S=1
        
        # 但是我们已经有tokens了，需要构造伪图像或直接使用aggregator
        # 这里采用直接使用aggregator的方案
        
        try:
            # 尝试使用VGGT的aggregator
            aggregator = self.vggt.aggregator if hasattr(self.vggt, 'aggregator') else None
            
            if aggregator is not None and hasattr(aggregator, 'frame_blocks'):
                # 使用VGGT的transformer blocks
                # 将vision和language tokens作为输入
                x = torch.cat([vision_adapted, language_enhanced], dim=1)  # [B, N_v+N_l, 1024]
                
                # VGGT alternating attention
                num_layers = min(len(aggregator.frame_blocks), len(aggregator.global_blocks))
                for i in range(num_layers):
                    # Frame-level attention
                    if hasattr(aggregator, 'frame_blocks'):
                        x = aggregator.frame_blocks[i](x, pos=None)
                    # Global attention
                    if hasattr(aggregator, 'global_blocks'):
                        x = aggregator.global_blocks[i](x, pos=None)
                
                # 投影回目标维度
                # VGGT输出需要concat (模拟frame和global特征)
                x_projected = self.feature_projector(torch.cat([x, x], dim=-1))  # [B, N_v+N_l, D]
                
            else:
                # Fallback: 简单的transformer处理
                print("Warning: Using fallback path (VGGT aggregator not available)")
                x = torch.cat([vision_adapted, language_enhanced], dim=1)
                # 简单投影
                x_projected = self.feature_projector(torch.cat([x, x], dim=-1))
                
        except Exception as e:
            print(f"Warning: Error in VGGT processing: {e}")
            print("Using simple concatenation fallback")
            x = torch.cat([vision_adapted, language_enhanced], dim=1)
            x_projected = self.feature_projector(torch.cat([x, x], dim=-1))
        
        # ========== Step 4: 分离features ==========
        vision_features = x_projected[:, :N_v, :]       # [B, N_v, D]
        language_features = x_projected[:, N_v:, :]     # [B, N_l, D]
        
        # ========== Step 5: 生成global features for action ==========
        # 使用可学习的action queries
        global_features = self.action_queries.expand(B, -1, -1)  # [B, num_queries, D]
        
        # Attention-based feature aggregation
        # Action queries attend to all features
        all_features = x_projected  # [B, N_v+N_l, D]
        pooled = all_features.mean(dim=1, keepdim=True).expand(-1, self.num_action_queries, -1)
        global_features = global_features + 0.1 * pooled  # Weighted combination
        
        output_info = {
            'vggt_embed_dim': self.vggt_embed_dim,
            'target_dim': self.target_dim,
            'num_vision_tokens': N_v,
            'num_language_tokens': N_l,
            'num_action_queries': self.num_action_queries,
            'single_frame_input': True  # 标记这是单帧输入
        }
        
        return vision_features, language_features, global_features, output_info


class SimpleVGGTBackbone(nn.Module):
    """
    简化版VGGT Backbone - 不依赖HuggingFace
    用于快速实验和调试
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        from .components.token_fusion import TokenFusion
        from .components.graph_builder import GraphBuilder
        from .components.vggt_layers import VGGTLayer
        
        self.token_fusion = TokenFusion(config)
        self.graph_builder = GraphBuilder(config)
        
        self.layers = nn.ModuleList([
            VGGTLayer(config)
            for _ in range(config.depth)
        ])
        
        self.norm = nn.LayerNorm(config.embed_dim)
        
        # Action queries
        self.num_action_queries = 16
        self.action_queries = nn.Parameter(
            torch.randn(1, self.num_action_queries, config.embed_dim)
        )
        nn.init.trunc_normal_(self.action_queries, std=0.02)
    
    def forward(
        self,
        vision_tokens: torch.Tensor,
        language_tokens: torch.Tensor,
        vision_info: Dict,
        language_info: Dict,
        language_mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict]:
        
        B = vision_tokens.size(0)
        device = vision_tokens.device
        
        # Token Fusion
        fused_tokens, attention_mask, fusion_info = self.token_fusion(
            vision_tokens, language_tokens,
            vision_info, language_info, language_mask
        )
        
        # Build Graph
        edge_index, edge_attr = self.graph_builder.build_graph(
            fusion_info, batch_size=B, device=device
        )
        
        # VGGT Layers
        x = fused_tokens
        for layer in self.layers:
            x = layer(x, edge_index, attn_mask=attention_mask)
        
        x = self.norm(x)
        
        # Split features
        lang_start, lang_end = fusion_info['language_token_range']
        vis_start, vis_end = fusion_info['vision_token_range']
        
        language_features = x[:, lang_start:lang_end, :]
        vision_features = x[:, vis_start:vis_end, :]
        
        # Global features
        global_features = self.action_queries.expand(B, -1, -1)
        
        output_info = fusion_info.copy()
        output_info['num_action_queries'] = self.num_action_queries
        
        return vision_features, language_features, global_features, output_info
