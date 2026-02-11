"""
VGGT Backbone
"""
import torch
import torch.nn as nn
from typing import Dict, Tuple

from .components.token_fusion import TokenFusion, AttentionMaskBuilder
from .components.graph_builder import GraphBuilder
from .components.vggt_layers import VGGTLayer

class VGGTBackbone(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        self.token_fusion = TokenFusion(config)
        self.graph_builder = GraphBuilder(config)
        self.mask_builder = AttentionMaskBuilder(config)
        
        self.layers = nn.ModuleList([
            VGGTLayer(config)
            for _ in range(config.depth)
        ])
        
        self.norm = nn.LayerNorm(config.embed_dim)
        
        self.use_action_queries = True
        if self.use_action_queries:
            self.action_queries = nn.Parameter(
                torch.randn(1, 16, config.embed_dim)
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
        
        # Build Attention Mask
        attn_mask_2d = self.mask_builder.build_mask(
            fusion_info, attention_mask,
            allow_language_attend_vision=True,
            allow_vision_attend_language=True
        )
        
        # VGGT Layers
        x = fused_tokens
        for layer in self.layers:
            x = layer(x, edge_index, attn_mask_2d)
        
        x = self.norm(x)
        
        # Split features
        lang_start, lang_end = fusion_info['language_token_range']
        vis_start, vis_end = fusion_info['vision_token_range']
        
        language_features = x[:, lang_start:lang_end, :]
        vision_features = x[:, vis_start:vis_end, :]
        
        # Global features
        if self.use_action_queries:
            queries = self.action_queries.expand(B, -1, -1)
            global_features = queries
        else:
            mask_expanded = attention_mask.unsqueeze(-1).expand_as(x)
            global_features = (x * mask_expanded).sum(dim=1) / mask_expanded.sum(dim=1)
        
        output_info = {
            **fusion_info,
            'has_action_queries': self.use_action_queries,
            'num_action_queries': self.action_queries.size(1) if self.use_action_queries else 0
        }
        
        return vision_features, language_features, global_features, output_info
