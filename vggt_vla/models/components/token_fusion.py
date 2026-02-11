"""
Token 融合策略
"""
import torch
import torch.nn as nn
from typing import Tuple, Dict

class TokenFusion(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.fusion_strategy = config.fusion_strategy
        
        # Token type embeddings
        self.token_type_embeddings = nn.Embedding(3, config.embed_dim)
        
        self.cls_token = nn.Parameter(torch.zeros(1, 1, config.embed_dim))
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        
    def forward(
        self,
        vision_tokens: torch.Tensor,
        language_tokens: torch.Tensor,
        vision_info: Dict,
        language_info: Dict,
        language_mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        
        if self.fusion_strategy == 'concat':
            return self._concat_fusion(
                vision_tokens, language_tokens,
                vision_info, language_info, language_mask
            )
        else:
            raise ValueError(f"Unknown fusion strategy: {self.fusion_strategy}")
    
    def _concat_fusion(
        self,
        vision_tokens, language_tokens,
        vision_info, language_info, language_mask
    ):
        B = vision_tokens.size(0)
        N_v = vision_tokens.size(1)
        N_l = language_tokens.size(1)
        
        # Concatenate [language, vision]
        fused_tokens = torch.cat([language_tokens, vision_tokens], dim=1)
        
        # Token type embeddings
        language_type = torch.ones(B, N_l, dtype=torch.long, device=fused_tokens.device)
        vision_type = torch.zeros(B, N_v, dtype=torch.long, device=fused_tokens.device)
        type_ids = torch.cat([language_type, vision_type], dim=1)
        
        type_embeds = self.token_type_embeddings(type_ids)
        fused_tokens = fused_tokens + type_embeds
        
        # Attention mask
        vision_mask = torch.ones(B, N_v, dtype=torch.long, device=language_mask.device)
        attention_mask = torch.cat([language_mask, vision_mask], dim=1)
        
        fusion_info = {
            'fusion_strategy': 'concat',
            'total_tokens': N_l + N_v,
            'vision_token_range': (N_l, N_l + N_v),
            'language_token_range': (0, N_l),
            'vision_spatial_structure': vision_info['grid_size'],
        }
        
        return fused_tokens, attention_mask, fusion_info


class AttentionMaskBuilder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
    
    def build_mask(
        self,
        fusion_info: Dict,
        attention_mask: torch.Tensor,
        allow_language_attend_vision: bool = True,
        allow_vision_attend_language: bool = True
    ) -> torch.Tensor:
        
        B = attention_mask.size(0)
        N_total = attention_mask.size(1)
        
        # Basic mask
        mask_2d = attention_mask.unsqueeze(1) * attention_mask.unsqueeze(2)
        
        if not allow_language_attend_vision or not allow_vision_attend_language:
            lang_start, lang_end = fusion_info['language_token_range']
            vis_start, vis_end = fusion_info['vision_token_range']
            
            if not allow_language_attend_vision:
                mask_2d[:, lang_start:lang_end, vis_start:vis_end] = 0
            
            if not allow_vision_attend_language:
                mask_2d[:, vis_start:vis_end, lang_start:lang_end] = 0
        
        return mask_2d
