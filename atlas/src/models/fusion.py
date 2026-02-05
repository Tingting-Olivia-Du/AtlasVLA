"""
Multimodal Fusion Module
Fuses language and 3D geometry features using cross-attention
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultimodalFusion(nn.Module):
    """
    Fuse language and 3D geometry features using cross-attention
    
    Architecture:
    - Language features attend to geometry features
    - Multiple cross-attention layers for deep fusion
    - Output projection to unified representation
    """
    
    def __init__(self, lang_dim=4096, geom_dim=512, hidden_dim=1024, 
                 num_layers=4, num_heads=16, dropout=0.1):
        super().__init__()
        
        self.lang_dim = lang_dim
        self.geom_dim = geom_dim
        self.hidden_dim = hidden_dim
        
        # Project language and geometry features to unified dimension
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
        
        # Cross-attention layers
        # Language features as query, geometry features as key/value
        self.cross_attn_layers = nn.ModuleList([
            nn.TransformerDecoderLayer(
                d_model=hidden_dim,
                nhead=num_heads,
                dim_feedforward=hidden_dim * 4,
                dropout=dropout,
                activation='gelu',
                batch_first=True,
                norm_first=True  # Pre-norm for better training stability
            ) for _ in range(num_layers)
        ])
        
        # Output projection
        self.output_norm = nn.LayerNorm(hidden_dim)
        self.output_proj = nn.Linear(hidden_dim, hidden_dim)
        
    def forward(self, lang_features, geom_features):
        """
        Args:
            lang_features: [B, L, lang_dim] - Language token features
            geom_features: [B, S, geom_dim] - Geometry features (can be multi-frame)
            
        Returns:
            fused_features: [B, hidden_dim] - Fused representation
        """
        # Project to unified dimension
        lang = self.lang_proj(lang_features)  # [B, L, hidden_dim]
        geom = self.geom_proj(geom_features)  # [B, S, hidden_dim]
        
        # Cross-attention: language queries attend to geometry keys/values
        # This allows language to attend to relevant 3D geometric information
        fused = lang
        for layer in self.cross_attn_layers:
            # In TransformerDecoderLayer:
            # - tgt (query) = lang_features
            # - memory (key/value) = geom_features
            fused = layer(fused, geom)  # [B, L, hidden_dim]
        
        # Aggregate language tokens (mean pooling or use [CLS] token)
        # If language encoder provides [CLS] token, use it; otherwise use mean
        fused = fused.mean(dim=1)  # [B, hidden_dim]
        
        # Final projection
        fused = self.output_norm(fused)
        fused = self.output_proj(fused)  # [B, hidden_dim]
        
        return fused
