"""
Action Head - MLP
"""
import torch
import torch.nn as nn
from typing import Optional

class MLPActionHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.action_dim = config.action_dim
        self.action_horizon = config.action_horizon if config.use_action_chunking else 1
        
        layers = []
        input_dim = config.input_dim
        
        for _ in range(config.num_hidden_layers):
            layers.extend([
                nn.Linear(input_dim, config.hidden_dim),
                nn.LayerNorm(config.hidden_dim),
                nn.ReLU(),
                nn.Dropout(config.dropout)
            ])
            input_dim = config.hidden_dim
        
        output_dim = self.action_dim * self.action_horizon
        layers.append(nn.Linear(input_dim, output_dim))
        
        self.mlp = nn.Sequential(*layers)
        
        self.use_action_normalization = True
        if self.use_action_normalization:
            self.register_buffer('action_mean', torch.zeros(self.action_dim))
            self.register_buffer('action_std', torch.ones(self.action_dim))
    
    def forward(self, global_features: torch.Tensor, 
                normalize: bool = True) -> torch.Tensor:
        
        if global_features.dim() == 3:
            global_features = global_features.mean(dim=1)
        
        action_pred = self.mlp(global_features)
        
        if self.action_horizon > 1:
            action_pred = action_pred.view(-1, self.action_horizon, self.action_dim)
        else:
            action_pred = action_pred.view(-1, self.action_dim)
        
        if normalize and self.use_action_normalization:
            action_pred = action_pred * self.action_std + self.action_mean
        
        return action_pred
    
    def set_action_stats(self, action_mean: torch.Tensor, action_std: torch.Tensor):
        self.action_mean.copy_(action_mean)
        self.action_std.copy_(action_std)


class ActionHeadWithSpatialFeatures(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        self.spatial_attention = nn.Sequential(
            nn.Linear(config.input_dim, config.input_dim // 4),
            nn.ReLU(),
            nn.Linear(config.input_dim // 4, 1)
        )
        
        self.action_mlp = nn.Sequential(
            nn.Linear(config.input_dim * 2, config.hidden_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, config.action_dim * config.action_horizon)
        )
        
        self.action_dim = config.action_dim
        self.action_horizon = config.action_horizon
    
    def forward(self, global_features, vision_features):
        B = vision_features.size(0)
        
        attn_scores = self.spatial_attention(vision_features)
        attn_weights = torch.softmax(attn_scores, dim=1)
        
        spatial_features = (vision_features * attn_weights).sum(dim=1)
        
        combined = torch.cat([global_features, spatial_features], dim=-1)
        
        action_pred = self.action_mlp(combined)
        action_pred = action_pred.view(B, self.action_horizon, self.action_dim)
        
        return action_pred
