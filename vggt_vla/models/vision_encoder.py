"""
视觉编码器 - 方案B: 直接输入原始图像到 VGGT
"""
import torch
import torch.nn as nn
from typing import Tuple

class DirectVisionEncoder(nn.Module):
    """
    方案B: 不使用预训练 vision tower，直接将图像 patch embedding
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.img_size = config.img_size
        self.patch_size = config.patch_size
        self.num_patches = (config.img_size // config.patch_size) ** 2
        
        # Patch Embedding Layer
        self.patch_embed = nn.Conv2d(
            in_channels=config.in_channels,
            out_channels=config.embed_dim,
            kernel_size=config.patch_size,
            stride=config.patch_size
        )
        
        self.norm = nn.LayerNorm(config.embed_dim)
        
        # 2D Positional Encoding
        self.pos_embed = nn.Parameter(
            torch.zeros(1, self.num_patches, config.embed_dim)
        )
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        
    def forward(self, images: torch.Tensor) -> Tuple[torch.Tensor, dict]:
        """
        Args:
            images: [B, 3, H, W]
        Returns:
            vision_tokens: [B, N_patches, D]
            spatial_info: dict
        """
        B = images.size(0)
        
        # Patch embedding
        x = self.patch_embed(images)
        x = x.flatten(2).transpose(1, 2)
        
        # Add positional encoding
        x = x + self.pos_embed
        x = self.norm(x)
        
        grid_size = self.img_size // self.patch_size
        spatial_info = {
            'grid_size': grid_size,
            'num_patches': self.num_patches,
            'has_spatial_structure': True,
            'patch_positions': self._get_patch_positions(grid_size)
        }
        
        return x, spatial_info
    
    def _get_patch_positions(self, grid_size: int) -> torch.Tensor:
        positions = []
        for i in range(grid_size):
            for j in range(grid_size):
                positions.append([i, j])
        return torch.tensor(positions)
