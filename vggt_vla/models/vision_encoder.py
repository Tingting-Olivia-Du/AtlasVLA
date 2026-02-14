"""
视觉编码器 - 支持多种方案
"""
import torch
import torch.nn as nn
from typing import Tuple, Optional
from transformers import AutoModel, AutoImageProcessor


class VisionEncoder(nn.Module):
    """
    灵活的视觉编码器，支持:
    1. 直接 patch embedding (无预训练)
    2. 预训练 vision tower (DINO, SLIP, CLIP等)
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.use_vision_tower = config.use_vision_tower
        
        if self.use_vision_tower:
            self._build_vision_tower(config)
        else:
            self._build_direct_encoder(config)
    
    def _build_direct_encoder(self, config):
        """直接 patch embedding (不使用预训练 vision tower)"""
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
        
        self.projector = None
    
    def _build_vision_tower(self, config):
        """使用预训练 vision tower"""
        print(f"Loading vision tower: {config.vision_tower_name}")
        
        # 支持多种 vision tower
        if 'dinov2' in config.vision_tower_name.lower():
            self.vision_tower = AutoModel.from_pretrained(
                config.vision_tower_name,
                trust_remote_code=True
            )
            self.vision_hidden_size = self.vision_tower.config.hidden_size
        elif 'clip' in config.vision_tower_name.lower():
            from transformers import CLIPVisionModel
            self.vision_tower = CLIPVisionModel.from_pretrained(
                config.vision_tower_name
            )
            self.vision_hidden_size = self.vision_tower.config.hidden_size
        elif 'siglip' in config.vision_tower_name.lower():
            self.vision_tower = AutoModel.from_pretrained(
                config.vision_tower_name,
                trust_remote_code=True
            )
            self.vision_hidden_size = self.vision_tower.config.vision_config.hidden_size
        else:
            raise ValueError(f"Unsupported vision tower: {config.vision_tower_name}")
        
        # Projector to target dimension
        self.projector = nn.Sequential(
            nn.Linear(self.vision_hidden_size, config.embed_dim),
            nn.LayerNorm(config.embed_dim),
            nn.GELU(),
            nn.Linear(config.embed_dim, config.embed_dim)
        )
        
        # Freeze vision tower if specified
        if config.freeze_vision_tower:
            print("Freezing vision tower...")
            for param in self.vision_tower.parameters():
                param.requires_grad = False
        
        # Image processor
        try:
            self.image_processor = AutoImageProcessor.from_pretrained(
                config.vision_tower_name
            )
        except:
            self.image_processor = None
    
    def forward(self, images: torch.Tensor) -> Tuple[torch.Tensor, dict]:
        """
        Args:
            images: [B, 3, H, W]
        Returns:
            vision_tokens: [B, N_patches, D]
            spatial_info: dict
        """
        if self.use_vision_tower:
            return self._forward_vision_tower(images)
        else:
            return self._forward_direct(images)
    
    def _forward_direct(self, images: torch.Tensor) -> Tuple[torch.Tensor, dict]:
        """直接 patch embedding"""
        B = images.size(0)
        
        # Patch embedding
        x = self.patch_embed(images)  # [B, D, H/P, W/P]
        x = x.flatten(2).transpose(1, 2)  # [B, N, D]
        
        # Add positional encoding
        # Ensure pos_embed is on the same device and dtype as x
        pos_embed = self.pos_embed.to(device=x.device, dtype=x.dtype)
        x = x + pos_embed
        x = self.norm(x)
        
        grid_size = self.img_size // self.patch_size
        spatial_info = {
            'grid_size': grid_size,
            'num_patches': self.num_patches,
            'has_spatial_structure': True,
            'patch_positions': self._get_patch_positions(grid_size)
        }
        
        return x, spatial_info
    
    def _forward_vision_tower(self, images: torch.Tensor) -> Tuple[torch.Tensor, dict]:
        """使用预训练 vision tower"""
        B = images.size(0)
        
        # Extract features from vision tower
        with torch.set_grad_enabled(not self.config.freeze_vision_tower):
            if 'dinov2' in self.config.vision_tower_name.lower():
                outputs = self.vision_tower(images, output_hidden_states=True)
                # 使用最后一层的patch tokens (去除CLS token)
                vision_features = outputs.last_hidden_state[:, 1:, :]
            elif 'clip' in self.config.vision_tower_name.lower():
                outputs = self.vision_tower(images, output_hidden_states=True)
                vision_features = outputs.last_hidden_state[:, 1:, :]
            elif 'siglip' in self.config.vision_tower_name.lower():
                outputs = self.vision_tower.vision_model(images)
                vision_features = outputs.last_hidden_state
            else:
                raise NotImplementedError
        
        # Project to target dimension
        # Convert to float32 to match projector dtype (projector weights are float32)
        # This prevents dtype mismatch errors when vision tower outputs bfloat16
        vision_features = vision_features.to(torch.float32)
        vision_tokens = self.projector(vision_features)
        
        # Infer grid size from number of patches
        num_patches = vision_tokens.size(1)
        grid_size = int(num_patches ** 0.5)
        
        spatial_info = {
            'grid_size': grid_size,
            'num_patches': num_patches,
            'has_spatial_structure': True,
            'patch_positions': self._get_patch_positions(grid_size)
        }
        
        return vision_tokens, spatial_info
    
    def _get_patch_positions(self, grid_size: int) -> torch.Tensor:
        positions = []
        for i in range(grid_size):
            for j in range(grid_size):
                positions.append([i, j])
        return torch.tensor(positions)


# 保持向后兼容
DirectVisionEncoder = VisionEncoder
