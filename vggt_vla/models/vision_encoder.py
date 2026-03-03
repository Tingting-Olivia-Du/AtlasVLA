"""
视觉编码器 - 支持多种方案
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
from transformers import AutoModel
try:
    from transformers import AutoImageProcessor
except ImportError:
    AutoImageProcessor = None  # transformers < 4.25 无此 API，vision_tower 下 image_processor 将为 None


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
        # Default raw-image normalization target for direct encoder and DINO fallback.
        self.register_buffer(
            "_imagenet_mean",
            torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(1, 3, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "_imagenet_std",
            torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(1, 3, 1, 1),
            persistent=False,
        )
        
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
        tower_name = config.vision_tower_name.lower().strip()
        self.use_fused_tower = tower_name in {
            "dinosiglip-vit-so-224px",
            "dinosiglip-vit-so-384px",
            "dinov2+siglip",
        }

        if self.use_fused_tower:
            # OpenVLA-style dual tower: DINO + SigLIP, then concat + MLP projector.
            dino_name = "facebook/dinov2-base"
            siglip_name = "google/siglip-base-patch16-224"
            self.vision_tower_dino = AutoModel.from_pretrained(
                dino_name, trust_remote_code=True
            )
            self.vision_tower_siglip = AutoModel.from_pretrained(
                siglip_name, trust_remote_code=True
            )
            dino_hidden = self.vision_tower_dino.config.hidden_size
            siglip_hidden = self.vision_tower_siglip.config.vision_config.hidden_size
            self.vision_hidden_size = dino_hidden + siglip_hidden
            print(
                f"  Fused tower enabled: DINO({dino_name}) + SigLIP({siglip_name})"
            )
            self.image_processor_dino = None
            self.image_processor_siglip = None
            if AutoImageProcessor is not None:
                try:
                    self.image_processor_dino = AutoImageProcessor.from_pretrained(dino_name)
                except Exception:
                    self.image_processor_dino = None
                try:
                    self.image_processor_siglip = AutoImageProcessor.from_pretrained(siglip_name)
                except Exception:
                    self.image_processor_siglip = None
        else:
            # 支持单塔 vision tower
            if 'dinov2' in tower_name:
                self.vision_tower = AutoModel.from_pretrained(
                    config.vision_tower_name,
                    trust_remote_code=True
                )
                self.vision_hidden_size = self.vision_tower.config.hidden_size
            elif 'clip' in tower_name:
                from transformers import CLIPVisionModel
                self.vision_tower = CLIPVisionModel.from_pretrained(
                    config.vision_tower_name
                )
                self.vision_hidden_size = self.vision_tower.config.hidden_size
            elif 'siglip' in tower_name:
                self.vision_tower = AutoModel.from_pretrained(
                    config.vision_tower_name,
                    trust_remote_code=True
                )
                self.vision_hidden_size = self.vision_tower.config.vision_config.hidden_size
            else:
                raise ValueError(f"Unsupported vision tower: {config.vision_tower_name}")

        # Projector to target dimension (single tower and fused tower share same head).
        self.projector = nn.Sequential(
            nn.Linear(self.vision_hidden_size, config.embed_dim),
            nn.LayerNorm(config.embed_dim),
            nn.GELU(),
            nn.Linear(config.embed_dim, config.embed_dim)
        )

        # Freeze vision tower if specified
        if config.freeze_vision_tower:
            print("Freezing vision tower...")
            if self.use_fused_tower:
                for param in self.vision_tower_dino.parameters():
                    param.requires_grad = False
                for param in self.vision_tower_siglip.parameters():
                    param.requires_grad = False
            else:
                for param in self.vision_tower.parameters():
                    param.requires_grad = False

        # Image processor
        if self.use_fused_tower:
            self.image_processor = None
        else:
            try:
                self.image_processor = AutoImageProcessor.from_pretrained(
                    config.vision_tower_name
                )
            except Exception:
                self.image_processor = None
    
    def forward(self, images: torch.Tensor) -> Tuple[torch.Tensor, dict]:
        """
        Args:
            images: [B, 3, H, W] 单视角 或 [B, 2, 3, H, W] 双视角 (agentview + wrist)
        Returns:
            vision_tokens: [B, N_patches, D]  (双视角时 N_patches = 2 * num_patches_per_view)
            spatial_info: dict (含 num_patches, patch_positions 等)
        """
        # 单帧多视角: [B, 2, 3, H, W] -> 编码后 [B, 2*P, D]
        if images.dim() == 5 and images.size(1) == 2:
            B, V, C, H, W = images.shape
            images_flat = images.view(B * V, C, H, W)
            if self.use_vision_tower:
                tokens_flat, spatial_info = self._forward_vision_tower(images_flat)
            else:
                tokens_flat, spatial_info = self._forward_direct(images_flat)
            P = tokens_flat.size(1)
            vision_tokens = tokens_flat.view(B, V * P, tokens_flat.size(-1))
            grid_size = spatial_info.get("grid_size", int(P ** 0.5))
            pos_single = self._get_patch_positions(grid_size)
            if not isinstance(pos_single, torch.Tensor):
                pos_single = torch.tensor(pos_single, device=vision_tokens.device, dtype=vision_tokens.dtype)
            else:
                pos_single = pos_single.to(device=vision_tokens.device, dtype=vision_tokens.dtype)
            patch_positions = torch.cat([pos_single, pos_single], dim=0)
            spatial_info = {
                "grid_size": grid_size,
                "num_patches": V * P,
                "has_spatial_structure": True,
                "patch_positions": patch_positions,
                "num_views": V,
                "single_frame_input": False,
            }
            return vision_tokens, spatial_info
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
        if getattr(self, "use_fused_tower", False):
            dino_images, siglip_images = self._preprocess_fused_tower_inputs(images)
            vision_features = self._forward_fused_vision_tower(dino_images, siglip_images)
        else:
            images = self._preprocess_single_tower_input(images)
            # Extract features from single vision tower
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

    def _forward_fused_vision_tower(self, dino_images: torch.Tensor, siglip_images: torch.Tensor) -> torch.Tensor:
        """OpenVLA-style fused dual tower: DINO + SigLIP."""
        with torch.set_grad_enabled(not self.config.freeze_vision_tower):
            dino_out = self.vision_tower_dino(dino_images, output_hidden_states=True)
            dino_tokens = dino_out.last_hidden_state[:, 1:, :]  # remove CLS

            siglip_out = self.vision_tower_siglip.vision_model(siglip_images)
            siglip_tokens = siglip_out.last_hidden_state

        dino_tokens, siglip_tokens = self._align_patch_token_length(dino_tokens, siglip_tokens)
        return torch.cat([dino_tokens, siglip_tokens], dim=-1)

    def _align_patch_token_length(
        self, token_a: torch.Tensor, token_b: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Align two patch token sequences to same patch count via 2D interpolation."""
        if token_a.size(1) == token_b.size(1):
            return token_a, token_b

        n_a = token_a.size(1)
        n_b = token_b.size(1)
        g_a = int(n_a ** 0.5)
        g_b = int(n_b ** 0.5)
        if g_a * g_a != n_a or g_b * g_b != n_b:
            # Fallback for non-square token counts.
            min_n = min(n_a, n_b)
            return token_a[:, :min_n, :], token_b[:, :min_n, :]

        if n_a < n_b:
            token_a = self._resize_patch_grid(token_a, src_grid=g_a, tgt_grid=g_b)
        else:
            token_b = self._resize_patch_grid(token_b, src_grid=g_b, tgt_grid=g_a)
        return token_a, token_b

    @staticmethod
    def _resize_patch_grid(tokens: torch.Tensor, src_grid: int, tgt_grid: int) -> torch.Tensor:
        bsz, _, dim = tokens.shape
        x = tokens.view(bsz, src_grid, src_grid, dim).permute(0, 3, 1, 2).contiguous()
        x = F.interpolate(x, size=(tgt_grid, tgt_grid), mode="bilinear", align_corners=False)
        return x.permute(0, 2, 3, 1).contiguous().view(bsz, tgt_grid * tgt_grid, dim)
    
    def _get_patch_positions(self, grid_size: int) -> torch.Tensor:
        positions = []
        for i in range(grid_size):
            for j in range(grid_size):
                positions.append([i, j])
        return torch.tensor(positions)

    @staticmethod
    def _extract_processor_size(image_processor) -> Optional[int]:
        if image_processor is None:
            return None
        size = getattr(image_processor, "size", None)
        if isinstance(size, dict):
            if "shortest_edge" in size:
                return int(size["shortest_edge"])
            if "height" in size and "width" in size and int(size["height"]) == int(size["width"]):
                return int(size["height"])
        if isinstance(size, (tuple, list)) and len(size) == 2 and int(size[0]) == int(size[1]):
            return int(size[0])
        if isinstance(size, int):
            return int(size)
        return None

    @staticmethod
    def _extract_processor_mean_std(image_processor, default_mean, default_std):
        if image_processor is None:
            return default_mean, default_std
        mean = getattr(image_processor, "image_mean", default_mean)
        std = getattr(image_processor, "image_std", default_std)
        if not isinstance(mean, (list, tuple)) or len(mean) != 3:
            mean = default_mean
        if not isinstance(std, (list, tuple)) or len(std) != 3:
            std = default_std
        return list(mean), list(std)

    def _normalize_images(
        self,
        images: torch.Tensor,
        mean,
        std,
        target_size: Optional[int] = None,
    ) -> torch.Tensor:
        x = images
        if target_size is not None and x.shape[-1] != target_size:
            x = F.interpolate(x, size=(target_size, target_size), mode="bilinear", align_corners=False)
        x = x.clamp(0.0, 1.0)
        mean_t = torch.tensor(mean, dtype=x.dtype, device=x.device).view(1, 3, 1, 1)
        std_t = torch.tensor(std, dtype=x.dtype, device=x.device).view(1, 3, 1, 1)
        return (x - mean_t) / std_t

    def _preprocess_single_tower_input(self, images: torch.Tensor) -> torch.Tensor:
        size = self._extract_processor_size(self.image_processor)
        if "siglip" in self.config.vision_tower_name.lower():
            default_mean, default_std = [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]
        else:
            default_mean, default_std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
        mean, std = self._extract_processor_mean_std(self.image_processor, default_mean, default_std)
        return self._normalize_images(images, mean=mean, std=std, target_size=size)

    def _preprocess_fused_tower_inputs(self, images: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        dino_size = self._extract_processor_size(self.image_processor_dino) or 224
        siglip_size = self._extract_processor_size(self.image_processor_siglip) or 224
        dino_mean, dino_std = self._extract_processor_mean_std(
            self.image_processor_dino, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
        )
        siglip_mean, siglip_std = self._extract_processor_mean_std(
            self.image_processor_siglip, [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]
        )
        dino_images = self._normalize_images(images, mean=dino_mean, std=dino_std, target_size=dino_size)
        siglip_images = self._normalize_images(images, mean=siglip_mean, std=siglip_std, target_size=siglip_size)
        return dino_images, siglip_images


# 保持向后兼容
DirectVisionEncoder = VisionEncoder
