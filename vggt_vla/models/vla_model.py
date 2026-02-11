"""
完整 VLA 模型
"""
import torch
import torch.nn as nn
from typing import List, Dict, Tuple, Optional

from .vision_encoder import VisionEncoder
from .language_encoder import LanguageEncoder
from .vggt_adapter import VGGTAdapter, SimpleVGGTBackbone
from .action_head import MLPActionHead, ActionHeadWithSpatialFeatures


class VLAModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        print("=" * 60)
        print("Initializing VLA Model")
        print("=" * 60)
        
        # Vision Encoder
        print("\n[1/4] Vision Encoder")
        self.vision_encoder = VisionEncoder(config.vision)
        if config.vision.use_vision_tower:
            print(f"  ✓ Using vision tower: {config.vision.vision_tower_name}")
        else:
            print(f"  ✓ Using direct patch embedding")
        
        # Language Encoder
        print("\n[2/4] Language Encoder")
        self.language_encoder = LanguageEncoder(config.language)
        
        # VGGT Backbone
        print("\n[3/4] VGGT Backbone")
        if config.vggt.use_pretrained_vggt:
            print("  Using facebook/vggt from HuggingFace")
            self.vggt_backbone = VGGTAdapter(config.vggt)
        else:
            print("  Using simplified VGGT implementation")
            self.vggt_backbone = SimpleVGGTBackbone(config.vggt)
        
        # Action Head
        print("\n[4/4] Action Head")
        if hasattr(config.action_head, 'use_spatial_features') and config.action_head.use_spatial_features:
            self.action_head = ActionHeadWithSpatialFeatures(config.action_head)
            print("  ✓ Using spatial attention action head")
        else:
            self.action_head = MLPActionHead(config.action_head)
            print("  ✓ Using MLP action head")
        
        self.use_spatial_action_head = isinstance(
            self.action_head, ActionHeadWithSpatialFeatures
        )
        
        print("\n" + "=" * 60)
        print("VLA Model Initialized Successfully")
        print("=" * 60 + "\n")
    
    def forward(
        self, 
        images: torch.Tensor,
        instructions: List[str],
        return_features: bool = False
    ) -> Dict[str, torch.Tensor]:
        
        # Vision encoding
        vision_tokens, vision_info = self.vision_encoder(images)
        
        # Language encoding
        language_tokens, language_mask, language_info = self.language_encoder(instructions)
        
        # VGGT backbone
        vision_features, language_features, global_features, output_info =             self.vggt_backbone(
                vision_tokens, 
                language_tokens,
                vision_info,
                language_info,
                language_mask
            )
        
        # Action prediction
        if self.use_spatial_action_head:
            actions = self.action_head(global_features, vision_features)
        else:
            actions = self.action_head(global_features)
        
        outputs = {'actions': actions}
        
        if return_features:
            outputs.update({
                'vision_features': vision_features,
                'language_features': language_features,
                'global_features': global_features,
                'vision_info': vision_info,
                'language_info': language_info,
                'output_info': output_info
            })
        
        return outputs
    
    def predict_action(
        self,
        images: torch.Tensor,
        instructions: List[str],
        deterministic: bool = True
    ) -> torch.Tensor:
        
        self.eval()
        with torch.no_grad():
            outputs = self.forward(images, instructions)
            actions = outputs['actions']
            
            if actions.dim() == 3:
                actions = actions[:, 0, :]
            
            return actions
    
    def get_param_groups(self, learning_rate: float, weight_decay: float):
        vision_params = list(self.vision_encoder.parameters())
        language_proj_params = list(self.language_encoder.projector.parameters())
        backbone_params = list(self.vggt_backbone.parameters())
        action_params = list(self.action_head.parameters())
        
        param_groups = [
            {
                'params': vision_params,
                'lr': learning_rate,
                'weight_decay': weight_decay
            },
            {
                'params': language_proj_params,
                'lr': learning_rate * 0.1,
                'weight_decay': weight_decay
            },
            {
                'params': backbone_params,
                'lr': learning_rate,
                'weight_decay': weight_decay
            },
            {
                'params': action_params,
                'lr': learning_rate * 2,
                'weight_decay': weight_decay * 0.1
            }
        ]
        
        return param_groups
