"""
完整 VLA 模型
"""
import torch
import torch.nn as nn
from typing import List, Dict, Tuple, Optional

from .vision_encoder import DirectVisionEncoder
from .language_encoder import LanguageEncoder
from .vggt_backbone import VGGTBackbone
from .action_head import MLPActionHead, ActionHeadWithSpatialFeatures


class VLAModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        self.vision_encoder = DirectVisionEncoder(config.vision)
        self.language_encoder = LanguageEncoder(config.language)
        self.vggt_backbone = VGGTBackbone(config.vggt)
        
        if hasattr(config.action_head, 'use_spatial_features') and            config.action_head.use_spatial_features:
            self.action_head = ActionHeadWithSpatialFeatures(config.action_head)
        else:
            self.action_head = MLPActionHead(config.action_head)
        
        self.use_spatial_action_head = isinstance(
            self.action_head, ActionHeadWithSpatialFeatures
        )
    
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
