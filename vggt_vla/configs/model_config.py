from dataclasses import dataclass
from typing import Optional, Literal

@dataclass
class VisionConfig:
    """视觉编码器配置 - 方案B: 直接使用原始图像"""
    img_size: int = 224
    patch_size: int = 16  # VGGT 默认
    in_channels: int = 3
    embed_dim: int = 768
    use_pretrained_patch_embed: bool = False
    
@dataclass
class LanguageConfig:
    """语言编码器配置"""
    model_name: str = "Qwen/Qwen2-0.5B"  # 使用 Qwen2-0.5B 替代 Qwen3
    max_length: int = 77
    freeze_encoder: bool = True
    use_lora: bool = False
    lora_rank: int = 8
    output_dim: int = 768

@dataclass
class VGGTConfig:
    """VGGT Backbone 配置"""
    embed_dim: int = 768
    depth: int = 6
    num_heads: int = 12
    mlp_ratio: float = 4.0
    
    # Graph structure
    graph_type: Literal['grid', 'knn', 'fully_connected'] = 'grid'
    k_neighbors: int = 9
    
    # Multi-modal fusion
    fusion_strategy: Literal['concat', 'cross_attention', 'interleave'] = 'concat'
    use_separate_graph_for_language: bool = False
    
    # Position encoding
    use_2d_pos_embed_for_vision: bool = True
    use_1d_pos_embed_for_language: bool = True
    
    # Token type
    use_token_type_embeddings: bool = True
    
@dataclass  
class ActionHeadConfig:
    """Action Head 配置"""
    input_dim: int = 768
    hidden_dim: int = 1024
    action_dim: int = 7
    action_horizon: int = 10
    num_hidden_layers: int = 2
    dropout: float = 0.1
    use_action_chunking: bool = True
    use_spatial_features: bool = False

@dataclass
class ModelConfig:
    """完整模型配置"""
    vision: VisionConfig = VisionConfig()
    language: LanguageConfig = LanguageConfig()
    vggt: VGGTConfig = VGGTConfig()
    action_head: ActionHeadConfig = ActionHeadConfig()
    
    hidden_dim: int = 768
