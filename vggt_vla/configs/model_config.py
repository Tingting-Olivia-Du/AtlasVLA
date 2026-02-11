from dataclasses import dataclass
from typing import Optional, Literal

@dataclass
class VisionConfig:
    """视觉编码器配置"""
    img_size: int = 224
    patch_size: int = 16
    in_channels: int = 3
    embed_dim: int = 768
    
    # Vision tower 选项
    use_vision_tower: bool = False  # 是否使用预训练 vision tower
    vision_tower_name: str = "facebook/dinov2-base"  # 可选: facebook/dinov2-base, openai/clip-vit-base-patch16, google/siglip-base-patch16-224
    freeze_vision_tower: bool = True
    
    # 直接patch embedding (当 use_vision_tower=False 时使用)
    use_pretrained_patch_embed: bool = False
    
@dataclass
class LanguageConfig:
    """语言编码器配置"""
    model_name: str = "Qwen/Qwen3-0.6B-Base"  # 使用 Qwen3-0.6B-Base
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
    
    # 是否使用 HuggingFace 的 facebook/vggt
    use_pretrained_vggt: bool = True  # True: 从HF加载, False: 使用简化实现
    freeze_vggt: bool = False  # 是否冻结VGGT参数
    
    # Graph structure (用于简化版VGGT)
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
