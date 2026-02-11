#!/usr/bin/env python3
"""
测试模型初始化和前向传播
用于验证架构设置正确
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import argparse
from configs.model_config import ModelConfig, VisionConfig, LanguageConfig, VGGTConfig, ActionHeadConfig
from models.vla_model import VLAModel


def test_model(config_type='simple'):
    """测试模型初始化和前向传播"""
    
    print("=" * 60)
    print(f"Testing VLA Model - {config_type} configuration")
    print("=" * 60 + "\n")
    
    # 创建配置
    if config_type == 'simple':
        vision_config = VisionConfig(
            use_vision_tower=False,
            embed_dim=768
        )
        language_config = LanguageConfig(
            model_name="Qwen/Qwen2-0.5B",
            freeze_encoder=True,
            output_dim=768
        )
        vggt_config = VGGTConfig(
            use_pretrained_vggt=False,
            embed_dim=768,
            depth=6
        )
    elif config_type == 'dinov2':
        vision_config = VisionConfig(
            use_vision_tower=True,
            vision_tower_name="facebook/dinov2-base",
            freeze_vision_tower=True,
            embed_dim=768
        )
        language_config = LanguageConfig(
            model_name="Qwen/Qwen2-0.5B",
            freeze_encoder=True,
            output_dim=768
        )
        vggt_config = VGGTConfig(
            use_pretrained_vggt=False,
            embed_dim=768,
            depth=6
        )
    else:
        raise ValueError(f"Unknown config type: {config_type}")
    
    action_head_config = ActionHeadConfig(
        input_dim=768,
        action_dim=7,
        action_horizon=10
    )
    
    config = ModelConfig(
        vision=vision_config,
        language=language_config,
        vggt=vggt_config,
        action_head=action_head_config
    )
    
    # 创建模型
    print("Creating model...")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = VLAModel(config).to(device)
    
    # 统计参数
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\n{'='*60}")
    print("Model Statistics")
    print(f"{'='*60}")
    print(f"Device: {device}")
    print(f"Total parameters: {total_params / 1e6:.2f}M")
    print(f"Trainable parameters: {trainable_params / 1e6:.2f}M")
    print(f"Frozen parameters: {(total_params - trainable_params) / 1e6:.2f}M")
    
    # 测试前向传播
    print(f"\n{'='*60}")
    print("Testing Forward Pass")
    print(f"{'='*60}")
    
    batch_size = 4
    images = torch.randn(batch_size, 3, 224, 224).to(device)
    instructions = [
        "pick up the red block",
        "place the cup on the table",
        "push the button",
        "open the drawer"
    ]
    
    print(f"\nInput:")
    print(f"  Images: {images.shape}")
    print(f"  Instructions: {len(instructions)} texts")
    
    try:
        model.eval()
        with torch.no_grad():
            outputs = model(images, instructions, return_features=True)
        
        print(f"\n✓ Forward pass successful!")
        print(f"\nOutput shapes:")
        print(f"  Actions: {outputs['actions'].shape}")
        if 'vision_features' in outputs:
            print(f"  Vision features: {outputs['vision_features'].shape}")
        if 'language_features' in outputs:
            print(f"  Language features: {outputs['language_features'].shape}")
        if 'global_features' in outputs:
            print(f"  Global features: {outputs['global_features'].shape}")
        
        # 测试预测
        print(f"\n{'='*60}")
        print("Testing Action Prediction")
        print(f"{'='*60}")
        
        action = model.predict_action(images, instructions)
        print(f"\nPredicted action: {action.shape}")
        print(f"Action range: [{action.min():.3f}, {action.max():.3f}]")
        
        print(f"\n{'='*60}")
        print("✓ All tests passed!")
        print(f"{'='*60}\n")
        
        return True
        
    except Exception as e:
        print(f"\n✗ Error during forward pass:")
        print(f"  {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='simple',
                       choices=['simple', 'dinov2'],
                       help='Configuration to test')
    args = parser.parse_args()
    
    success = test_model(args.config)
    
    if success:
        print("\n" + "="*60)
        print("Model is ready for training!")
        print("="*60 + "\n")
        exit(0)
    else:
        print("\n" + "="*60)
        print("Model test failed. Please check the errors above.")
        print("="*60 + "\n")
        exit(1)


if __name__ == '__main__':
    main()
