#!/usr/bin/env python3
"""
测试 facebook/vggt + Qwen3-0.6B-Base 配置
验证单帧输入处理
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import argparse
from configs.model_config import ModelConfig, VisionConfig, LanguageConfig, VGGTConfig, ActionHeadConfig
from models.vla_model import VLAModel


def test_vggt_qwen3():
    """测试 VGGT + Qwen3 配置"""
    
    print("\n" + "=" * 80)
    print("Testing VLA Model: facebook/vggt + Qwen3-0.6B-Base")
    print("Single Frame Input")
    print("=" * 80 + "\n")
    
    # 创建配置
    vision_config = VisionConfig(
        use_vision_tower=False,  # 不使用vision tower，更快
        img_size=224,
        patch_size=16,
        embed_dim=768
    )
    
    language_config = LanguageConfig(
        model_name="Qwen/Qwen3-0.6B-Base",  # ✅ Qwen3-0.6B-Base
        freeze_encoder=True,
        output_dim=768,
        max_length=77
    )
    
    vggt_config = VGGTConfig(
        use_pretrained_vggt=True,  # ✅ 使用 facebook/vggt
        freeze_vggt=True,  # 冻结VGGT
        embed_dim=768,
        depth=6
    )
    
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
    
    try:
        model = VLAModel(config).to(device)
    except Exception as e:
        print(f"\n✗ Error creating model: {e}")
        print("\nTroubleshooting:")
        print("1. Make sure you have internet connection for HuggingFace")
        print("2. Or install local vggt: cd ../vggt && pip install -e .")
        print("3. Check if transformers is updated: pip install -U transformers")
        return False
    
    # 统计参数
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\n{'='*80}")
    print("Model Statistics")
    print(f"{'='*80}")
    print(f"Device: {device}")
    print(f"Total parameters: {total_params / 1e6:.2f}M")
    print(f"Trainable parameters: {trainable_params / 1e6:.2f}M")
    print(f"Frozen parameters: {(total_params - trainable_params) / 1e6:.2f}M")
    print(f"Trainable ratio: {trainable_params / total_params * 100:.1f}%")
    
    # 测试单帧输入
    print(f"\n{'='*80}")
    print("Testing Single Frame Forward Pass")
    print(f"{'='*80}")
    
    batch_size = 2  # 小batch size用于测试
    
    # ✅ 单帧输入: [B, 3, H, W]
    images = torch.randn(batch_size, 3, 224, 224).to(device)
    instructions = [
        "pick up the red block and place it on the table",
        "open the drawer and put the cup inside"
    ]
    
    print(f"\nInput:")
    print(f"  Images: {images.shape} (single frame per sample)")
    print(f"  Instructions: {len(instructions)} texts")
    print(f"    - '{instructions[0]}'")
    print(f"    - '{instructions[1]}'")
    
    try:
        model.eval()
        with torch.no_grad():
            outputs = model(images, instructions, return_features=True)
        
        print(f"\n✓ Forward pass successful!")
        print(f"\nOutput shapes:")
        print(f"  Actions: {outputs['actions'].shape}")
        print(f"    - Batch size: {outputs['actions'].shape[0]}")
        print(f"    - Action horizon: {outputs['actions'].shape[1]}")
        print(f"    - Action dim: {outputs['actions'].shape[2]}")
        
        if 'vision_features' in outputs:
            print(f"  Vision features: {outputs['vision_features'].shape}")
        if 'language_features' in outputs:
            print(f"  Language features: {outputs['language_features'].shape}")
        if 'global_features' in outputs:
            print(f"  Global features: {outputs['global_features'].shape}")
        
        if 'output_info' in outputs:
            info = outputs['output_info']
            if 'single_frame_input' in info:
                print(f"\n  ✓ Single frame processing confirmed: {info['single_frame_input']}")
        
        # 测试预测
        print(f"\n{'='*80}")
        print("Testing Action Prediction")
        print(f"{'='*80}")
        
        action = model.predict_action(images, instructions)
        print(f"\nPredicted action shape: {action.shape}")
        print(f"Action statistics:")
        print(f"  Mean: {action.mean():.3f}")
        print(f"  Std: {action.std():.3f}")
        print(f"  Range: [{action.min():.3f}, {action.max():.3f}]")
        
        print(f"\n{'='*80}")
        print("✓ All tests passed!")
        print(f"{'='*80}\n")
        
        print("Summary:")
        print("  ✓ facebook/vggt loaded successfully")
        print("  ✓ Qwen3-0.6B-Base integrated")
        print("  ✓ Single frame input working")
        print("  ✓ Action prediction working")
        print(f"  ✓ Model ready for training ({trainable_params / 1e6:.1f}M trainable params)")
        
        return True
        
    except Exception as e:
        print(f"\n✗ Error during forward pass:")
        print(f"  {str(e)}")
        import traceback
        traceback.print_exc()
        
        print("\nDebugging tips:")
        print("1. Check if VGGT aggregator is accessible")
        print("2. Verify tensor shapes and device placement")
        print("3. Try with smaller batch size")
        
        return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--verbose', action='store_true',
                       help='Print verbose output')
    args = parser.parse_args()
    
    success = test_vggt_qwen3()
    
    if success:
        print("\n" + "="*80)
        print("✓ Model is ready for training!")
        print("="*80)
        print("\nNext steps:")
        print("1. Train with: bash scripts/quick_start.sh configs/train_vggt_qwen3.yaml")
        print("2. Or with DINOv2: bash scripts/quick_start.sh configs/train_vggt_qwen3_dinov2.yaml")
        print("3. Monitor with: set use_wandb: true in config and run wandb at wandb.ai")
        print("="*80 + "\n")
        exit(0)
    else:
        print("\n" + "="*80)
        print("✗ Model test failed. Please check the errors above.")
        print("="*80 + "\n")
        exit(1)


if __name__ == '__main__':
    main()
