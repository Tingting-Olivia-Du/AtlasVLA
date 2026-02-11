#!/usr/bin/env python3
"""
VLA-VGGT 训练脚本
支持:
- HuggingFace LIBERO 数据集
- 可选的 Vision Tower (DINO/CLIP/SigLIP)
- facebook/vggt 或简化版 VGGT
- Qwen3-0.6B 语言编码器
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import argparse
import yaml
from pathlib import Path
from dataclasses import asdict

from configs.model_config import ModelConfig, VisionConfig, LanguageConfig, VGGTConfig, ActionHeadConfig
from models.vla_model import VLAModel
from data.libero_hf_dataset import get_libero_hf_dataloaders
from training.trainer import Trainer


def parse_args():
    parser = argparse.ArgumentParser(description="Train VLA-VGGT model")
    
    # Data
    parser.add_argument('--dataset_repo', type=str, 
                       default='lerobot/libero_spatial_image',
                       help='HuggingFace dataset repository')
    parser.add_argument('--task_names', nargs='+', default=None,
                       help='Task names to train on (None = all tasks)')
    parser.add_argument('--cache_dir', type=str, default=None,
                       help='HuggingFace cache directory')
    
    # Training
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-5)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--grad_clip', type=float, default=1.0)
    
    # Model Architecture
    parser.add_argument('--use_vision_tower', action='store_true',
                       help='Use pretrained vision tower (DINO/CLIP/SigLIP)')
    parser.add_argument('--vision_tower_name', type=str, 
                       default='facebook/dinov2-base',
                       help='Vision tower model name')
    parser.add_argument('--freeze_vision_tower', action='store_true',
                       help='Freeze vision tower parameters')
    
    parser.add_argument('--language_model', type=str,
                       default='Qwen/Qwen3-0.6B-Base',
                       help='Language model name')
    parser.add_argument('--freeze_language', action='store_true',
                       help='Freeze language encoder')
    
    parser.add_argument('--use_pretrained_vggt', action='store_true',
                       help='Use facebook/vggt from HuggingFace')
    parser.add_argument('--freeze_vggt', action='store_true',
                       help='Freeze VGGT backbone')
    
    parser.add_argument('--action_horizon', type=int, default=10,
                       help='Action prediction horizon')
    parser.add_argument('--action_dim', type=int, default=7,
                       help='Action dimension')
    
    # Logging
    parser.add_argument('--log_dir', type=str, default='./logs',
                       help='Logging directory')
    parser.add_argument('--exp_name', type=str, default=None,
                       help='Experiment name')
    parser.add_argument('--save_freq', type=int, default=10,
                       help='Save checkpoint every N epochs')
    
    # System
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use (cuda/cpu)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    
    # Config file (overrides command line args)
    parser.add_argument('--config', type=str, default=None,
                       help='Path to YAML config file')
    
    args = parser.parse_args()
    
    # Load config file if provided
    if args.config is not None:
        with open(args.config, 'r') as f:
            config_dict = yaml.safe_load(f)
        
        # Update args with config file
        for key, value in config_dict.items():
            if hasattr(args, key):
                setattr(args, key, value)
    
    return args


def build_config(args):
    """从命令行参数构建模型配置"""
    
    # Vision Config
    vision_config = VisionConfig(
        img_size=224,
        patch_size=16,
        in_channels=3,
        embed_dim=768,
        use_vision_tower=args.use_vision_tower,
        vision_tower_name=args.vision_tower_name,
        freeze_vision_tower=args.freeze_vision_tower
    )
    
    # Language Config
    language_config = LanguageConfig(
        model_name=args.language_model,
        max_length=77,
        freeze_encoder=args.freeze_language,
        output_dim=768
    )
    
    # VGGT Config
    vggt_config = VGGTConfig(
        embed_dim=768,
        depth=6,
        num_heads=12,
        mlp_ratio=4.0,
        use_pretrained_vggt=args.use_pretrained_vggt,
        freeze_vggt=args.freeze_vggt,
        graph_type='grid',
        fusion_strategy='concat'
    )
    
    # Action Head Config
    action_head_config = ActionHeadConfig(
        input_dim=768,
        hidden_dim=1024,
        action_dim=args.action_dim,
        action_horizon=args.action_horizon,
        num_hidden_layers=2,
        dropout=0.1,
        use_action_chunking=True,
        use_spatial_features=False
    )
    
    # Full Model Config
    config = ModelConfig(
        vision=vision_config,
        language=language_config,
        vggt=vggt_config,
        action_head=action_head_config,
        hidden_dim=768
    )
    
    return config


def main():
    args = parse_args()
    
    # Set seed
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
    # Create log directory
    if args.exp_name is None:
        exp_name = f"vla_vggt"
        if args.use_vision_tower:
            exp_name += f"_{args.vision_tower_name.split('/')[-1]}"
        if args.use_pretrained_vggt:
            exp_name += "_pretrained"
        args.exp_name = exp_name
    
    log_dir = Path(args.log_dir) / args.exp_name
    log_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "=" * 80)
    print(f"VLA-VGGT Training")
    print("=" * 80)
    print(f"Experiment: {args.exp_name}")
    print(f"Log dir: {log_dir}")
    print(f"Dataset: {args.dataset_repo}")
    print(f"Device: {args.device}")
    print("=" * 80 + "\n")
    
    # Build config
    config = build_config(args)
    
    # Save config
    config_path = log_dir / "config.yaml"
    with open(config_path, 'w') as f:
        yaml.dump({
            'vision': asdict(config.vision),
            'language': asdict(config.language),
            'vggt': asdict(config.vggt),
            'action_head': asdict(config.action_head),
            'training': {
                'batch_size': args.batch_size,
                'num_epochs': args.num_epochs,
                'lr': args.lr,
                'weight_decay': args.weight_decay,
            }
        }, f, default_flow_style=False)
    print(f"✓ Saved config to {config_path}\n")
    
    # Create model
    print("Creating model...")
    model = VLAModel(config)
    model = model.to(args.device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nModel Parameters:")
    print(f"  Total: {total_params / 1e6:.2f}M")
    print(f"  Trainable: {trainable_params / 1e6:.2f}M")
    print(f"  Frozen: {(total_params - trainable_params) / 1e6:.2f}M\n")
    
    # Load data
    print("Loading data...")
    train_loader, val_loader = get_libero_hf_dataloaders(
        repo_id=args.dataset_repo,
        task_names=args.task_names,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        action_horizon=args.action_horizon,
        cache_dir=args.cache_dir
    )
    
    # Create optimizer
    param_groups = model.get_param_groups(
        learning_rate=args.lr,
        weight_decay=args.weight_decay
    )
    optimizer = torch.optim.AdamW(param_groups)
    
    # Create scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.num_epochs,
        eta_min=args.lr * 0.01
    )
    
    # Create trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        config=config,
        device=args.device,
        log_dir=str(log_dir),
        grad_clip=args.grad_clip,
        save_freq=args.save_freq
    )
    
    # Train
    print("\n" + "=" * 80)
    print("Starting Training")
    print("=" * 80 + "\n")
    trainer.train(args.num_epochs)
    
    print("\n" + "=" * 80)
    print("Training Complete!")
    print("=" * 80)
    print(f"Best model saved to: {log_dir / 'best_model.pth'}")
    print(f"Logs saved to: {log_dir}")
    print("=" * 80 + "\n")


if __name__ == '__main__':
    main()
