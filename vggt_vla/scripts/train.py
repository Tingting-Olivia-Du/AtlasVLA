"""
Training script
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import argparse
from configs.model_config import ModelConfig
from models.vla_model import VLAModel
from data.libero_dataset import get_libero_dataloaders
from training.trainer import Trainer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, required=True)
    parser.add_argument('--task_names', nargs='+', default=None)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--log_dir', type=str, default='./logs')
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()
    
    config = ModelConfig()
    
    print("Creating model...")
    model = VLAModel(config).to(args.device)
    
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model has {num_params / 1e6:.2f}M trainable parameters")
    
    print("Loading data...")
    train_loader, val_loader = get_libero_dataloaders(
        data_path=args.data_path,
        task_names=args.task_names,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        action_horizon=config.action_head.action_horizon
    )
    
    param_groups = model.get_param_groups(
        learning_rate=args.lr,
        weight_decay=1e-5
    )
    optimizer = torch.optim.AdamW(param_groups)
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.num_epochs
    )
    
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        config=config,
        device=args.device,
        log_dir=args.log_dir
    )
    
    print("Starting training...")
    trainer.train(args.num_epochs)


if __name__ == '__main__':
    main()
