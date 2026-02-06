"""
Training script for VGGT-based VLA model
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, LambdaLR
import os
import json
from tqdm import tqdm
from typing import Dict, Optional
import wandb  # Experiment tracking
import math
import logging

from ..models import VGGTVLA
from ..data import LIBERODataset, LIBEROHFDataset
from .loss import VLALoss


class VLATrainer:
    """
    Trainer for VGGT-based VLA model
    
    Handles:
    - Model initialization
    - Data loading
    - Training loop
    - Validation
    - Checkpointing
    - Logging
    """
    
    def __init__(
        self,
        model: VGGTVLA,
        train_dataset,  # LIBERODataset or LIBEROHFDataset
        val_dataset = None,  # LIBERODataset or LIBEROHFDataset
        batch_size: int = 8,
        learning_rate: float = 1e-4,
        weight_decay: float = 0.01,
        num_epochs: int = 50,
        device: str = "cuda",
        save_dir: str = "./checkpoints",
        log_interval: int = 100,
        val_interval: int = 1000,
        save_interval: int = 5000,
        use_wandb: bool = False,
        wandb_project: str = "atlas-vla",
        wandb_entity: Optional[str] = None,
        wandb_name: Optional[str] = None,
        wandb_tags: Optional[list] = None,
        wandb_notes: Optional[str] = None,
        warmup_steps: int = 0,
        gradient_accumulation_steps: int = 1,
        train_sampler=None,
        val_sampler=None,
        **kwargs
    ):
        self.model = model.to(device)
        self.device = device
        self.num_epochs = num_epochs
        self.save_dir = save_dir
        self.log_interval = log_interval
        self.val_interval = val_interval
        self.save_interval = save_interval
        self.warmup_steps = warmup_steps
        self.gradient_accumulation_steps = gradient_accumulation_steps
        
        # Create save directory
        os.makedirs(save_dir, exist_ok=True)
        
        # Get collate function from dataset (supports both LIBERODataset and LIBEROHFDataset)
        collate_fn = getattr(train_dataset, 'collate_fn', None)
        if collate_fn is None:
            # Fallback: try to get from class
            from atlas.src.data import LIBERODataset, LIBEROHFDataset
            if isinstance(train_dataset, LIBEROHFDataset):
                collate_fn = LIBEROHFDataset.collate_fn
            else:
                collate_fn = LIBERODataset.collate_fn
        
        # Data loaders
        # Use sampler if provided (for distributed training), otherwise use shuffle
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=(train_sampler is None),  # Don't shuffle if using sampler
            sampler=train_sampler,
            num_workers=4,
            pin_memory=True,
            collate_fn=collate_fn
        )
        
        if val_dataset is not None:
            # Get collate function for validation dataset
            val_collate_fn = getattr(val_dataset, 'collate_fn', None)
            if val_collate_fn is None:
                from atlas.src.data import LIBERODataset, LIBEROHFDataset
                if isinstance(val_dataset, LIBEROHFDataset):
                    val_collate_fn = LIBEROHFDataset.collate_fn
                else:
                    val_collate_fn = LIBERODataset.collate_fn
            
            self.val_loader = DataLoader(
                val_dataset,
                batch_size=batch_size,
                shuffle=False,
                sampler=val_sampler,
                num_workers=4,
                pin_memory=True,
                collate_fn=val_collate_fn
            )
        else:
            self.val_loader = None
            
        # Loss function
        self.criterion = VLALoss(**kwargs.get("loss", {}))
        
        # Optimizer - only optimize trainable parameters
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        self.optimizer = AdamW(
            trainable_params,
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        # Learning rate scheduler with warmup
        total_steps = num_epochs * len(self.train_loader)
        if warmup_steps > 0:
            # Warmup + CosineAnnealing
            def lr_lambda(current_step):
                if current_step < warmup_steps:
                    # Linear warmup
                    return float(current_step) / float(max(1, warmup_steps))
                # Cosine annealing after warmup
                progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
                return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))
            
            self.scheduler = LambdaLR(self.optimizer, lr_lambda)
        else:
            # Only CosineAnnealing
            self.scheduler = CosineAnnealingLR(
                self.optimizer,
                T_max=total_steps,
                eta_min=1e-6
            )
        
        # Mixed precision training
        self.scaler = torch.cuda.amp.GradScaler()
        
        # Training state
        self.epoch = 0
        self.global_step = 0
        self.best_val_loss = float('inf')
        
        # Wandb logging
        self.use_wandb = use_wandb
        if self.use_wandb:
            # 准备wandb配置
            wandb_config = {
                "project": wandb_project,
                "config": kwargs,
            }
            
            # 添加可选参数
            if wandb_entity:
                wandb_config["entity"] = wandb_entity
            if wandb_name:
                wandb_config["name"] = wandb_name
            if wandb_tags:
                wandb_config["tags"] = wandb_tags
            if wandb_notes:
                wandb_config["notes"] = wandb_notes
            
            # 初始化wandb
            wandb.init(**wandb_config)
            
            # 记录模型架构信息
            if hasattr(self.model, 'module'):  # DDP模型
                model_to_log = self.model.module
            else:
                model_to_log = self.model
            
            total_params = sum(p.numel() for p in model_to_log.parameters())
            trainable_params = sum(p.numel() for p in model_to_log.parameters() if p.requires_grad)
            
            wandb.config.update({
                "total_parameters": total_params,
                "trainable_parameters": trainable_params,
                "frozen_parameters": total_params - trainable_params,
            })
            
    def train_epoch(self):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0.0
        total_pose_loss = 0.0
        total_gripper_loss = 0.0
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {self.epoch}")
        
        # Reset gradient accumulation
        self.optimizer.zero_grad()
        
        for batch_idx, batch in enumerate(pbar):
            # Move to device
            images = batch["images"].to(self.device)  # [B, S, 3, H, W]
            actions = batch["action"].to(self.device)  # [B, 7]
            language_tasks = batch["language_task"]
            
            # Forward pass with mixed precision
            with torch.cuda.amp.autocast():
                outputs = self.model(images, language_tasks)
                
                # Compute loss
                loss_dict = self.criterion(
                    predictions={
                        "pose": outputs["pose"],
                        "gripper": outputs["gripper"]
                    },
                    targets={
                        "pose": batch["pose"].to(self.device),
                        "gripper": batch["gripper"].to(self.device)
                    }
                )
                
                # Scale loss by accumulation steps
                loss = loss_dict["total_loss"] / self.gradient_accumulation_steps
                
            # Backward pass (accumulate gradients)
            self.scaler.scale(loss).backward()
            
            # Update statistics (use unscaled loss for logging)
            total_loss += loss_dict["total_loss"].item()
            total_pose_loss += loss_dict["pose_loss"].item()
            total_gripper_loss += loss_dict["gripper_loss"].item()
            
            # Update parameters every gradient_accumulation_steps
            if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
                # Gradient clipping (optional but recommended)
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                # Optimizer step
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.scheduler.step()
                self.optimizer.zero_grad()
                
                self.global_step += 1
            
            # Logging (only log when we actually update parameters)
            if (batch_idx + 1) % self.gradient_accumulation_steps == 0 and self.global_step % self.log_interval == 0:
                avg_loss = total_loss / self.log_interval
                avg_pose_loss = total_pose_loss / self.log_interval
                avg_gripper_loss = total_gripper_loss / self.log_interval
                
                log_dict = {
                    "train/loss": avg_loss,
                    "train/pose_loss": avg_pose_loss,
                    "train/gripper_loss": avg_gripper_loss,
                    "train/lr": self.scheduler.get_last_lr()[0],
                    "train/epoch": self.epoch,
                    "train/step": self.global_step,
                }
                
                if self.use_wandb:
                    wandb.log(log_dict, step=self.global_step)
                else:
                    logging.info(f"Step {self.global_step}: Loss={avg_loss:.4f}, "
                                f"Pose={avg_pose_loss:.4f}, Gripper={avg_gripper_loss:.4f}")
                    
                total_loss = 0.0
                total_pose_loss = 0.0
                total_gripper_loss = 0.0
                
            # Validation (only validate when we actually update parameters)
            if (batch_idx + 1) % self.gradient_accumulation_steps == 0 and self.val_loader is not None and self.global_step % self.val_interval == 0:
                val_metrics = self.validate()
                
                if self.use_wandb:
                    wandb.log(val_metrics, step=self.global_step)
                    
            # Save checkpoint (only save when we actually update parameters)
            if (batch_idx + 1) % self.gradient_accumulation_steps == 0 and self.global_step % self.save_interval == 0:
                self.save_checkpoint(f"checkpoint_step_{self.global_step}.pt")
                
            # Update progress bar
            pbar.set_postfix({
                "loss": f"{loss_dict['total_loss'].item():.4f}",
                "pose": f"{loss_dict['pose_loss'].item():.4f}",
                "gripper": f"{loss_dict['gripper_loss'].item():.4f}"
            })
        
        # Handle remaining gradients at the end of epoch
        # (if number of batches is not divisible by gradient_accumulation_steps)
        if len(self.train_loader) % self.gradient_accumulation_steps != 0:
            # Gradient clipping
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            # Optimizer step
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.scheduler.step()
            self.optimizer.zero_grad()
            
            self.global_step += 1
            
    @torch.no_grad()
    def validate(self):
        """Validate on validation set"""
        if self.val_loader is None:
            return {}
            
        self.model.eval()
        total_loss = 0.0
        total_pose_loss = 0.0
        total_gripper_loss = 0.0
        all_metrics = []
        
        for batch in tqdm(self.val_loader, desc="Validating"):
            images = batch["images"].to(self.device)
            language_tasks = batch["language_task"]
            
            with torch.cuda.amp.autocast():
                outputs = self.model(images, language_tasks)
                
                loss_dict = self.criterion(
                    predictions={
                        "pose": outputs["pose"],
                        "gripper": outputs["gripper"]
                    },
                    targets={
                        "pose": batch["pose"].to(self.device),
                        "gripper": batch["gripper"].to(self.device)
                    }
                )
                
                metrics = self.criterion.compute_metrics(
                    predictions={
                        "pose": outputs["pose"],
                        "gripper": outputs["gripper"]
                    },
                    targets={
                        "pose": batch["pose"].to(self.device),
                        "gripper": batch["gripper"].to(self.device)
                    }
                )
                
            total_loss += loss_dict["total_loss"].item()
            total_pose_loss += loss_dict["pose_loss"].item()
            total_gripper_loss += loss_dict["gripper_loss"].item()
            all_metrics.append(metrics)
            
        # Average metrics
        num_batches = len(self.val_loader)
        val_metrics = {
            "val/loss": total_loss / num_batches,
            "val/pose_loss": total_pose_loss / num_batches,
            "val/gripper_loss": total_gripper_loss / num_batches,
        }
        
        # Average other metrics
        for key in all_metrics[0].keys():
            val_metrics[f"val/{key}"] = sum(m[key] for m in all_metrics) / len(all_metrics)
            
        # Save best model
        if val_metrics["val/loss"] < self.best_val_loss:
            self.best_val_loss = val_metrics["val/loss"]
            self.save_checkpoint("best_model.pt")
            
        self.model.train()
        return val_metrics
        
    def train(self):
        """Main training loop"""
        # Check if model is wrapped with DDP
        is_ddp = isinstance(self.model, (torch.nn.DataParallel, torch.nn.parallel.DistributedDataParallel))
        model_to_check = self.model.module if is_ddp else self.model
        
        logging.info(f"Starting training for {self.num_epochs} epochs")
        logging.info(f"Trainable parameters: {sum(p.numel() for p in model_to_check.parameters() if p.requires_grad):,}")
        logging.info(f"Gradient accumulation steps: {self.gradient_accumulation_steps}")
        logging.info(f"Warmup steps: {self.warmup_steps}")
        
        for epoch in range(self.num_epochs):
            self.epoch = epoch
            
            # Set epoch for distributed sampler
            if hasattr(self.train_loader.sampler, 'set_epoch'):
                self.train_loader.sampler.set_epoch(epoch)
            
            self.train_epoch()
            
            # Final validation at end of epoch
            if self.val_loader is not None:
                val_metrics = self.validate()
                logging.info(f"Epoch {epoch} validation: {val_metrics}")
                
            # Save epoch checkpoint
            self.save_checkpoint(f"checkpoint_epoch_{epoch}.pt")
            
        logging.info("Training completed!")
        
    def save_checkpoint(self, filename: str):
        """Save training checkpoint"""
        # Handle DDP model: save only the underlying model
        model_to_save = self.model.module if isinstance(self.model, (torch.nn.DataParallel, torch.nn.parallel.DistributedDataParallel)) else self.model
        
        checkpoint = {
            "epoch": self.epoch,
            "global_step": self.global_step,
            "model_state_dict": model_to_save.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "scaler_state_dict": self.scaler.state_dict(),
            "best_val_loss": self.best_val_loss,
        }
        
        filepath = os.path.join(self.save_dir, filename)
        torch.save(checkpoint, filepath)
        logging.info(f"Saved checkpoint to {filepath}")
        
    def load_checkpoint(self, filepath: str):
        """Load training checkpoint"""
        checkpoint = torch.load(filepath, map_location=self.device)
        
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        self.scaler.load_state_dict(checkpoint["scaler_state_dict"])
        self.epoch = checkpoint["epoch"]
        self.global_step = checkpoint["global_step"]
        self.best_val_loss = checkpoint.get("best_val_loss", float('inf'))
        
        logging.info(f"Loaded checkpoint from {filepath}")
