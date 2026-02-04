"""
Training script for VGGT-based VLA model
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR
import os
import json
from tqdm import tqdm
from typing import Dict, Optional
import wandb  # Optional: for experiment tracking

from ..models import VGGTVLA
from ..data import LIBERODataset
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
        train_dataset: LIBERODataset,
        val_dataset: Optional[LIBERODataset] = None,
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
        **kwargs
    ):
        self.model = model.to(device)
        self.device = device
        self.num_epochs = num_epochs
        self.save_dir = save_dir
        self.log_interval = log_interval
        self.val_interval = val_interval
        self.save_interval = save_interval
        
        # Create save directory
        os.makedirs(save_dir, exist_ok=True)
        
        # Data loaders
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True,
            collate_fn=LIBERODataset.collate_fn
        )
        
        if val_dataset is not None:
            self.val_loader = DataLoader(
                val_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=4,
                pin_memory=True,
                collate_fn=LIBERODataset.collate_fn
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
        
        # Learning rate scheduler
        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=num_epochs * len(self.train_loader),
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
        if use_wandb:
            wandb.init(project=wandb_project, config=kwargs)
            
    def train_epoch(self):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0.0
        total_pose_loss = 0.0
        total_gripper_loss = 0.0
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {self.epoch}")
        
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
                
                loss = loss_dict["total_loss"]
                
            # Backward pass
            self.optimizer.zero_grad()
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.scheduler.step()
            
            # Update statistics
            total_loss += loss.item()
            total_pose_loss += loss_dict["pose_loss"].item()
            total_gripper_loss += loss_dict["gripper_loss"].item()
            self.global_step += 1
            
            # Logging
            if self.global_step % self.log_interval == 0:
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
                    print(f"Step {self.global_step}: Loss={avg_loss:.4f}, "
                          f"Pose={avg_pose_loss:.4f}, Gripper={avg_gripper_loss:.4f}")
                    
                total_loss = 0.0
                total_pose_loss = 0.0
                total_gripper_loss = 0.0
                
            # Validation
            if self.val_loader is not None and self.global_step % self.val_interval == 0:
                val_metrics = self.validate()
                
                if self.use_wandb:
                    wandb.log(val_metrics, step=self.global_step)
                    
            # Save checkpoint
            if self.global_step % self.save_interval == 0:
                self.save_checkpoint(f"checkpoint_step_{self.global_step}.pt")
                
            # Update progress bar
            pbar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "pose": f"{loss_dict['pose_loss'].item():.4f}",
                "gripper": f"{loss_dict['gripper_loss'].item():.4f}"
            })
            
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
        print(f"Starting training for {self.num_epochs} epochs")
        print(f"Trainable parameters: {sum(p.numel() for p in self.model.parameters() if p.requires_grad):,}")
        
        for epoch in range(self.num_epochs):
            self.epoch = epoch
            self.train_epoch()
            
            # Final validation at end of epoch
            if self.val_loader is not None:
                val_metrics = self.validate()
                print(f"Epoch {epoch} validation: {val_metrics}")
                
            # Save epoch checkpoint
            self.save_checkpoint(f"checkpoint_epoch_{epoch}.pt")
            
        print("Training completed!")
        
    def save_checkpoint(self, filename: str):
        """Save training checkpoint"""
        checkpoint = {
            "epoch": self.epoch,
            "global_step": self.global_step,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "scaler_state_dict": self.scaler.state_dict(),
            "best_val_loss": self.best_val_loss,
        }
        
        filepath = os.path.join(self.save_dir, filename)
        torch.save(checkpoint, filepath)
        print(f"Saved checkpoint to {filepath}")
        
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
        
        print(f"Loaded checkpoint from {filepath}")
