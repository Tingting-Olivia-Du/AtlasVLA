"""
Training loop
"""
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import os

from .losses import action_loss_fn
from .metrics import compute_metrics


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        train_loader,
        val_loader,
        optimizer,
        scheduler,
        config,
        device: str = 'cuda',
        log_dir: str = './logs'
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.config = config
        self.device = device
        
        self.writer = SummaryWriter(log_dir)
        self.log_dir = log_dir
        
        self.current_epoch = 0
        self.global_step = 0
        self.best_val_loss = float('inf')
    
    def train_epoch(self):
        self.model.train()
        total_loss = 0
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {self.current_epoch}")
        for batch_idx, batch in enumerate(pbar):
            images = batch['image'].to(self.device)
            instructions = batch['instruction']
            actions_gt = batch['actions'].to(self.device)
            
            self.optimizer.zero_grad()
            outputs = self.model(images, instructions)
            actions_pred = outputs['actions']
            
            loss = action_loss_fn(actions_pred, actions_gt, self.config)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            
            total_loss += loss.item()
            self.global_step += 1
            
            if batch_idx % 10 == 0:
                pbar.set_postfix({'loss': loss.item()})
                self.writer.add_scalar('train/loss', loss.item(), self.global_step)
        
        avg_loss = total_loss / len(self.train_loader)
        return avg_loss
    
    @torch.no_grad()
    def validate(self):
        self.model.eval()
        total_loss = 0
        all_metrics = []
        
        for batch in tqdm(self.val_loader, desc="Validating"):
            images = batch['image'].to(self.device)
            instructions = batch['instruction']
            actions_gt = batch['actions'].to(self.device)
            
            outputs = self.model(images, instructions)
            actions_pred = outputs['actions']
            
            loss = action_loss_fn(actions_pred, actions_gt, self.config)
            total_loss += loss.item()
            
            metrics = compute_metrics(actions_pred, actions_gt)
            all_metrics.append(metrics)
        
        avg_loss = total_loss / len(self.val_loader)
        
        avg_metrics = {}
        for key in all_metrics[0].keys():
            avg_metrics[key] = sum(m[key] for m in all_metrics) / len(all_metrics)
        
        return avg_loss, avg_metrics
    
    def train(self, num_epochs: int):
        print(f"Starting training for {num_epochs} epochs")
        
        for epoch in range(num_epochs):
            self.current_epoch = epoch
            
            train_loss = self.train_epoch()
            val_loss, val_metrics = self.validate()
            
            if self.scheduler is not None:
                self.scheduler.step()
            
            print(f"\nEpoch {epoch}:")
            print(f"  Train Loss: {train_loss:.4f}")
            print(f"  Val Loss: {val_loss:.4f}")
            for key, value in val_metrics.items():
                print(f"  {key}: {value:.4f}")
            
            self.writer.add_scalar('epoch/train_loss', train_loss, epoch)
            self.writer.add_scalar('epoch/val_loss', val_loss, epoch)
            for key, value in val_metrics.items():
                self.writer.add_scalar(f'epoch/{key}', value, epoch)
            
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.save_checkpoint('best_model.pth')
                print(f"  Saved best model (val_loss: {val_loss:.4f})")
            
            if (epoch + 1) % 10 == 0:
                self.save_checkpoint(f'checkpoint_epoch_{epoch}.pth')
        
        print("Training complete!")
        self.writer.close()
    
    def save_checkpoint(self, filename: str):
        checkpoint = {
            'epoch': self.current_epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_loss': self.best_val_loss,
            'config': self.config
        }
        
        if self.scheduler is not None:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        
        path = os.path.join(self.log_dir, filename)
        torch.save(checkpoint, path)
    
    def load_checkpoint(self, filename: str):
        path = os.path.join(self.log_dir, filename)
        checkpoint = torch.load(path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.current_epoch = checkpoint['epoch']
        self.global_step = checkpoint['global_step']
        self.best_val_loss = checkpoint['best_val_loss']
        
        if self.scheduler is not None and 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        print(f"Loaded checkpoint from epoch {self.current_epoch}")
