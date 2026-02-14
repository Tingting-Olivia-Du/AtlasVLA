"""
Training loop (支持单卡 / 多卡 DDP)
日志：文本 train.log + 可选 Weights & Biases (wandb)
"""
import torch
import torch.nn as nn
from tqdm import tqdm
import os
from datetime import datetime

from .losses import action_loss_fn
from .metrics import compute_metrics

# wandb 可选：未安装时仅禁用 wandb 日志
try:
    import wandb
    _HAS_WANDB = True
except ImportError:
    _HAS_WANDB = False


def is_main_process(rank: int = 0) -> bool:
    """是否为主进程 (rank 0)，用于控制日志和保存"""
    if torch.distributed.is_initialized():
        return rank == 0
    return True


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
        log_dir: str = './logs',
        grad_clip: float = 1.0,
        save_freq: int = 10,
        save_every_steps: int = None,
        best_model_save_every_steps: int = None,
        rank: int = 0,
        world_size: int = 1,
        use_wandb: bool = False,
        wandb_project: str = 'vla-vggt',
        wandb_run_name: str = None,
        dataset_name: str = None,
        resume_epoch: int = 0,
        resume_global_step: int = 0,
        resume_best_val_loss: float = None,
        wandb_run_id: str = None,
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.config = config
        self.device = device
        self.grad_clip = grad_clip
        self.save_freq = save_freq
        self.save_every_steps = save_every_steps
        self.best_model_save_every_steps = best_model_save_every_steps
        self.last_best_save_step = 0
        self.last_step_checkpoint_saved = 0
        self.rank = rank
        self.world_size = world_size
        self.is_main = is_main_process(rank)
        self.use_wandb = use_wandb and _HAS_WANDB and self.is_main
        self.wandb_project = wandb_project
        self.wandb_run_name = wandb_run_name or os.path.basename(os.path.normpath(log_dir))
        
        # 数据集名称和时间戳用于 checkpoint 命名
        self.dataset_name = dataset_name or "unknown"
        self.training_start_time = datetime.now().strftime("%Y%m%d_%H%M%S")

        self.log_dir = log_dir
        # 文本日志文件：命名含数据集和时间戳，便于区分多次运行
        self.log_file = None
        if self.is_main:
            dataset_clean = ''.join(c if c.isalnum() or c in ('_', '-') else '_' for c in self.dataset_name)
            dataset_clean = dataset_clean.replace('/', '_').replace('\\', '_')
            log_filename = f"train_{dataset_clean}_{self.training_start_time}.log"
            log_path = os.path.join(log_dir, log_filename)
            self.log_file = open(log_path, "a", encoding="utf-8")
            self.log_file.write(f"\n{'='*60}\nTraining started (log_dir={log_dir})\n{'='*60}\n")
            self.log_file.flush()

        self.current_epoch = resume_epoch
        self.global_step = resume_global_step
        self.best_val_loss = float('inf') if resume_best_val_loss is None else resume_best_val_loss
        if save_every_steps and self.global_step > 0:
            self.last_step_checkpoint_saved = (self.global_step // save_every_steps) * save_every_steps
        if best_model_save_every_steps and self.global_step > 0:
            self.last_best_save_step = self.global_step  # 续训时不立即覆盖 best

        # Weights & Biases
        if self.use_wandb:
            init_kwargs = dict(project=self.wandb_project, name=self.wandb_run_name, dir=log_dir, config=self._config_for_wandb())
            if wandb_run_id:
                init_kwargs['id'] = wandb_run_id
                init_kwargs['resume'] = 'allow'
            wandb.init(**init_kwargs)
            self._log("Weights & Biases logging enabled." + (" (resumed run)" if wandb_run_id else ""))
        elif use_wandb and not _HAS_WANDB and self.is_main:
            print("Warning: use_wandb=True but wandb not installed. Run: pip install wandb")

    def _config_for_wandb(self):
        """把 config 转成 wandb 可记录的 dict（避免不可序列化对象）"""
        from dataclasses import asdict
        try:
            return {
                "vision": asdict(self.config.vision) if hasattr(self.config, 'vision') else {},
                "language": asdict(self.config.language) if hasattr(self.config, 'language') else {},
                "vggt": asdict(self.config.vggt) if hasattr(self.config, 'vggt') else {},
                "action_head": asdict(self.config.action_head) if hasattr(self.config, 'action_head') else {},
            }
        except Exception:
            return {}

    def train_epoch(self, max_steps: int = None):
        self.model.train()
        total_loss = 0

        # DDP: 每个 epoch 打乱 DistributedSampler
        if hasattr(self.train_loader, 'sampler') and self.train_loader.sampler is not None:
            if hasattr(self.train_loader.sampler, 'set_epoch'):
                self.train_loader.sampler.set_epoch(self.current_epoch)

        pbar = tqdm(self.train_loader, desc=f"Epoch {self.current_epoch}", disable=not self.is_main)
        for batch_idx, batch in enumerate(pbar):
            if max_steps and self.global_step >= max_steps:
                break
            images = batch['image'].to(self.device)
            instructions = batch['instruction']
            actions_gt = batch['actions'].to(self.device)
            
            self.optimizer.zero_grad()
            outputs = self.model(images, instructions)
            actions_pred = outputs['actions']
            
            loss = action_loss_fn(actions_pred, actions_gt, self.config)
            
            loss.backward()
            if self.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
            self.optimizer.step()
            
            total_loss += loss.item()
            self.global_step += 1

            # Step-based checkpoint (every save_every_steps)
            if (self.save_every_steps and self.is_main and
                self.global_step >= self.last_step_checkpoint_saved + self.save_every_steps):
                self.last_step_checkpoint_saved = (self.global_step // self.save_every_steps) * self.save_every_steps
                ckpt_name = self._generate_checkpoint_name(prefix='checkpoint', epoch=self.current_epoch, step=self.global_step)
                self.save_checkpoint(ckpt_name)
                self._log(f"  ✓ Saved step checkpoint: {ckpt_name} (step {self.global_step})")
            
            if batch_idx % 10 == 0:
                pbar.set_postfix({'loss': loss.item()})
                if self.use_wandb:
                    wandb.log({"train/loss": loss.item(), "global_step": self.global_step}, step=self.global_step)
        
        avg_loss = total_loss / len(self.train_loader)
        return avg_loss
    
    @torch.no_grad()
    def validate(self):
        self.model.eval()
        total_loss = 0
        all_metrics = []
        
        for batch in tqdm(self.val_loader, desc="Validating", disable=not self.is_main):
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
    
    def _log(self, msg: str):
        """主进程：同时打印并写入日志文件"""
        if self.is_main:
            print(msg)
            if self.log_file is not None:
                self.log_file.write(msg + "\n")
                self.log_file.flush()

    def train(self, num_epochs: int, max_steps: int = None):
        start_epoch = self.current_epoch
        if self.is_main:
            msg = f"Starting training for {num_epochs} epochs (world_size={self.world_size})"
            if max_steps:
                msg += f", max_steps={max_steps}"
            if start_epoch > 0:
                msg += f", resuming from epoch {start_epoch}"
            self._log(msg)

        for epoch in range(start_epoch, num_epochs):
            if max_steps and self.global_step >= max_steps:
                if self.is_main:
                    self._log(f"Reached max_steps={max_steps}, stopping.")
                break
            self.current_epoch = epoch

            train_loss = self.train_epoch(max_steps=max_steps)
            val_loss, val_metrics = self.validate()

            if self.scheduler is not None:
                self.scheduler.step()

            if self.is_main:
                self._log(f"\nEpoch {epoch}:")
                self._log(f"  Train Loss: {train_loss:.4f}")
                self._log(f"  Val Loss: {val_loss:.4f}")
                for key, value in val_metrics.items():
                    self._log(f"  {key}: {value:.4f}")

                if self.use_wandb:
                    log_dict = {"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss}
                    for key, value in val_metrics.items():
                        log_dict[f"val/{key}"] = value
                    wandb.log(log_dict, step=epoch)

            if val_loss < self.best_val_loss:
                should_save_best = self.is_main
                if should_save_best and self.best_model_save_every_steps:
                    should_save_best = (self.global_step - self.last_best_save_step) >= self.best_model_save_every_steps
                if should_save_best:
                    self.best_val_loss = val_loss
                    self.last_best_save_step = self.global_step
                    checkpoint_name = self._generate_checkpoint_name(
                        prefix='best_model',
                        epoch=epoch,
                        val_loss=val_loss,
                        step=self.global_step
                    )
                    self.save_checkpoint(checkpoint_name)
                    self._log(f"  ✓ Saved best model: {checkpoint_name} (val_loss: {val_loss:.4f}, step {self.global_step})")
                else:
                    self.best_val_loss = val_loss

            if (epoch + 1) % self.save_freq == 0 and self.is_main:
                checkpoint_name = self._generate_checkpoint_name(
                    prefix='checkpoint',
                    epoch=epoch + 1
                )
                self.save_checkpoint(checkpoint_name)
                self._log(f"  ✓ Saved checkpoint: {checkpoint_name}")

            if max_steps and self.global_step >= max_steps:
                if self.is_main:
                    self._log(f"Reached max_steps={max_steps}, stopping.")
                break

        if self.is_main:
            self._log("Training complete!")
            if self.use_wandb:
                wandb.finish()
            if self.log_file is not None:
                self.log_file.close()
                self.log_file = None
    
    def _generate_checkpoint_name(self, prefix: str, epoch: int, val_loss: float = None, step: int = None) -> str:
        """生成 checkpoint 文件名，支持 epoch/step/loss"""
        dataset_clean = ''.join(c if c.isalnum() or c in ('_', '-') else '_' for c in self.dataset_name)
        dataset_clean = dataset_clean.replace('/', '_').replace('\\', '_')
        parts = [prefix, dataset_clean, self.training_start_time, f"epoch{epoch}"]
        if step is not None:
            parts.append(f"step{step}")
        if val_loss is not None:
            parts.append(f"loss{val_loss:.4f}")
        return "_".join(parts) + ".pt"
    
    def save_checkpoint(self, filename: str):
        model_for_save = self.model.module if hasattr(self.model, 'module') else self.model
        checkpoint = {
            'epoch': self.current_epoch,
            'global_step': self.global_step,
            'model_state_dict': model_for_save.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_loss': self.best_val_loss,
            'config': self.config,
            'dataset_name': self.dataset_name,
            'training_start_time': self.training_start_time,
        }
        
        if self.scheduler is not None:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        if self.use_wandb and _HAS_WANDB and hasattr(wandb, 'run') and wandb.run is not None:
            checkpoint['wandb_run_id'] = wandb.run.id

        path = os.path.join(self.log_dir, filename)
        torch.save(checkpoint, path)
    
    def load_checkpoint(self, filename: str, path_override: str = None):
        path = path_override if path_override else os.path.join(self.log_dir, filename)
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        state_dict = checkpoint.get('model_state_dict') or checkpoint.get('state_dict')
        target = self.model.module if hasattr(self.model, 'module') else self.model
        target.load_state_dict(state_dict, strict=True)
        if 'optimizer_state_dict' in checkpoint:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        # path_override = resume from external checkpoint: start from next epoch
        next_epoch = checkpoint.get('epoch', 0) + 1 if path_override else checkpoint.get('epoch', 0)
        self.current_epoch = next_epoch
        self.global_step = checkpoint.get('global_step', 0)
        self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        if self.scheduler is not None and 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        return checkpoint.get('wandb_run_id')

