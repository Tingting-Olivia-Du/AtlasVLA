"""
Training loop (支持单卡 / 多卡 DDP)
日志：文本 train.log + 可选 Weights & Biases (wandb)
"""
import torch
import torch.nn as nn
from tqdm import tqdm
import os
import sys
import json
import glob
import subprocess
import signal
from datetime import datetime
from typing import Optional, List

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
        wandb_entity: str = None,
        wandb_project: str = 'vla-vggt',
        wandb_run_name: str = None,
        dataset_name: str = None,
        resume_epoch: int = 0,
        resume_global_step: int = 0,
        resume_best_val_loss: float = None,
        wandb_run_id: str = None,
        action_mean=None,
        action_std=None,
        online_eval_every_steps: int = None,
        online_eval_benchmark: str = "libero_spatial",
        online_eval_task_ids: Optional[List[int]] = None,
        online_eval_num_episodes: int = 2,
        online_eval_max_steps: int = 220,
        online_eval_num_envs: int = 1,
        online_eval_action_chunk_size: int = 8,
        online_eval_max_init_states: Optional[int] = 2,
        online_eval_output_dir: Optional[str] = None,
        online_eval_device: Optional[str] = None,
        online_eval_save_videos: bool = False,
        online_eval_use_multi_view: bool = True,
        scheduler_step_per_batch: bool = False,
    ):
        self.model = model
        self.action_mean = action_mean  # (action_dim,) or None；不为 None 时训练用归一化 target 与 action_normalize=False
        self.action_std = action_std
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.scheduler_step_per_batch = scheduler_step_per_batch
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
        self.distributed = world_size > 1
        self.use_wandb = use_wandb and _HAS_WANDB and self.is_main
        self.wandb_entity = wandb_entity
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

        # Online eval hook (small LIBERO simulation eval every N steps)
        self.online_eval_every_steps = int(online_eval_every_steps) if online_eval_every_steps else None
        self.online_eval_benchmark = online_eval_benchmark
        self.online_eval_task_ids = online_eval_task_ids if online_eval_task_ids is not None else [0, 1, 2]
        self.online_eval_num_episodes = int(online_eval_num_episodes)
        self.online_eval_max_steps = int(online_eval_max_steps)
        self.online_eval_num_envs = int(online_eval_num_envs)
        self.online_eval_action_chunk_size = int(online_eval_action_chunk_size)
        self.online_eval_max_init_states = online_eval_max_init_states
        self.online_eval_output_dir = online_eval_output_dir or os.path.join(log_dir, "online_eval")
        self.online_eval_device = online_eval_device or str(device)
        self.online_eval_save_videos = bool(online_eval_save_videos)
        self.online_eval_use_multi_view = bool(online_eval_use_multi_view)
        self.last_online_eval_step = (
            (self.global_step // self.online_eval_every_steps) * self.online_eval_every_steps
            if self.online_eval_every_steps and self.global_step > 0 else 0
        )
        if self.is_main and self.online_eval_every_steps:
            os.makedirs(self.online_eval_output_dir, exist_ok=True)
            self._log(
                "Online eval hook enabled: every %d steps, benchmark=%s, tasks=%s, episodes=%d"
                % (
                    self.online_eval_every_steps,
                    self.online_eval_benchmark,
                    self.online_eval_task_ids,
                    self.online_eval_num_episodes,
                )
            )

        # Weights & Biases
        if self.use_wandb:
            init_kwargs = dict(project=self.wandb_project, name=self.wandb_run_name, dir=log_dir, config=self._config_for_wandb())
            if self.wandb_entity:
                init_kwargs['entity'] = self.wandb_entity
            if wandb_run_id:
                init_kwargs['id'] = wandb_run_id
                init_kwargs['resume'] = 'allow'
            run = wandb.init(**init_kwargs)
            self._log("Weights & Biases logging enabled (project=%s, entity=%s)." % (self.wandb_project, self.wandb_entity or "default"))
            if run is not None and getattr(run, 'url', None):
                self._log("  Run: %s" % run.url)
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
            if self.action_mean is not None and self.action_std is not None:
                mean = self.action_mean.to(actions_gt.device)
                std = self.action_std.to(actions_gt.device)
                actions_gt = (actions_gt - mean) / std
            self.optimizer.zero_grad()
            outputs = self.model(
                images, instructions,
                action_normalize=(self.action_mean is None or self.action_std is None)
            )
            actions_pred = outputs['actions']
            loss = action_loss_fn(actions_pred, actions_gt, self.config)
            
            loss.backward()
            if self.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
            self.optimizer.step()
            if self.scheduler is not None and self.scheduler_step_per_batch:
                self.scheduler.step()
            
            total_loss += loss.item()
            self.global_step += 1

            # Step-based checkpoint (every save_every_steps)
            if (self.save_every_steps and self.is_main and
                self.global_step >= self.last_step_checkpoint_saved + self.save_every_steps):
                self.last_step_checkpoint_saved = (self.global_step // self.save_every_steps) * self.save_every_steps
                ckpt_name = self._generate_checkpoint_name(prefix='checkpoint', epoch=self.current_epoch, step=self.global_step)
                self.save_checkpoint(ckpt_name)
                self._log(f"  ✓ Saved step checkpoint: {ckpt_name} (step {self.global_step})")

            self._maybe_run_online_eval()
            
            if batch_idx % 10 == 0:
                pbar.set_postfix({'loss': loss.item()})
                if self.use_wandb:
                    wandb.log({"train/loss": loss.item(), "global_step": self.global_step}, step=self.global_step)
        
        avg_loss = total_loss / len(self.train_loader)
        return avg_loss

    def _maybe_run_online_eval(self):
        """DDP-safe online eval trigger."""
        if not self.online_eval_every_steps:
            return
        if self.global_step < self.last_online_eval_step + self.online_eval_every_steps:
            return

        trigger_step = (self.global_step // self.online_eval_every_steps) * self.online_eval_every_steps

        if torch.distributed.is_initialized():
            torch.distributed.barrier()
        if self.is_main:
            self._run_online_eval_once(trigger_step)
        if torch.distributed.is_initialized():
            torch.distributed.barrier()
        self.last_online_eval_step = trigger_step

    def _run_online_eval_once(self, trigger_step: int):
        """Run a small online LIBERO eval by calling eval_vla.py on a temp checkpoint."""
        model_for_save = self.model.module if hasattr(self.model, "module") else self.model
        was_training = model_for_save.training
        model_for_save.eval()

        step_dir = os.path.join(self.online_eval_output_dir, f"step_{trigger_step}")
        os.makedirs(step_dir, exist_ok=True)
        ckpt_path = os.path.join(step_dir, f"online_eval_step_{trigger_step}.pt")

        # Save minimal checkpoint for evaluator.
        checkpoint = {
            "epoch": self.current_epoch,
            "global_step": self.global_step,
            "model_state_dict": model_for_save.state_dict(),
            "config": self.config,
            "best_val_loss": self.best_val_loss,
        }
        if hasattr(model_for_save, "action_head"):
            ah = model_for_save.action_head
            if hasattr(ah, "action_mean") and hasattr(ah, "action_std"):
                checkpoint["norm_stats"] = {
                    "action_mean": ah.action_mean.detach().cpu().numpy().tolist(),
                    "action_std": ah.action_std.detach().cpu().numpy().tolist(),
                }
        torch.save(checkpoint, ckpt_path)

        repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        eval_script = os.path.join(repo_root, "eval", "eval_vla.py")
        cmd = [
            sys.executable,
            eval_script,
            "--checkpoint", ckpt_path,
            "--benchmark", self.online_eval_benchmark,
            "--num_episodes", str(self.online_eval_num_episodes),
            "--max_steps", str(self.online_eval_max_steps),
            "--num_envs", str(self.online_eval_num_envs),
            "--action_chunk_size", str(self.online_eval_action_chunk_size),
            "--output_dir", step_dir,
            "--device", str(self.online_eval_device),
            "--task_ids",
        ] + [str(t) for t in self.online_eval_task_ids]

        if self.online_eval_max_init_states is not None:
            cmd += ["--max_init_states", str(self.online_eval_max_init_states)]
        if self.online_eval_save_videos:
            cmd += ["--save_videos"]
        if not self.online_eval_use_multi_view:
            cmd += ["--no_multi_view"]

        self._log(f"[OnlineEval] step={trigger_step} start: {' '.join(cmd)}")
        try:
            proc = subprocess.run(
                cmd,
                cwd=repo_root,
                check=False,
                capture_output=True,
                text=True,
            )
            if proc.returncode != 0:
                self._log(f"[OnlineEval] step={trigger_step} failed (exit={proc.returncode})")
                if proc.stderr:
                    self._log("[OnlineEval][stderr]\n" + proc.stderr[-2000:])
            else:
                result_files = sorted(glob.glob(os.path.join(step_dir, "eval_results_*.json")))
                if result_files:
                    with open(result_files[-1], "r", encoding="utf-8") as f:
                        result = json.load(f)
                    succ = float(result.get("overall_success_rate", 0.0))
                    self._log(f"[OnlineEval] step={trigger_step} success_rate={succ:.4f}")
                    if self.use_wandb:
                        wandb.log(
                            {"online_eval/success_rate": succ, "online_eval/step": trigger_step},
                            step=trigger_step,
                        )
                else:
                    self._log(f"[OnlineEval] step={trigger_step} no result json found in {step_dir}")
        except Exception as e:
            self._log(f"[OnlineEval] step={trigger_step} exception: {type(e).__name__}: {e}")
        finally:
            if was_training:
                model_for_save.train()
    
    @torch.no_grad()
    def validate(self):
        self.model.eval()
        total_loss = 0
        all_metrics = []
        
        for batch in tqdm(self.val_loader, desc="Validating", disable=not self.is_main):
            images = batch['image'].to(self.device)
            instructions = batch['instruction']
            actions_gt = batch['actions'].to(self.device)
            if self.action_mean is not None and self.action_std is not None:
                mean = self.action_mean.to(actions_gt.device)
                std = self.action_std.to(actions_gt.device)
                actions_gt = (actions_gt - mean) / std
            outputs = self.model(
                images, instructions,
                action_normalize=(self.action_mean is None or self.action_std is None)
            )
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
        interrupted = False
        interrupt_signal = None

        def _handle_termination(sig, _frame):
            nonlocal interrupt_signal
            interrupt_signal = sig
            raise KeyboardInterrupt

        prev_sigint = signal.getsignal(signal.SIGINT)
        prev_sigterm = signal.getsignal(signal.SIGTERM)
        signal.signal(signal.SIGINT, _handle_termination)
        signal.signal(signal.SIGTERM, _handle_termination)

        if self.is_main:
            msg = f"Starting training for {num_epochs} epochs (world_size={self.world_size})"
            if max_steps:
                msg += f", max_steps={max_steps}"
            if start_epoch > 0:
                msg += f", resuming from epoch {start_epoch}"
            self._log(msg)

        try:
            for epoch in range(start_epoch, num_epochs):
                if max_steps and self.global_step >= max_steps:
                    if self.is_main:
                        self._log(f"Reached max_steps={max_steps}, stopping.")
                    break
                self.current_epoch = epoch

                train_loss = self.train_epoch(max_steps=max_steps)
                val_loss, val_metrics = self.validate()

                if self.scheduler is not None and not self.scheduler_step_per_batch:
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
                        wandb.log(log_dict, step=self.global_step)

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
        except KeyboardInterrupt:
            interrupted = True
            signal_name = signal.Signals(interrupt_signal).name if interrupt_signal else "SIGINT"
            if self.is_main:
                self._log(f"\nReceived {signal_name}, saving interrupt checkpoint ...")
                ckpt_name = self.save_interrupt_checkpoint()
                self._log(f"  ✓ Saved interrupt checkpoint: {ckpt_name}")
        finally:
            signal.signal(signal.SIGINT, prev_sigint)
            signal.signal(signal.SIGTERM, prev_sigterm)
            if self.is_main:
                self._log("Training interrupted." if interrupted else "Training complete!")
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
        # 显式保存 norm_stats，eval / 继续训练时可直接读取，与 OpenVLA config.json 的 norm_stats 一致
        if hasattr(model_for_save, 'action_head'):
            ah = model_for_save.action_head
            if hasattr(ah, 'action_mean') and hasattr(ah, 'action_std'):
                checkpoint['norm_stats'] = {
                    'action_mean': ah.action_mean.cpu().numpy().tolist(),
                    'action_std': ah.action_std.cpu().numpy().tolist(),
                }
        if self.scheduler is not None:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        if self.use_wandb and _HAS_WANDB and hasattr(wandb, 'run') and wandb.run is not None:
            checkpoint['wandb_run_id'] = wandb.run.id

        path = os.path.join(self.log_dir, filename)
        torch.save(checkpoint, path)

    def save_interrupt_checkpoint(self) -> str:
        """保存中断时 checkpoint，便于 Ctrl+C/SIGTERM 后恢复训练。"""
        dataset_clean = ''.join(c if c.isalnum() or c in ('_', '-') else '_' for c in self.dataset_name)
        dataset_clean = dataset_clean.replace('/', '_').replace('\\', '_')
        interrupt_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = (
            f"interrupt_checkpoint_{dataset_clean}_{self.training_start_time}"
            f"_epoch{self.current_epoch}_step{self.global_step}_{interrupt_time}.pt"
        )
        self.save_checkpoint(filename)
        return filename
    
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
        # 恢复 norm_stats 到 Trainer，继续训练时用 checkpoint 的 mean/std 做 target 归一化
        if 'norm_stats' in checkpoint:
            ns = checkpoint['norm_stats']
            if 'action_mean' in ns and 'action_std' in ns:
                self.action_mean = torch.tensor(ns['action_mean'], dtype=torch.float32, device=self.device)
                self.action_std = torch.tensor(ns['action_std'], dtype=torch.float32, device=self.device)

        return checkpoint.get('wandb_run_id')

