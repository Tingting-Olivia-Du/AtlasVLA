#!/usr/bin/env python3
"""
VLA-VGGT 训练脚本
支持:
- HuggingFace LIBERO 数据集
- 可选的 Vision Tower (DINO/CLIP/SigLIP)
- facebook/vggt 或简化版 VGGT
- Qwen3-0.6B 语言编码器
- 多 GPU 分布式训练 (DDP)
"""
import sys
import os
import warnings

# 屏蔽 wandb 内部触发的 pkg_resources 弃用警告（非本仓库问题）
warnings.filterwarnings("ignore", message="pkg_resources is deprecated", category=UserWarning)

# 保证 vggt_vla 在 sys.path 最前，避免多进程 spawn 时继承父进程 path 后误导入 vggt/training/trainer（iopath 等）
_vggt_vla_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _vggt_vla_root)

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
import argparse
import yaml
import math
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
    parser.add_argument('--task_indices', type=int, nargs='+', default=None,
                       help='Task indices for LeRobot format (e.g. 0 1 2 for first 3 tasks)')
    parser.add_argument('--max_episodes', type=int, default=None,
                       help='Max episodes to use (for quick debug)')
    parser.add_argument('--max_samples', type=int, default=None,
                       help='Max training samples (for quick debug)')
    parser.add_argument('--cache_dir', type=str, default=None,
                       help='HuggingFace cache directory')
    
    # Training
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--max_steps', type=int, default=None,
                       help='Max training steps (stop when reached; overrides num_epochs if set)')
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-5)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--grad_clip', type=float, default=1.0)
    parser.add_argument('--warmup_steps', type=int, default=0,
                       help='Linear warmup steps for step-level scheduler (0 disables warmup)')
    parser.add_argument('--min_lr_ratio', type=float, default=0.01,
                       help='Minimum LR ratio for cosine decay after warmup')
    
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
                       help='Use facebook/VGGT-1B from HuggingFace')
    parser.add_argument('--freeze_vggt', action='store_true',
                       help='Freeze VGGT backbone')
    parser.add_argument('--use_vggt_lora', action='store_true',
                       help='Enable LoRA finetuning on VGGT backbone')
    parser.add_argument('--vggt_lora_rank', type=int, default=8,
                       help='LoRA rank for VGGT')
    parser.add_argument('--vggt_lora_alpha', type=int, default=16,
                       help='LoRA alpha for VGGT')
    parser.add_argument('--vggt_lora_dropout', type=float, default=0.05,
                       help='LoRA dropout for VGGT')
    parser.add_argument('--vggt_lora_target_modules', nargs='+',
                       default=['qkv', 'proj', 'fc1', 'fc2'],
                       help='VGGT module names to apply LoRA (e.g. qkv proj fc1 fc2)')
    parser.add_argument('--allow_vggt_fallback', action='store_true',
                       help='Allow silent fallback path when VGGT forward fails (not recommended for production training)')
    
    parser.add_argument('--use_multi_view', default='true',
                       help='Use dual view (agentview + wrist). Set in yaml: use_multi_view: true. Default: true')
    parser.add_argument('--no_multi_view', action='store_true',
                       help='禁用双视角(与 use_multi_view 冲突时以 no_multi_view 为准)')
    parser.add_argument('--action_horizon', type=int, default=10,
                       help='Action prediction horizon')
    parser.add_argument('--action_dim', type=int, default=7,
                       help='Action dimension')
    parser.add_argument('--loss_mse_weight', type=float, default=0.5,
                       help='Weight of MSE term in action loss')
    parser.add_argument('--loss_huber_weight', type=float, default=0.5,
                       help='Weight of Huber term in action loss')
    parser.add_argument('--loss_huber_delta', type=float, default=1.0,
                       help='Delta for Huber loss')
    parser.add_argument('--loss_smooth_weight', type=float, default=0.02,
                       help='Weight of temporal smoothness term')
    parser.add_argument('--loss_gripper_weight', type=float, default=2.0,
                       help='Extra weight for gripper dimension in action loss')
    parser.add_argument('--loss_smooth_exclude_gripper', type=lambda x: str(x).lower() in ('1', 'true', 'yes'),
                       default=True,
                       help='Exclude gripper dim from smoothness term (true/false)')
    
    # Logging
    parser.add_argument('--log_dir', type=str, default='./logs',
                       help='Logging directory')
    parser.add_argument('--exp_name', type=str, default=None,
                       help='Experiment name')
    parser.add_argument('--save_freq', type=int, default=10,
                       help='Save checkpoint every N epochs (set large to disable)')
    parser.add_argument('--save_every_steps', type=int, default=None,
                       help='Save checkpoint every N steps (e.g. 10000 for 10k, 20k, 30k...)')
    parser.add_argument('--best_model_save_every_steps', type=int, default=None,
                       help='Only save best model every N steps (default: every epoch when val improves)')
    parser.add_argument('--use_wandb', action='store_true',
                       help='Use Weights & Biases for logging')
    parser.add_argument('--wandb_entity', type=str, default="tingtingdu06-uw-madison",
                       help='W&B entity (your username or team), e.g. tingtingdu06-uw-madison')
    parser.add_argument('--wandb_project', type=str, default='vla-vggt',
                       help='W&B project name')
    parser.add_argument('--wandb_run_name', type=str, default=None,
                       help='W&B run name (default: exp_name)')

    # Periodic online eval hook (small LIBERO sim eval during training)
    parser.add_argument('--online_eval_every_steps', type=int, default=5000,
                       help='Run small online eval every N steps (0/None to disable)')
    parser.add_argument('--online_eval_benchmark', type=str, default='libero_spatial',
                       help='LIBERO benchmark for online eval hook')
    parser.add_argument('--online_eval_task_ids', type=int, nargs='+', default=[0, 1, 2],
                       help='Task IDs used by online eval hook')
    parser.add_argument('--online_eval_num_episodes', type=int, default=2,
                       help='Episodes per task in online eval hook')
    parser.add_argument('--online_eval_max_steps', type=int, default=220,
                       help='Max rollout steps in online eval hook')
    parser.add_argument('--online_eval_num_envs', type=int, default=1,
                       help='Parallel env count in online eval hook')
    parser.add_argument('--online_eval_action_chunk_size', type=int, default=8,
                       help='Action chunk size in online eval hook')
    parser.add_argument('--online_eval_max_init_states', type=int, default=2,
                       help='Limit init states per task in online eval hook')
    parser.add_argument('--online_eval_save_videos', action='store_true',
                       help='Save videos during online eval hook')
    
    # System
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use (cuda/cpu)')
    parser.add_argument('--gpus', type=str, default=None,
                       help='GPU IDs for multi-GPU training, e.g. "0,1,2,3". If not set, use all visible CUDA devices.')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    
    # Config file (used as defaults; explicit CLI args should take priority)
    parser.add_argument('--config', type=str, default=None,
                       help='Path to YAML config file')
    
    # Resume from checkpoint
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to checkpoint (.pth/.pt) to resume training from')
    parser.add_argument('--wandb_resume', action='store_true',
                       help='Resume same wandb run when using --resume')
    parser.add_argument('--wandb_run_id', type=str, default=None,
                       help='Wandb run id for resume (from wandb dir run-*_ID or run-ID.wandb); used when checkpoint has no wandb_run_id')
    
    args = parser.parse_args()
    
    # Load config file if provided.
    # Priority rule:
    #   explicit CLI arg > YAML config > parser default
    if args.config is not None:
        with open(args.config, 'r') as f:
            config_dict = yaml.safe_load(f)
        
        # Update args with config values only when user did not explicitly
        # override that argument on the command line.
        for key, value in config_dict.items():
            if hasattr(args, key):
                current_value = getattr(args, key)
                default_value = parser.get_default(key)
                if current_value == default_value:
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
        use_vggt_lora=getattr(args, 'use_vggt_lora', False),
        vggt_lora_rank=getattr(args, 'vggt_lora_rank', 8),
        vggt_lora_alpha=getattr(args, 'vggt_lora_alpha', 16),
        vggt_lora_dropout=getattr(args, 'vggt_lora_dropout', 0.05),
        vggt_lora_target_modules=getattr(args, 'vggt_lora_target_modules', ['qkv', 'proj', 'fc1', 'fc2']),
        allow_vggt_fallback=getattr(args, 'allow_vggt_fallback', False),
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
        use_spatial_features=False,
        loss_mse_weight=getattr(args, 'loss_mse_weight', 0.5),
        loss_huber_weight=getattr(args, 'loss_huber_weight', 0.5),
        loss_huber_delta=getattr(args, 'loss_huber_delta', 1.0),
        loss_smooth_weight=getattr(args, 'loss_smooth_weight', 0.02),
        loss_gripper_weight=getattr(args, 'loss_gripper_weight', 2.0),
        loss_smooth_exclude_gripper=getattr(args, 'loss_smooth_exclude_gripper', True),
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


def setup_distributed(rank: int, world_size: int):
    """初始化分布式环境"""
    os.environ['MASTER_ADDR'] = os.environ.get('MASTER_ADDR', 'localhost')
    os.environ['MASTER_PORT'] = os.environ.get('MASTER_PORT', '29500')
    dist.init_process_group("nccl", rank=rank, world_size=world_size)


def cleanup_distributed():
    """清理分布式环境"""
    if dist.is_initialized():
        dist.destroy_process_group()


def main_worker(rank: int, world_size: int, args):
    """每个 GPU 进程的主函数"""
    distributed = world_size > 1
    if distributed:
        setup_distributed(rank, world_size)

    # 当前进程使用的 device
    device = torch.device(f'cuda:{rank}' if torch.cuda.is_available() else 'cpu')

    # Set seed
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # Create log directory (only rank 0)
    if args.exp_name is None:
        exp_name = f"vla_vggt"
        if args.use_vision_tower:
            exp_name += f"_{args.vision_tower_name.split('/')[-1]}"
        if args.use_pretrained_vggt:
            exp_name += "_pretrained"
        args.exp_name = exp_name

    log_dir = Path(args.log_dir) / args.exp_name
    if rank == 0:
        log_dir.mkdir(parents=True, exist_ok=True)

    if rank == 0:
        print("\n" + "=" * 80)
        print(f"VLA-VGGT Training")
        print("=" * 80)
        print(f"Experiment: {args.exp_name}")
        print(f"Log dir: {log_dir}")
        print(f"Dataset: {args.dataset_repo}")
        print(f"GPUs: {args.gpus or 'auto'}, world_size={world_size}")
        if args.resume:
            print(f"Resume: {args.resume}")
        print("=" * 80 + "\n")

    # Resume: load checkpoint to get config and resume state
    resume_state = None
    if args.resume and str(args.resume).strip():
        ckpt_path = Path(args.resume)
        if not ckpt_path.is_absolute():
            ckpt_path = Path.cwd() / ckpt_path
        if ckpt_path.exists():
            ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
            wandb_rid = ckpt.get('wandb_run_id') if args.wandb_resume else None
            if args.wandb_resume and wandb_rid is None and getattr(args, 'wandb_run_id', None):
                wandb_rid = args.wandb_run_id
            resume_state = {
                'path': str(ckpt_path.resolve()),
                'epoch': ckpt.get('epoch', 0),
                'global_step': ckpt.get('global_step', 0),
                'best_val_loss': ckpt.get('best_val_loss', float('inf')),
                'wandb_run_id': wandb_rid,
                'config': ckpt.get('config'),
            }
            if rank == 0:
                print(f"Resuming from: {resume_state['path']} (epoch {resume_state['epoch']})")
        else:
            raise FileNotFoundError(f"Resume checkpoint not found: {ckpt_path}")

    # Build config: 模型结构用 checkpoint，训练参数（batch_size 等）始终用当前 args
    if resume_state and resume_state.get('config') is not None:
        config = resume_state['config']
        if rank == 0:
            print("Using model config from checkpoint (batch_size/lr 等仍用当前 config)")
    else:
        config = build_config(args)

    # Save config (only rank 0)
    if rank == 0:
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
    if rank == 0:
        print("Creating model...")
    model = VLAModel(config)
    model = model.to(device)

    if distributed:
        # find_unused_parameters=True: 有冻结模块（language/vision/VGGT）时部分参数不参与 loss，需开启
        model = DDP(model, device_ids=[rank], find_unused_parameters=True)

    if rank == 0:
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"\nModel Parameters:")
        print(f"  Total: {total_params / 1e6:.2f}M")
        print(f"  Trainable: {trainable_params / 1e6:.2f}M")
        print(f"  Frozen: {(total_params - trainable_params) / 1e6:.2f}M\n")

    # Load data
    if rank == 0:
        print("Loading data...")
    # 双视角: 优先读 config 的 use_multi_view，否则默认 True；--no_multi_view 可覆盖
    use_multi_view = getattr(args, 'use_multi_view', None)
    if use_multi_view is None:
        use_multi_view = not getattr(args, 'no_multi_view', False)
    else:
        use_multi_view = bool(use_multi_view)
    if getattr(args, 'no_multi_view', False):
        use_multi_view = False
    if rank == 0:
        print(f"  Multi-view (agentview + wrist): {use_multi_view}")
    # split_seed=42 固定 train/val 划分，便于后续在「训练集上」做离线 eval 时复现同一批样本
    train_loader, val_loader, action_stats = get_libero_hf_dataloaders(
        repo_id=args.dataset_repo,
        task_names=args.task_names,
        task_indices=args.task_indices,
        max_episodes=args.max_episodes,
        max_samples=args.max_samples,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        action_horizon=args.action_horizon,
        action_dim=config.action_head.action_dim,
        train_split_ratio=getattr(args, 'train_split_ratio', 0.9),
        split_seed=getattr(args, 'split_seed', 42),
        cache_dir=args.cache_dir,
        use_multi_view=use_multi_view,
        distributed=distributed,
        rank=rank,
        world_size=world_size,
        compute_action_stats=True,
        action_stats_max_samples=20000,
    )

    action_mean, action_std = None, None
    if distributed:
        have_stats_t = torch.tensor(
            [1 if (action_stats is not None) else 0], device=device, dtype=torch.long
        )
        dist.broadcast(have_stats_t, 0)
        have_stats = have_stats_t.item() == 1
    else:
        have_stats = action_stats is not None
    if have_stats:
        if action_stats is not None:
            mean_t = torch.from_numpy(action_stats["mean"]).float().to(device)
            std_t = torch.from_numpy(action_stats["std"]).float().to(device)
            if distributed:
                dist.broadcast(mean_t, 0)
                dist.broadcast(std_t, 0)
        else:
            mean_t = torch.zeros(config.action_head.action_dim, device=device)
            std_t = torch.ones(config.action_head.action_dim, device=device)
            dist.broadcast(mean_t, 0)
            dist.broadcast(std_t, 0)
        act_head = (model.module if distributed else model).action_head
        act_head.set_action_stats(mean_t, std_t)
        action_mean, action_std = mean_t, std_t
        if rank == 0 and action_stats is not None:
            print("  ✓ Action normalization: set_action_stats from dataset")

    # Create optimizer
    param_groups = model.module.get_param_groups(learning_rate=args.lr, weight_decay=args.weight_decay) if distributed else model.get_param_groups(learning_rate=args.lr, weight_decay=args.weight_decay)
    optimizer = torch.optim.AdamW(param_groups)

    # Create scheduler
    # - warmup_steps > 0: step-level linear warmup + cosine decay
    # - warmup_steps == 0: keep legacy epoch-level cosine schedule
    scheduler_step_per_batch = False
    if getattr(args, 'warmup_steps', 0) and int(args.warmup_steps) > 0:
        scheduler_step_per_batch = True
        total_train_steps = int(args.max_steps) if args.max_steps else int(args.num_epochs) * len(train_loader)
        warmup_steps = int(args.warmup_steps)
        min_lr_ratio = float(getattr(args, 'min_lr_ratio', 0.01))
        if total_train_steps <= 0:
            total_train_steps = 1
        if warmup_steps >= total_train_steps:
            warmup_steps = max(1, total_train_steps - 1)

        def lr_lambda(current_step: int):
            if current_step < warmup_steps:
                return float(current_step + 1) / float(max(1, warmup_steps))
            progress = float(current_step - warmup_steps) / float(max(1, total_train_steps - warmup_steps))
            progress = min(max(progress, 0.0), 1.0)
            cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
            return min_lr_ratio + (1.0 - min_lr_ratio) * cosine

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
        if rank == 0:
            print(f"  Scheduler: Warmup+Cosine (step-level), warmup_steps={warmup_steps}, total_steps={total_train_steps}, min_lr_ratio={min_lr_ratio}")
    else:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=args.num_epochs,
            eta_min=args.lr * 0.01
        )
        if rank == 0:
            print(f"  Scheduler: CosineAnnealingLR (epoch-level), T_max={args.num_epochs}, eta_min={args.lr * 0.01:g}")

    # 构建数据集名称用于 checkpoint 命名
    dataset_name = args.dataset_repo.split('/')[-1]  # 例如 "lerobot/libero_spatial_image" -> "libero_spatial_image"
    if args.task_indices is not None:
        dataset_name += f"_tasks{'-'.join(map(str, args.task_indices))}"
    elif args.task_names is not None:
        task_str = '-'.join(args.task_names[:3])  # 最多显示前3个任务名
        if len(args.task_names) > 3:
            task_str += f"_and{len(args.task_names)-3}more"
        dataset_name += f"_{task_str}"
    
    # Create trainer
    resume_epoch = (resume_state['epoch'] + 1) if resume_state else 0
    resume_global_step = resume_state['global_step'] if resume_state else 0
    resume_best_val_loss = resume_state['best_val_loss'] if resume_state else None
    wandb_run_id = resume_state['wandb_run_id'] if resume_state and args.wandb_resume else None

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        config=config,
        device=str(device),
        log_dir=str(log_dir),
        grad_clip=args.grad_clip,
        save_freq=args.save_freq,
        save_every_steps=args.save_every_steps,
        best_model_save_every_steps=args.best_model_save_every_steps,
        rank=rank,
        world_size=world_size,
        use_wandb=args.use_wandb,
        wandb_entity=getattr(args, 'wandb_entity', None),
        wandb_project=args.wandb_project,
        wandb_run_name=args.wandb_run_name or args.exp_name,
        dataset_name=dataset_name,
        resume_epoch=resume_epoch,
        resume_global_step=resume_global_step,
        resume_best_val_loss=resume_best_val_loss,
        wandb_run_id=wandb_run_id,
        action_mean=action_mean,
        action_std=action_std,
        online_eval_every_steps=getattr(args, 'online_eval_every_steps', None),
        online_eval_benchmark=getattr(args, 'online_eval_benchmark', 'libero_spatial'),
        online_eval_task_ids=getattr(args, 'online_eval_task_ids', [0, 1, 2]),
        online_eval_num_episodes=getattr(args, 'online_eval_num_episodes', 2),
        online_eval_max_steps=getattr(args, 'online_eval_max_steps', 220),
        online_eval_num_envs=getattr(args, 'online_eval_num_envs', 1),
        online_eval_action_chunk_size=getattr(args, 'online_eval_action_chunk_size', 8),
        online_eval_max_init_states=getattr(args, 'online_eval_max_init_states', 2),
        online_eval_save_videos=getattr(args, 'online_eval_save_videos', False),
        online_eval_use_multi_view=use_multi_view,
        online_eval_device=str(device),
        scheduler_step_per_batch=scheduler_step_per_batch,
    )

    # Load full checkpoint (model, optimizer, scheduler) when resuming
    if resume_state:
        trainer.load_checkpoint("", path_override=resume_state['path'])

    # Train
    if rank == 0:
        print("\n" + "=" * 80)
        print("Starting Training")
        print("=" * 80 + "\n")
    trainer.train(args.num_epochs, max_steps=args.max_steps)

    if rank == 0:
        print("\n" + "=" * 80)
        print("Training Complete!")
        print("=" * 80)
        print(f"Best model saved to: {log_dir / 'best_model.pt'}")
        print(f"Logs saved to: {log_dir}")
        print("=" * 80 + "\n")

    if distributed:
        cleanup_distributed()


def main():
    args = parse_args()
    print(f"[Args] resolved gpus={args.gpus!r}, config={args.config!r}")

    # 解析 GPU 列表: --gpus "0,1,2,3" 指定多卡; null/未设置则使用全部可见 GPU
    if args.gpus is not None and str(args.gpus).strip():
        gpu_ids = [int(x.strip()) for x in str(args.gpus).split(',') if x.strip()]
    else:
        gpu_ids = list(range(torch.cuda.device_count())) if torch.cuda.is_available() else []

    world_size = len(gpu_ids) if gpu_ids else 1

    if world_size > 1:
        # 设置可见 GPU，spawn 的每个进程会看到不同的 CUDA_VISIBLE_DEVICES
        os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(map(str, gpu_ids))
        # spawn 时每个进程的 rank 对应 cuda:0, cuda:1, ... (因为 CUDA_VISIBLE_DEVICES 已限制)
        mp.spawn(main_worker, nprocs=world_size, args=(world_size, args))
    else:
        # 单卡
        if gpu_ids:
            os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_ids[0])
        main_worker(0, 1, args)


if __name__ == '__main__':
    main()
