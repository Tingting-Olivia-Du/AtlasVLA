"""
Main training script for Atlas VLA
Supports single GPU and multi-GPU distributed training
"""

import argparse
import yaml
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import sys
import os
from datetime import datetime, timedelta
import logging

# Add project root to path for development (if not installed as package)
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Add vggt to path (critical for VGGT import)
vggt_path = os.path.join(project_root, 'vggt')
if vggt_path not in sys.path:
    sys.path.insert(0, vggt_path)

try:
    from atlas.src.models import VGGTVLA
    from atlas.src.data import LIBERODataset, LIBEROHFDataset
    from atlas.src.training import VLATrainer
except ImportError:
    # Fallback for direct script execution
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../vggt'))
    from atlas.src.models import VGGTVLA
    from atlas.src.data import LIBERODataset, LIBEROHFDataset
    from atlas.src.training import VLATrainer


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def setup_distributed():
    """Initialize distributed training"""
    # Check if running in distributed mode
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ.get('LOCAL_RANK', 0))
        
        # Set device BEFORE initializing process group (important!)
        torch.cuda.set_device(local_rank)
        device = torch.device(f'cuda:{local_rank}')
        
        # Initialize process group with timeout to avoid hanging
        try:
            # Use default timeout (30 minutes) or get from env
            timeout_str = os.environ.get('TORCH_DISTRIBUTED_TIMEOUT', '1800')  # 30 minutes default
            timeout = int(timeout_str) if timeout_str.isdigit() else 1800
            
            # Print debug info before init (all ranks)
            print(f"[Rank {rank}] Initializing process group: backend=nccl, world_size={world_size}, local_rank={local_rank}")
            print(f"[Rank {rank}] CUDA device: {device}, CUDA available: {torch.cuda.is_available()}")
            
            dist.init_process_group(
                backend='nccl',
                init_method='env://',  # Use environment variables
                timeout=timedelta(seconds=timeout)
            )
            
            print(f"[Rank {rank}] Process group initialized successfully")
        except Exception as e:
            print(f"[Rank {rank}] Error initializing process group: {e}")
            print(f"[Rank {rank}] Rank: {rank}, World size: {world_size}, Local rank: {local_rank}")
            import traceback
            print(traceback.format_exc())
            raise
        
        return True, rank, world_size, local_rank, device
    else:
        return False, 0, 1, 0, torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def cleanup_distributed():
    """Cleanup distributed training"""
    if dist.is_initialized():
        dist.destroy_process_group()


def setup_logging(config: dict, rank: int = 0):
    """
    Setup logging configuration
    Only rank 0 process logs to file in distributed training
    """
    logging_config = config.get("logging", {})
    
    # Only setup file logging on rank 0
    if rank != 0:
        return None
    
    save_to_file = logging_config.get("save_to_file", False)
    if not save_to_file:
        return None
    
    # Determine log file path
    log_file = logging_config.get("log_file")
    if log_file is None:
        # Auto-generate log file name
        log_dir = logging_config.get("log_dir", "./logs")
        os.makedirs(log_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = os.path.join(log_dir, f"train_{timestamp}.log")
    
    # Create log directory if needed
    log_dir = os.path.dirname(log_file)
    if log_dir and not os.path.exists(log_dir):
        os.makedirs(log_dir, exist_ok=True)
    
    # Setup logging
    console_output = logging_config.get("console_output", True)
    
    # Configure root logger
    handlers = []
    
    # File handler
    file_handler = logging.FileHandler(log_file, mode='a', encoding='utf-8')
    file_handler.setLevel(logging.INFO)
    file_formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(file_formatter)
    handlers.append(file_handler)
    
    # Console handler (if enabled)
    if console_output:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        console_handler.setFormatter(console_formatter)
        handlers.append(console_handler)
    
    # Configure root logger
    logging.basicConfig(
        level=logging.INFO,
        handlers=handlers,
        force=True  # Override existing handlers
    )
    
    logging.info(f"Logging to file: {log_file}")
    return log_file


def main():
    parser = argparse.ArgumentParser(description="Train Atlas VLA model")
    parser.add_argument(
        "--config",
        type=str,
        default="atlas/configs/train_config.yaml",
        help="Path to config file"
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint to resume from"
    )
    args = parser.parse_args()
    
    # Setup distributed training
    is_distributed, rank, world_size, local_rank, device = setup_distributed()
    
    # Load configuration
    config = load_config(args.config)
    
    # Setup logging (only on rank 0)
    log_file = setup_logging(config, rank=rank)
    
    # Override device from config if not distributed
    if not is_distributed:
        device_str = config.get("device", "cuda" if torch.cuda.is_available() else "cpu")
        device = torch.device(device_str)
    
    # Only print on rank 0
    if rank == 0:
        logging.info(f"Training setup:")
        logging.info(f"  Distributed: {is_distributed}")
        if is_distributed:
            logging.info(f"  World size: {world_size}")
            logging.info(f"  Rank: {rank}, Local rank: {local_rank}")
        logging.info(f"  Device: {device}")
        if log_file:
            logging.info(f"  Log file: {log_file}")
    
    # Initialize model
    if rank == 0:
        logging.info("Initializing model...")
    
    # Get HuggingFace token from config
    hf_token = config.get("huggingface", {}).get("token")
    if not hf_token:
        # Try to get from environment as fallback
        hf_token = os.environ.get('HF_TOKEN') or os.environ.get('HUGGINGFACE_TOKEN')
    
    if hf_token:
        # Set as environment variable for transformers library (multiple names for compatibility)
        os.environ['HF_TOKEN'] = hf_token
        os.environ['HUGGINGFACE_TOKEN'] = hf_token
        os.environ['HF_AUTH_TOKEN'] = hf_token  # Some libraries use this
        if rank == 0:
            logging.info("HuggingFace token loaded and set in environment")
            logging.info(f"Token (first 15 chars): {hf_token[:15]}...")
            logging.info(f"Token length: {len(hf_token)}")
    else:
        if rank == 0:
            logging.warning("No HuggingFace token found in config or environment!")
            logging.warning("Model loading will likely fail. Please set token in config or use huggingface-cli login")
    
    try:
        model_config = config["model"]
        model = VGGTVLA(
            vggt_model=None,  # Will load from checkpoint
            lang_encoder_name=model_config["lang_encoder_name"],
            freeze_vggt=model_config["freeze_vggt"],
            freeze_lang_encoder=model_config["freeze_lang_encoder"],
            geom_output_dim=model_config["geom_output_dim"],
            fusion_hidden_dim=model_config["fusion_hidden_dim"],
            action_dim=model_config["action_dim"],
            use_pointnet=model_config.get("use_pointnet", True),
            use_pose=model_config.get("use_pose", True),
            hf_token=hf_token,  # Pass token to model (will also be set in env)
        )
        model = model.to(device)
        
        if rank == 0:
            logging.info("Model initialized successfully")
    except Exception as e:
        if rank == 0:
            logging.error(f"Error initializing model: {e}")
            import traceback
            logging.error(traceback.format_exc())
        raise
    
    # Wrap model with DDP if distributed
    if is_distributed:
        try:
            model = DDP(
                model, 
                device_ids=[local_rank], 
                output_device=local_rank, 
                find_unused_parameters=True,
                broadcast_buffers=True,
                gradient_as_bucket_view=True  # More memory efficient
            )
            if rank == 0:
                logging.info(f"Model wrapped with DDP (world_size={world_size})")
        except Exception as e:
            if rank == 0:
                logging.error(f"Error wrapping model with DDP: {e}")
                import traceback
                logging.error(traceback.format_exc())
            raise
    
    if rank == 0:
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logging.info(f"Model initialized. Total parameters: {total_params:,}")
        logging.info(f"Trainable parameters: {trainable_params:,}")
    
    # Load datasets
    if rank == 0:
        logging.info("Loading datasets...")
    data_config = config["data"]
    
    # 支持两种数据源：HuggingFace或本地HDF5转换后的数据
    use_huggingface = data_config.get("use_huggingface", False)
    
    if use_huggingface:
        # 使用HuggingFace数据（推荐，无需转换）
        if rank == 0:
            hf_dataset_name = data_config.get("hf_dataset_name", "physical-intelligence/libero")
            streaming = data_config.get("streaming", False)
            logging.info(f"  Using HuggingFace dataset: {hf_dataset_name}")
            logging.info(f"  Streaming mode: {streaming}")
        
        # For distributed training: rank 0 loads first to cache, then others load from cache
        if is_distributed:
            if rank == 0:
                logging.info("  Rank 0: Loading dataset (will be cached for other ranks)...")
                train_dataset = LIBEROHFDataset(
                    dataset_name=data_config.get("hf_dataset_name", "physical-intelligence/libero"),
                    split=data_config.get("train_split", "train"),
                    image_size=data_config["image_size"],
                    use_wrist_camera=data_config["use_wrist_camera"],
                    streaming=data_config.get("streaming", False),
                    cache_dir=data_config.get("hf_cache_dir", None),
                    token=hf_token,
                )
                logging.info("  Rank 0: Dataset loaded, waiting for other ranks...")
            
            # Synchronize: wait for rank 0 to finish loading/caching
            dist.barrier()
            
            # Other ranks load from cache
            if rank != 0:
                train_dataset = LIBEROHFDataset(
                    dataset_name=data_config.get("hf_dataset_name", "physical-intelligence/libero"),
                    split=data_config.get("train_split", "train"),
                    image_size=data_config["image_size"],
                    use_wrist_camera=data_config["use_wrist_camera"],
                    streaming=data_config.get("streaming", False),
                    cache_dir=data_config.get("hf_cache_dir", None),
                    token=hf_token,
                )
        else:
            # Single GPU: load normally
            train_dataset = LIBEROHFDataset(
                dataset_name=data_config.get("hf_dataset_name", "physical-intelligence/libero"),
                split=data_config.get("train_split", "train"),
                image_size=data_config["image_size"],
                use_wrist_camera=data_config["use_wrist_camera"],
                streaming=data_config.get("streaming", False),
                cache_dir=data_config.get("hf_cache_dir", None),
                token=hf_token,
            )
        
        val_dataset = None
        # HuggingFace数据通常只有train split，如果需要验证集可以后续添加
        if data_config.get("val_split"):
            # 可以手动分割或使用subset
            if rank == 0:
                logging.info("  Note: HuggingFace dataset typically only has 'train' split")
                logging.info("  Validation split will be skipped")
    else:
        # 使用本地HDF5转换后的数据
        if rank == 0:
            logging.info(f"  Using local dataset from: {data_config['data_dir']}")
        
        train_dataset = LIBERODataset(
            data_dir=data_config["data_dir"],
            split=data_config["train_split"],
            image_size=data_config["image_size"],
            use_wrist_camera=data_config["use_wrist_camera"],
        )
        
        val_dataset = None
        if data_config.get("val_split"):
            val_dataset = LIBERODataset(
                data_dir=data_config["data_dir"],
                split=data_config["val_split"],
                image_size=data_config["image_size"],
                use_wrist_camera=data_config["use_wrist_camera"],
            )
    
    # Create distributed samplers if needed
    train_sampler = None
    val_sampler = None
    if is_distributed:
        train_sampler = DistributedSampler(
            train_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True
        )
        if val_dataset is not None:
            val_sampler = DistributedSampler(
                val_dataset,
                num_replicas=world_size,
                rank=rank,
                shuffle=False
            )
    
    # Initialize trainer
    if rank == 0:
        logging.info("Initializing trainer...")
    training_config = config["training"]
    
    # Adjust learning rate for multi-GPU (linear scaling rule)
    base_lr = float(training_config["learning_rate"])  # Ensure it's a float, not string
    if is_distributed:
        # Linear scaling: lr = base_lr * world_size
        # Or use sqrt scaling: lr = base_lr * sqrt(world_size) (more conservative)
        effective_lr = base_lr * world_size
    else:
        effective_lr = base_lr
    
    if rank == 0:
        if is_distributed:
            logging.info(f"  Base LR: {base_lr}, Effective LR (scaled by {world_size}): {effective_lr}")
        else:
            logging.info(f"  Learning rate: {effective_lr}")
    
    # Prepare wandb config
    wandb_config = config.get("wandb", {})
    
    trainer = VLATrainer(
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        batch_size=data_config["batch_size"],
        learning_rate=effective_lr,
        weight_decay=training_config["weight_decay"],
        num_epochs=training_config["num_epochs"],
        device=device,
        save_dir=config["checkpoint"]["save_dir"],
        log_interval=training_config["log_interval"],
        val_interval=training_config["val_interval"],
        save_interval=training_config["save_interval"],
        use_wandb=wandb_config.get("enabled", False) and rank == 0,  # Only rank 0 logs to wandb
        wandb_project=wandb_config.get("project", "atlas-vla"),
        wandb_entity=wandb_config.get("entity"),
        wandb_name=wandb_config.get("name"),
        wandb_tags=wandb_config.get("tags", []),
        wandb_notes=wandb_config.get("notes", ""),
        warmup_steps=training_config.get("warmup_steps", 0),
        gradient_accumulation_steps=training_config.get("gradient_accumulation_steps", 1),
        train_sampler=train_sampler,
        val_sampler=val_sampler,
        loss=training_config.get("loss", {}),
    )
    
    # Resume from checkpoint if specified
    if args.resume or config["checkpoint"].get("resume_from"):
        checkpoint_path = args.resume or config["checkpoint"]["resume_from"]
        trainer.load_checkpoint(checkpoint_path)
        if rank == 0:
            logging.info(f"Resumed training from {checkpoint_path}")
    
    # Start training
    try:
        if rank == 0:
            logging.info("Starting training...")
        trainer.train()
        if rank == 0:
            logging.info("Training completed successfully")
    except KeyboardInterrupt:
        if rank == 0:
            logging.warning("Training interrupted by user")
        raise
    except Exception as e:
        if rank == 0:
            logging.error(f"Training failed with error: {e}")
            import traceback
            logging.error(traceback.format_exc())
        raise
    finally:
        # Cleanup distributed training
        if rank == 0:
            logging.info("Cleaning up distributed training...")
        cleanup_distributed()


if __name__ == "__main__":
    main()
