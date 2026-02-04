"""
Main training script for Atlas VLA
"""

import argparse
import yaml
import torch
import sys
import os

# Add project root to path for development (if not installed as package)
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

try:
    from atlas.src.models import VGGTVLA
    from atlas.src.data import LIBERODataset
    from atlas.src.training import VLATrainer
except ImportError:
    # Fallback for direct script execution
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../vggt'))
    from atlas.src.models import VGGTVLA
    from atlas.src.data import LIBERODataset
    from atlas.src.training import VLATrainer


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


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
    
    # Load configuration
    config = load_config(args.config)
    
    # Set device
    device = config.get("device", "cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Initialize model
    print("Initializing model...")
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
    )
    model = model.to(device)
    
    print(f"Model initialized. Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    # Load datasets
    print("Loading datasets...")
    data_config = config["data"]
    
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
    
    # Initialize trainer
    print("Initializing trainer...")
    training_config = config["training"]
    trainer = VLATrainer(
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        batch_size=data_config["batch_size"],
        learning_rate=training_config["learning_rate"],
        weight_decay=training_config["weight_decay"],
        num_epochs=training_config["num_epochs"],
        device=device,
        save_dir=config["checkpoint"]["save_dir"],
        log_interval=training_config["log_interval"],
        val_interval=training_config["val_interval"],
        save_interval=training_config["save_interval"],
        use_wandb=config.get("wandb", {}).get("enabled", False),
        wandb_project=config.get("wandb", {}).get("project", "atlas-vla"),
        loss=training_config.get("loss", {}),
    )
    
    # Resume from checkpoint if specified
    if args.resume or config["checkpoint"].get("resume_from"):
        checkpoint_path = args.resume or config["checkpoint"]["resume_from"]
        trainer.load_checkpoint(checkpoint_path)
        print(f"Resumed training from {checkpoint_path}")
    
    # Start training
    trainer.train()


if __name__ == "__main__":
    main()
