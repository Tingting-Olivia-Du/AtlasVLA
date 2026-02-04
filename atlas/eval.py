"""
Evaluation script for Atlas VLA model
"""

import argparse
import torch
import sys
import os
import yaml
from tqdm import tqdm

# Add project root to path for development (if not installed as package)
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

try:
    from atlas.src.models import VGGTVLA
    from atlas.src.data import LIBERODataset
    from atlas.src.training import VLALoss
except ImportError:
    # Fallback for direct script execution
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../vggt'))
    from atlas.src.models import VGGTVLA
    from atlas.src.data import LIBERODataset
    from atlas.src.training import VLALoss

from torch.utils.data import DataLoader


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def evaluate(model, dataloader, criterion, device):
    """Evaluate model on dataset"""
    model.eval()
    
    total_loss = 0.0
    total_pose_loss = 0.0
    total_gripper_loss = 0.0
    all_metrics = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            images = batch["images"].to(device)
            language_tasks = batch["language_task"]
            
            with torch.cuda.amp.autocast():
                outputs = model(images, language_tasks)
                
                loss_dict = criterion(
                    predictions={
                        "pose": outputs["pose"],
                        "gripper": outputs["gripper"]
                    },
                    targets={
                        "pose": batch["pose"].to(device),
                        "gripper": batch["gripper"].to(device)
                    }
                )
                
                metrics = criterion.compute_metrics(
                    predictions={
                        "pose": outputs["pose"],
                        "gripper": outputs["gripper"]
                    },
                    targets={
                        "pose": batch["pose"].to(device),
                        "gripper": batch["gripper"].to(device)
                    }
                )
                
            total_loss += loss_dict["total_loss"].item()
            total_pose_loss += loss_dict["pose_loss"].item()
            total_gripper_loss += loss_dict["gripper_loss"].item()
            all_metrics.append(metrics)
            
    # Average metrics
    num_batches = len(dataloader)
    results = {
        "loss": total_loss / num_batches,
        "pose_loss": total_pose_loss / num_batches,
        "gripper_loss": total_gripper_loss / num_batches,
    }
    
    # Average other metrics
    for key in all_metrics[0].keys():
        results[key] = sum(m[key] for m in all_metrics) / len(all_metrics)
        
    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate Atlas VLA model")
    parser.add_argument(
        "--config",
        type=str,
        default="atlas/configs/train_config.yaml",
        help="Path to config file"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint"
    )
    parser.add_argument(
        "--split",
        type=str,
        default="val",
        help="Dataset split to evaluate on"
    )
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Set device
    device = config.get("device", "cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Initialize model
    print("Loading model...")
    model_config = config["model"]
    model = VGGTVLA(
        vggt_model=None,
        lang_encoder_name=model_config["lang_encoder_name"],
        freeze_vggt=True,  # Always freeze during eval
        freeze_lang_encoder=True,
        geom_output_dim=model_config["geom_output_dim"],
        fusion_hidden_dim=model_config["fusion_hidden_dim"],
        action_dim=model_config["action_dim"],
        use_pointnet=model_config.get("use_pointnet", True),
        use_pose=model_config.get("use_pose", True),
    )
    
    # Load checkpoint
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()
    
    print(f"Loaded checkpoint from {args.checkpoint}")
    if "epoch" in checkpoint:
        print(f"Checkpoint epoch: {checkpoint['epoch']}")
        
    # Load dataset
    print(f"Loading {args.split} dataset...")
    data_config = config["data"]
    dataset = LIBERODataset(
        data_dir=data_config["data_dir"],
        split=args.split,
        image_size=data_config["image_size"],
        use_wrist_camera=data_config["use_wrist_camera"],
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=data_config["batch_size"],
        shuffle=False,
        num_workers=data_config.get("num_workers", 4),
        pin_memory=True,
        collate_fn=LIBERODataset.collate_fn
    )
    
    # Loss function
    criterion = VLALoss(**config["training"].get("loss", {}))
    
    # Evaluate
    print("Running evaluation...")
    results = evaluate(model, dataloader, criterion, device)
    
    # Print results
    print("\n" + "="*50)
    print("Evaluation Results")
    print("="*50)
    for key, value in results.items():
        print(f"{key}: {value:.6f}")
    print("="*50)


if __name__ == "__main__":
    main()
