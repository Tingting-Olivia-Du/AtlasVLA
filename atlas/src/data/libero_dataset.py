"""
LIBERO Dataset Loader
Loads LIBERO manipulation dataset for VLA training
"""

import torch
from torch.utils.data import Dataset
import numpy as np
from PIL import Image
import os
import json
from typing import Dict, List, Optional, Tuple
import pandas as pd


class LIBERODataset(Dataset):
    """
    LIBERO Dataset for Vision-Language-Action training
    
    Dataset format:
    - Images: 256x256x3 RGB (workspace + wrist cameras)
    - Actions: 7-DOF (6-DOF end-effector pose + gripper)
    - Language: Task descriptions
    
    Args:
        data_dir: Root directory containing LIBERO data
        split: 'train' or 'val'
        image_size: Target image size (default 518 for VGGT)
        use_wrist_camera: Whether to use wrist camera images
        transform: Optional image transforms
    """
    
    def __init__(
        self,
        data_dir: str,
        split: str = "train",
        image_size: int = 518,
        use_wrist_camera: bool = True,
        transform=None,
    ):
        self.data_dir = data_dir
        self.split = split
        self.image_size = image_size
        self.use_wrist_camera = use_wrist_camera
        self.transform = transform
        
        # Load dataset metadata
        self.episodes = self._load_episodes()
        
        print(f"Loaded {len(self.episodes)} episodes from {split} split")
        
    def _load_episodes(self) -> List[Dict]:
        """
        Load episode metadata from LIBERO dataset
        
        LIBERO data structure:
        data_dir/
          train/
            episode_000/
              images/
                workspace_000.png
                wrist_000.png
              actions.parquet
              language_task.txt
        """
        episodes = []
        split_dir = os.path.join(self.data_dir, self.split)
        
        if not os.path.exists(split_dir):
            raise ValueError(f"Split directory not found: {split_dir}")
            
        # Find all episode directories
        episode_dirs = sorted([
            d for d in os.listdir(split_dir) 
            if os.path.isdir(os.path.join(split_dir, d)) and d.startswith("episode_")
        ])
        
        for episode_dir in episode_dirs:
            episode_path = os.path.join(split_dir, episode_dir)
            
            # Load actions
            actions_path = os.path.join(episode_path, "actions.parquet")
            if not os.path.exists(actions_path):
                # Try alternative format (HuggingFace format)
                actions_path = os.path.join(episode_path, "actions.csv")
                
            if os.path.exists(actions_path):
                if actions_path.endswith(".parquet"):
                    actions_df = pd.read_parquet(actions_path)
                else:
                    actions_df = pd.read_csv(actions_path)
                    
                # Load language task
                lang_path = os.path.join(episode_path, "language_task.txt")
                if os.path.exists(lang_path):
                    with open(lang_path, 'r') as f:
                        language_task = f.read().strip()
                else:
                    # Try to get from metadata
                    language_task = self._get_language_from_metadata(episode_path)
                    
                # Get image paths
                images_dir = os.path.join(episode_path, "images")
                if os.path.exists(images_dir):
                    workspace_images = sorted([
                        f for f in os.listdir(images_dir) 
                        if f.startswith("workspace_") and f.endswith((".png", ".jpg"))
                    ])
                    
                    if self.use_wrist_camera:
                        wrist_images = sorted([
                            f for f in os.listdir(images_dir)
                            if f.startswith("wrist_") and f.endswith((".png", ".jpg"))
                        ])
                    else:
                        wrist_images = []
                        
                    # Match images with actions
                    num_frames = len(workspace_images)
                    if num_frames > 0 and len(actions_df) >= num_frames:
                        for i in range(num_frames):
                            episode_data = {
                                "episode_path": episode_path,
                                "frame_idx": i,
                                "workspace_image": os.path.join(images_dir, workspace_images[i]) if i < len(workspace_images) else None,
                                "wrist_image": os.path.join(images_dir, wrist_images[i]) if self.use_wrist_camera and i < len(wrist_images) else None,
                                "action": actions_df.iloc[i].values[:7].astype(np.float32),  # 7-DOF action
                                "language_task": language_task,
                            }
                            episodes.append(episode_data)
                            
        return episodes
    
    def _get_language_from_metadata(self, episode_path: str) -> str:
        """Try to extract language task from metadata files"""
        # Check for metadata.json
        metadata_path = os.path.join(episode_path, "metadata.json")
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
                return metadata.get("language_task", "Unknown task")
        return "Unknown task"
    
    def __len__(self) -> int:
        return len(self.episodes)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a single data sample
        
        Returns:
            dict containing:
                - images: [S, 3, H, W] - Stacked images (workspace + wrist)
                - action: [7] - 7-DOF action
                - language_task: str - Language instruction
        """
        episode = self.episodes[idx]
        
        # Load images
        images = []
        
        # Workspace camera
        if episode["workspace_image"] and os.path.exists(episode["workspace_image"]):
            workspace_img = Image.open(episode["workspace_image"]).convert("RGB")
            workspace_img = workspace_img.resize((self.image_size, self.image_size), Image.BICUBIC)
            workspace_tensor = torch.from_numpy(np.array(workspace_img)).permute(2, 0, 1).float() / 255.0
            images.append(workspace_tensor)
        else:
            # Create dummy image if missing
            images.append(torch.zeros(3, self.image_size, self.image_size))
            
        # Wrist camera (optional)
        if self.use_wrist_camera and episode["wrist_image"] and os.path.exists(episode["wrist_image"]):
            wrist_img = Image.open(episode["wrist_image"]).convert("RGB")
            wrist_img = wrist_img.resize((self.image_size, self.image_size), Image.BICUBIC)
            wrist_tensor = torch.from_numpy(np.array(wrist_img)).permute(2, 0, 1).float() / 255.0
            images.append(wrist_tensor)
            
        # Stack images: [S, 3, H, W]
        images = torch.stack(images, dim=0)
        
        # Apply transforms if provided
        if self.transform is not None:
            images = self.transform(images)
            
        # Load action
        action = torch.from_numpy(episode["action"]).float()
        
        return {
            "images": images,  # [S, 3, H, W]
            "action": action,  # [7]
            "pose": action[:6],  # [6] - 6-DOF pose
            "gripper": action[6:7],  # [1] - gripper
            "language_task": episode["language_task"],
            "episode_idx": idx,
        }
    
    @staticmethod
    def collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
        """
        Custom collate function to handle variable sequence lengths
        
        Args:
            batch: List of samples from __getitem__
            
        Returns:
            Batched data dictionary
        """
        # Get maximum sequence length
        max_seq_len = max([item["images"].shape[0] for item in batch])
        
        # Pad images to same sequence length
        batched_images = []
        batched_actions = []
        batched_poses = []
        batched_grippers = []
        language_tasks = []
        
        for item in batch:
            seq_len = item["images"].shape[0]
            
            # Pad images if needed
            if seq_len < max_seq_len:
                padding = torch.zeros(max_seq_len - seq_len, *item["images"].shape[1:])
                padded_images = torch.cat([item["images"], padding], dim=0)
            else:
                padded_images = item["images"]
                
            batched_images.append(padded_images)
            batched_actions.append(item["action"])
            batched_poses.append(item["pose"])
            batched_grippers.append(item["gripper"])
            language_tasks.append(item["language_task"])
            
        return {
            "images": torch.stack(batched_images, dim=0),  # [B, S, 3, H, W]
            "action": torch.stack(batched_actions, dim=0),  # [B, 7]
            "pose": torch.stack(batched_poses, dim=0),  # [B, 6]
            "gripper": torch.stack(batched_grippers, dim=0),  # [B, 1]
            "language_task": language_tasks,
        }
