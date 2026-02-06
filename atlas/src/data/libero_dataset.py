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

from .action_normalizer import ActionNormalizer


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
        # 改进4: 多帧时序训练支持
        num_temporal_frames: int = 1,  # 时序帧数，1表示单帧（原始行为）
        temporal_stride: int = 1,  # 帧之间的步长
        # 改进1: 动作归一化支持
        normalize_actions: bool = False,  # 是否归一化动作
        action_stats_path: Optional[str] = None,  # 动作统计信息文件路径
    ):
        self.data_dir = data_dir
        self.split = split
        self.image_size = image_size
        self.use_wrist_camera = use_wrist_camera
        self.transform = transform
        
        # 改进4: 多帧时序训练参数
        self.num_temporal_frames = num_temporal_frames
        self.temporal_stride = temporal_stride
        
        # 改进1: 动作归一化
        self.normalize_actions = normalize_actions
        if normalize_actions:
            self.action_normalizer = ActionNormalizer(stats_path=action_stats_path)
        else:
            self.action_normalizer = None
        
        # Load dataset metadata
        self.episodes = self._load_episodes()
        
        print(f"Loaded {len(self.episodes)} episodes from {split} split")
        if num_temporal_frames > 1:
            print(f"  使用多帧时序训练: {num_temporal_frames} 帧, stride={temporal_stride}")
        if normalize_actions:
            print(f"  动作归一化: 已启用")
        
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
                        # 改进4: 支持多帧时序采样
                        # 计算可以采样的起始帧范围（确保有足够的帧）
                        max_start_idx = num_frames - (self.num_temporal_frames - 1) * self.temporal_stride
                        max_start_idx = max(1, max_start_idx)  # 至少保留1个样本
                        
                        for i in range(max_start_idx):
                            episode_data = {
                                "episode_path": episode_path,
                                "frame_idx": i,
                                "num_frames_in_episode": num_frames,  # 记录episode的总帧数
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
        获取单个数据样本
        
        改进4: 支持多帧时序采样
        改进1: 支持动作归一化
        
        Returns:
            dict containing:
                - images: [S*T, 3, H, W] - 堆叠的图像（S=相机数，T=时序帧数）
                - action: [7] - 7-DOF动作（可能已归一化）
                - language_task: str - 语言指令
        """
        episode = self.episodes[idx]
        episode_path = episode["episode_path"]
        start_frame_idx = episode["frame_idx"]
        num_frames_in_episode = episode.get("num_frames_in_episode", 1)
        
        # 改进4: 加载多帧时序图像
        all_images = []  # 存储所有时序帧的图像
        
        for t in range(self.num_temporal_frames):
            # 计算当前帧的索引
            frame_idx = min(start_frame_idx + t * self.temporal_stride, num_frames_in_episode - 1)
            
            # 构建图像路径
            images_dir = os.path.join(episode_path, "images")
            workspace_image_path = os.path.join(images_dir, f"workspace_{frame_idx:03d}.png")
            wrist_image_path = os.path.join(images_dir, f"wrist_{frame_idx:03d}.png")
            
            # 加载当前帧的图像
            frame_images = []
            
            # Workspace camera
            if os.path.exists(workspace_image_path):
                workspace_img = Image.open(workspace_image_path).convert("RGB")
                workspace_img = workspace_img.resize((self.image_size, self.image_size), Image.BICUBIC)
                workspace_tensor = torch.from_numpy(np.array(workspace_img)).permute(2, 0, 1).float() / 255.0
                frame_images.append(workspace_tensor)
            else:
                # 创建dummy图像
                frame_images.append(torch.zeros(3, self.image_size, self.image_size))
            
            # Wrist camera (optional)
            if self.use_wrist_camera:
                if os.path.exists(wrist_image_path):
                    wrist_img = Image.open(wrist_image_path).convert("RGB")
                    wrist_img = wrist_img.resize((self.image_size, self.image_size), Image.BICUBIC)
                    wrist_tensor = torch.from_numpy(np.array(wrist_img)).permute(2, 0, 1).float() / 255.0
                    frame_images.append(wrist_tensor)
                else:
                    frame_images.append(torch.zeros(3, self.image_size, self.image_size))
            
            # 堆叠当前帧的多个视角: [S, 3, H, W]
            frame_stack = torch.stack(frame_images, dim=0)
            all_images.append(frame_stack)
        
        # 堆叠所有时序帧: [T, S, 3, H, W] -> [T*S, 3, H, W]
        if self.num_temporal_frames > 1:
            images = torch.cat(all_images, dim=0)  # [T*S, 3, H, W]
        else:
            images = all_images[0]  # [S, 3, H, W]
        
        # Apply transforms if provided
        if self.transform is not None:
            images = self.transform(images)
        
        # 加载动作
        action = torch.from_numpy(episode["action"]).float()
        
        # 改进1: 动作归一化
        if self.normalize_actions and self.action_normalizer is not None:
            action = self.action_normalizer.normalize(action)
        
        return {
            "images": images,  # [S或T*S, 3, H, W]
            "action": action,  # [7] - 可能已归一化
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
