"""
LIBERO Dataset Loader
"""
import os
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from typing import Dict, Optional, List


class LIBERODataset(Dataset):
    def __init__(
        self,
        data_path: str,
        task_names: Optional[List[str]] = None,
        transform: Optional[transforms.Compose] = None,
        mode: str = 'train',
        action_horizon: int = 10,
        use_state: bool = False
    ):
        self.data_path = data_path
        self.transform = transform
        self.mode = mode
        self.action_horizon = action_horizon
        self.use_state = use_state
        
        self.episodes = []
        self._load_episodes(task_names)
        
        print(f"Loaded {len(self.episodes)} episodes for {mode}")
    
    def _load_episodes(self, task_names):
        with h5py.File(self.data_path, 'r') as f:
            data_group = f['data']
            
            if task_names is None:
                task_names = list(data_group.keys())
            
            for task_name in task_names:
                if task_name not in data_group:
                    print(f"Warning: task {task_name} not found")
                    continue
                
                task_group = data_group[task_name]
                demo_keys = [k for k in task_group.keys() if k.startswith('demo_')]
                
                for demo_key in demo_keys:
                    demo = task_group[demo_key]
                    episode_len = demo['actions'].shape[0]
                    language = demo['language'][()].decode('utf-8')
                    
                    self.episodes.append({
                        'task_name': task_name,
                        'demo_key': demo_key,
                        'length': episode_len,
                        'language': language
                    })
    
    def __len__(self):
        total_samples = sum(ep['length'] - self.action_horizon for ep in self.episodes)
        return max(total_samples, len(self.episodes))
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        episode_idx, timestep = self._idx_to_episode_timestep(idx)
        episode_info = self.episodes[episode_idx]
        
        with h5py.File(self.data_path, 'r') as f:
            demo_path = f'data/{episode_info["task_name"]}/{episode_info["demo_key"]}'
            demo = f[demo_path]
            
            image = demo['obs/agentview_rgb'][timestep]
            if self.transform:
                image = self.transform(image)
            else:
                image = torch.from_numpy(image).float().permute(2, 0, 1) / 255.0
            
            max_t = min(timestep + self.action_horizon, episode_info['length'])
            actions = demo['actions'][timestep:max_t]
            
            if actions.shape[0] < self.action_horizon:
                pad_len = self.action_horizon - actions.shape[0]
                actions = np.concatenate([
                    actions,
                    np.repeat(actions[-1:], pad_len, axis=0)
                ], axis=0)
            
            actions = torch.from_numpy(actions).float()
            language = episode_info['language']
            
            output = {
                'image': image,
                'instruction': language,
                'actions': actions,
                'timestep': timestep,
                'episode_length': episode_info['length']
            }
            
            if self.use_state:
                ee_pos = demo['obs/ee_pos'][timestep]
                ee_quat = demo['obs/ee_quat'][timestep]
                gripper = demo['obs/gripper_qpos'][timestep]
                
                state = np.concatenate([ee_pos, ee_quat, gripper])
                output['state'] = torch.from_numpy(state).float()
            
            return output
    
    def _idx_to_episode_timestep(self, idx):
        cumsum = 0
        for ep_idx, ep_info in enumerate(self.episodes):
            valid_steps = max(1, ep_info['length'] - self.action_horizon)
            if idx < cumsum + valid_steps:
                timestep = idx - cumsum
                return ep_idx, timestep
            cumsum += valid_steps
        
        ep_idx = np.random.randint(len(self.episodes))
        max_t = max(0, self.episodes[ep_idx]['length'] - self.action_horizon)
        timestep = np.random.randint(max_t) if max_t > 0 else 0
        return ep_idx, timestep


def get_libero_dataloaders(
    data_path: str,
    task_names: Optional[List[str]] = None,
    batch_size: int = 32,
    num_workers: int = 4,
    action_horizon: int = 10,
    train_split: float = 0.9
):
    train_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.RandomCrop(224, padding=8),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.RandomRotation(5),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    
    val_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    
    full_dataset = LIBERODataset(
        data_path=data_path,
        task_names=task_names,
        transform=None,
        action_horizon=action_horizon
    )
    
    num_episodes = len(full_dataset.episodes)
    indices = np.random.permutation(num_episodes)
    split_idx = int(num_episodes * train_split)
    
    train_episodes = [full_dataset.episodes[i] for i in indices[:split_idx]]
    val_episodes = [full_dataset.episodes[i] for i in indices[split_idx:]]
    
    train_dataset = LIBERODataset(
        data_path=data_path,
        task_names=task_names,
        transform=train_transform,
        mode='train',
        action_horizon=action_horizon
    )
    train_dataset.episodes = train_episodes
    
    val_dataset = LIBERODataset(
        data_path=data_path,
        task_names=task_names,
        transform=val_transform,
        mode='val',
        action_horizon=action_horizon
    )
    val_dataset.episodes = val_episodes
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader
