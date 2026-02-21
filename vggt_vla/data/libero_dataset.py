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
        use_state: bool = False,
        use_multi_view: bool = True,
    ):
        self.data_path = data_path
        self.transform = transform
        self.mode = mode
        self.action_horizon = action_horizon
        self.use_state = use_state
        self.use_multi_view = use_multi_view
        
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
            
            def _load_img(key, fallback=None):
                if key in demo:
                    img = demo[key][timestep]
                    if self.transform:
                        img = self.transform(img)
                    else:
                        img = torch.from_numpy(img).float().permute(2, 0, 1) / 255.0
                    return img
                return fallback

            image_main = _load_img('obs/agentview_rgb')
            if image_main is None:
                raise KeyError("obs/agentview_rgb not found")
            if self.use_multi_view:
                image_wrist = _load_img('obs/robot0_eye_in_hand_image', image_main)
                if image_wrist is None:
                    image_wrist = _load_img('obs/wrist_rgb', image_main)
                if image_wrist is None:
                    image_wrist = image_main.clone()
                image = torch.stack([image_main, image_wrist], dim=0)
            else:
                image = image_main
            
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


def compute_action_stats_libero(
    data_path: str,
    task_names: Optional[List[str]] = None,
    max_samples: Optional[int] = 20000,
    verbose: bool = True,
) -> Optional[Dict[str, np.ndarray]]:
    """
    从本地 LIBERO HDF5 中统计 action 的 mean/std。
    返回 {'mean': (action_dim,), 'std': (action_dim,)} 或 None。
    """
    actions = []
    with h5py.File(data_path, "r") as f:
        data_group = f["data"]
        keys = list(data_group.keys()) if task_names is None else task_names
        for task_name in keys:
            if task_name not in data_group:
                continue
            task_group = data_group[task_name]
            demo_keys = [k for k in task_group.keys() if k.startswith("demo_")]
            for demo_key in demo_keys:
                demo = task_group[demo_key]
                if "actions" not in demo:
                    continue
                arr = demo["actions"][:]
                actions.append(arr)
                if max_samples and sum(a.shape[0] for a in actions) >= max_samples:
                    break
            if max_samples and sum(a.shape[0] for a in actions) >= max_samples:
                break
    if not actions:
        if verbose:
            print("  Warning: no actions found for stats.")
        return None
    actions = np.concatenate(actions, axis=0)
    if max_samples and len(actions) > max_samples:
        actions = actions[np.random.choice(len(actions), max_samples, replace=False)]
    if len(actions) < 2:
        if verbose:
            print("  Warning: not enough actions for stats.")
        return None
    mean = np.ascontiguousarray(actions.mean(axis=0).astype(np.float64))
    std = actions.std(axis=0)
    std = np.maximum(std, 1e-6)
    std = np.ascontiguousarray(std.astype(np.float64))
    if verbose:
        print(f"  Action stats from {len(actions)} samples: mean shape {mean.shape}")
    return {"mean": mean, "std": std}


def get_libero_dataloaders(
    data_path: str,
    task_names: Optional[List[str]] = None,
    batch_size: int = 32,
    num_workers: int = 4,
    action_horizon: int = 10,
    train_split: float = 0.9,
    use_multi_view: bool = True,
    compute_action_stats: bool = True,
    action_stats_max_samples: Optional[int] = 20000,
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
        action_horizon=action_horizon,
        use_multi_view=use_multi_view,
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
        action_horizon=action_horizon,
        use_multi_view=use_multi_view,
    )
    train_dataset.episodes = train_episodes

    val_dataset = LIBERODataset(
        data_path=data_path,
        task_names=task_names,
        transform=val_transform,
        mode='val',
        action_horizon=action_horizon,
        use_multi_view=use_multi_view,
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

    action_stats = None
    if compute_action_stats:
        action_stats = compute_action_stats_libero(
            data_path=data_path,
            task_names=task_names,
            max_samples=action_stats_max_samples,
            verbose=True,
        )

    return train_loader, val_loader, action_stats
