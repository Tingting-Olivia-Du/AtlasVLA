"""
LIBERO Dataset Loader - 从 HuggingFace 加载
支持: lerobot/libero_spatial_image 等
"""
import os
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from typing import Dict, Optional, List
from datasets import load_dataset
from PIL import Image
from tqdm import tqdm


class LIBEROHFDataset(Dataset):
    """
    从 HuggingFace 加载 LIBERO 数据集
    支持: lerobot/libero_spatial_image, lerobot/libero_object 等
    """
    def __init__(
        self,
        repo_id: str = "lerobot/libero_spatial_image",
        split: str = "train",
        task_names: Optional[List[str]] = None,
        task_indices: Optional[List[int]] = None,
        max_episodes: Optional[int] = None,
        max_samples: Optional[int] = None,
        transform: Optional[transforms.Compose] = None,
        action_horizon: int = 10,
        use_state: bool = False,
        cache_dir: Optional[str] = None,
        streaming: bool = False,
        verbose: bool = True,
        _hf_dataset=None,
        _episodes=None,
    ):
        self.repo_id = repo_id
        self.split = split
        self.transform = transform
        self.action_horizon = action_horizon
        self.use_state = use_state
        self.verbose = verbose

        def _log(msg): return print(msg) if verbose else None

        # 可选：直接复用已有 dataset 和 episodes，避免重复加载与分组
        if _hf_dataset is not None and _episodes is not None:
            self.dataset = _hf_dataset
            self.episodes = _episodes
            self.max_samples = max_samples
            _log(f"  ✓ Reusing dataset ({len(self.dataset)} samples, {len(self.episodes)} episodes)")
            return

        _log(f"Loading dataset from HuggingFace: {repo_id}")
        _log(f"  Split: {split}")

        # 加载数据集
        try:
            self.dataset = load_dataset(
                repo_id,
                split=split,
                cache_dir=cache_dir,
                streaming=streaming
            )
            _log(f"  ✓ Loaded {len(self.dataset)} samples")
        except Exception as e:
            print(f"Error loading dataset: {e}")
            print("Falling back to demo mode...")
            self.dataset = self._create_dummy_dataset(verbose=verbose)

        # 过滤任务 (task_names 用于有 'task' 字段的数据集; task_indices 用于 LeRobot 等有 task_index 的数据集)
        if task_names is not None:
            _log(f"  Filtering tasks: {task_names}")
            self.dataset = self.dataset.filter(
                lambda x: x.get('task', '') in task_names
            )
            _log(f"  ✓ Filtered to {len(self.dataset)} samples")
        if task_indices is not None:
            _log(f"  Filtering task_indices: {task_indices}")
            self.dataset = self.dataset.filter(
                lambda x: x.get('task_index', -1) in task_indices
            )
            _log(f"  ✓ Filtered to {len(self.dataset)} samples")

        self.max_samples = max_samples
        _log(f"  Grouping by episodes (scanning {len(self.dataset)} samples)...")
        self.episodes = self._group_by_episodes()

        # 限制 episode 数量
        if max_episodes is not None and len(self.episodes) > max_episodes:
            _log(f"  Limiting to {max_episodes} episodes")
            self.episodes = self.episodes[:max_episodes]
        _log(f"  ✓ Organized into {len(self.episodes)} episodes")
    
    def _create_dummy_dataset(self, verbose: bool = True):
        """创建虚拟数据集用于测试"""
        if verbose:
            print("Creating dummy dataset for testing...")
        dummy_data = []
        for ep in range(10):  # 10个episodes
            for t in range(100):  # 每个episode 100步
                dummy_data.append({
                    'episode_index': ep,
                    'timestamp': t,
                    'observation.image': Image.fromarray(
                        np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
                    ),
                    'action': np.random.randn(7).astype(np.float32),
                    'language_instruction': f"Pick and place task {ep}",
                })
        
        from datasets import Dataset as HFDataset
        return HFDataset.from_list(dummy_data)
    
    def _group_by_episodes(self):
        """按 episode 分组数据（大数据集时显示进度）"""
        episodes = {}
        n = len(self.dataset)
        it = tqdm(enumerate(self.dataset), total=n, desc="  Grouping episodes", disable=not self.verbose, unit="samples")
        for idx, sample in it:
            ep_idx = sample.get('episode_index', 0)
            if ep_idx not in episodes:
                episodes[ep_idx] = []
            episodes[ep_idx].append(idx)

        # 转换为列表
        episode_list = []
        for ep_idx, indices in episodes.items():
            episode_list.append({
                'episode_index': ep_idx,
                'indices': indices,
                'length': len(indices),
                'start_idx': indices[0],
                'end_idx': indices[-1]
            })

        return episode_list
    
    def __len__(self):
        # 计算可用的训练样本数（考虑 action_horizon）
        total_samples = sum(
            max(1, ep['length'] - self.action_horizon) 
            for ep in self.episodes
        )
        if self.max_samples is not None:
            return min(total_samples, self.max_samples)
        return total_samples
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        # 找到对应的episode和timestep
        episode_idx, timestep = self._idx_to_episode_timestep(idx)
        episode_info = self.episodes[episode_idx]
        
        # 获取数据
        sample_idx = episode_info['start_idx'] + timestep
        sample = self.dataset[sample_idx]
        
        # 获取图像 (lerobot 格式: observation.images.image, observation.images.wrist_image)
        image = sample.get('observation.image', None)
        if image is None:
            # 尝试 LeRobot/LIBERO 格式的字段名
            for key in [
                'observation.images.image',   # LeRobot libero_spatial_image 主视角
                'observation.images.wrist_image',  # 腕部相机
                'image', 'obs/image', 'observation/image'
            ]:
                if key in sample:
                    image = sample[key]
                    break

        if image is None:
            raise ValueError(f"Could not find image in sample keys: {sample.keys()}")
        
        # 转换图像
        if isinstance(image, Image.Image):
            image = np.array(image)
        
        if self.transform:
            image = self.transform(image)
        else:
            image = torch.from_numpy(image).float()
            if image.ndim == 3 and image.shape[-1] == 3:
                image = image.permute(2, 0, 1) / 255.0
        
        # 获取动作序列
        actions = []
        for t in range(self.action_horizon):
            action_idx = min(sample_idx + t, episode_info['end_idx'])
            action_sample = self.dataset[action_idx]
            action = action_sample.get('action', np.zeros(7))
            if isinstance(action, list):
                action = np.array(action)
            actions.append(action)
        
        actions = np.stack(actions, axis=0)
        actions = torch.from_numpy(actions).float()
        
        # 获取语言指令
        language = sample.get('language_instruction', '')
        if not language:
            language = sample.get('instruction', 'No instruction')
        
        output = {
            'image': image,
            'instruction': language,
            'actions': actions,
            'timestep': timestep,
            'episode_length': episode_info['length']
        }
        
        # 可选: 添加state信息
        if self.use_state:
            state = sample.get('observation.state', None)
            if state is not None:
                if isinstance(state, list):
                    state = np.array(state)
                output['state'] = torch.from_numpy(state).float()
        
        return output
    
    def _idx_to_episode_timestep(self, idx):
        """将全局索引转换为 (episode_idx, timestep)"""
        cumsum = 0
        for ep_idx, ep_info in enumerate(self.episodes):
            valid_steps = max(1, ep_info['length'] - self.action_horizon)
            if idx < cumsum + valid_steps:
                timestep = idx - cumsum
                return ep_idx, timestep
            cumsum += valid_steps
        
        # Fallback: 随机选择
        ep_idx = np.random.randint(len(self.episodes))
        max_t = max(0, self.episodes[ep_idx]['length'] - self.action_horizon)
        timestep = np.random.randint(max_t) if max_t > 0 else 0
        return ep_idx, timestep


def get_libero_hf_dataloaders(
    repo_id: str = "lerobot/libero_spatial_image",
    task_names: Optional[List[str]] = None,
    task_indices: Optional[List[int]] = None,
    max_episodes: Optional[int] = None,
    max_samples: Optional[int] = None,
    batch_size: int = 32,
    num_workers: int = 4,
    action_horizon: int = 10,
    train_split_ratio: float = 0.9,
    cache_dir: Optional[str] = None,
    distributed: bool = False,
    rank: int = 0,
    world_size: int = 1
):
    """
    创建 LIBERO HuggingFace 数据加载器
    
    Args:
        repo_id: HuggingFace dataset repository ID
        task_names: 要加载的任务名称列表 (有 'task' 字段的数据集)
        task_indices: 要加载的 task_index 列表 (LeRobot 格式, 如 [0,1,2] 只加载前3个任务)
        max_episodes: 最多使用的 episode 数量 (用于快速调试)
        max_samples: 最多使用的训练样本数 (用于快速调试)
        batch_size: batch size
        num_workers: 数据加载的worker数量
        action_horizon: 动作预测的时间跨度
        train_split_ratio: 训练集比例
        cache_dir: HuggingFace缓存目录
    """
    
    # 数据增强
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
    
    is_main = (rank == 0)
    # 只加载并分组一次
    full_dataset = LIBEROHFDataset(
        repo_id=repo_id,
        split="train",
        task_names=task_names,
        task_indices=task_indices,
        max_episodes=max_episodes,
        max_samples=max_samples,
        transform=None,
        action_horizon=action_horizon,
        cache_dir=cache_dir,
        verbose=is_main
    )

    # 按 episode 分割训练集和验证集
    num_episodes = len(full_dataset.episodes)
    indices = np.random.permutation(num_episodes)
    split_idx = int(num_episodes * train_split_ratio)
    train_episodes = [full_dataset.episodes[i] for i in indices[:split_idx]]
    val_episodes = [full_dataset.episodes[i] for i in indices[split_idx:]]

    # 复用同一份 dataset，只替换 episodes 和 transform，避免再次加载和分组
    if is_main:
        print("  Building train/val datasets (reusing loaded data)...")
    train_dataset = LIBEROHFDataset(
        repo_id=repo_id,
        split="train",
        task_names=task_names,
        task_indices=task_indices,
        max_episodes=max_episodes,
        max_samples=max_samples,
        transform=train_transform,
        action_horizon=action_horizon,
        cache_dir=cache_dir,
        verbose=is_main,
        _hf_dataset=full_dataset.dataset,
        _episodes=train_episodes,
    )
    val_dataset = LIBEROHFDataset(
        repo_id=repo_id,
        split="train",
        task_names=task_names,
        task_indices=task_indices,
        max_episodes=max_episodes,
        max_samples=max_samples,
        transform=val_transform,
        action_horizon=action_horizon,
        cache_dir=cache_dir,
        verbose=is_main,
        _hf_dataset=full_dataset.dataset,
        _episodes=val_episodes,
    )
    
    if is_main:
        print(f"\nDataset split:")
        print(f"  Train: {len(train_dataset)} samples from {len(train_episodes)} episodes")
        print(f"  Val: {len(val_dataset)} samples from {len(val_episodes)} episodes")

    # 创建 DataLoader (DDP 时使用 DistributedSampler)
    train_sampler = None
    if distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(
            train_dataset, num_replicas=world_size, rank=rank, shuffle=True
        )
        shuffle = False
    else:
        shuffle = True

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=train_sampler,
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
