"""
LIBERO Dataset Loader - HuggingFace流式加载版本
直接从HuggingFace加载数据，不下载到本地
"""
import torch
from torch.utils.data import Dataset
import numpy as np
from PIL import Image
from typing import Dict, List, Optional
import io
import os

from .action_normalizer import ActionNormalizer

try:
    from datasets import load_dataset, IterableDataset
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False
    print("Warning: datasets library not available. Install with: pip install datasets")


class LIBEROHFDataset(Dataset):
    """
    LIBERO Dataset直接从HuggingFace加载（流式，不下载）
    
    优点:
    - 不需要下载数据到本地
    - 节省磁盘空间
    - 自动处理数据格式
    
    缺点:
    - 需要稳定的网络连接
    - 首次加载可能较慢（需要下载）
    - 数据会缓存在HuggingFace缓存目录
    
    Args:
        dataset_name: HuggingFace数据集名称 (默认: "lerobot/libero_object_image")
        split: 'train' 或 'val'/'validation'
        image_size: 目标图像尺寸 (默认 518 for VGGT)
        use_wrist_camera: 是否使用wrist相机图像
        streaming: 是否使用流式加载 (默认True，节省内存)
        cache_dir: 缓存目录 (默认None，使用HuggingFace默认缓存)
    """
    
    def __init__(
        self,
        dataset_name: str = "lerobot/libero_object_image",
        split: str = "train",
        image_size: int = 518,
        use_wrist_camera: bool = True,
        streaming: bool = True,
        cache_dir: Optional[str] = None,
        token: Optional[str] = None,
        # 改进4: 多帧时序训练支持
        num_temporal_frames: int = 1,  # 时序帧数，1表示单帧（原始行为）
        temporal_stride: int = 1,  # 帧之间的步长
        # 改进1: 动作归一化支持
        normalize_actions: bool = False,  # 是否归一化动作
        action_stats_path: Optional[str] = None,  # 动作统计信息文件路径
    ):
        if not HF_AVAILABLE:
            raise ImportError(
                "datasets library is required. Install with: pip install datasets"
            )
        
        self.dataset_name = dataset_name
        self.split = split
        self.image_size = image_size
        self.use_wrist_camera = use_wrist_camera
        self.streaming = streaming
        
        # 改进4: 多帧时序训练参数
        self.num_temporal_frames = num_temporal_frames
        self.temporal_stride = temporal_stride
        
        # 改进1: 动作归一化
        self.normalize_actions = normalize_actions
        if normalize_actions:
            self.action_normalizer = ActionNormalizer(stats_path=action_stats_path)
        else:
            self.action_normalizer = None
        
        # 处理split名称
        if split == "val":
            hf_split = "validation"
        else:
            hf_split = split
        
        print(f"Loading LIBERO dataset from HuggingFace: {dataset_name}")
        print(f"  Split: {hf_split}")
        print(f"  Streaming: {streaming}")
        print(f"  Cache dir: {cache_dir or 'HuggingFace default'}")
        
        # 加载数据集
        try:
            # Prepare token parameter if provided
            load_kwargs = {}
            if token:
                load_kwargs['token'] = token
                print(f"  Using HuggingFace token for dataset authentication")
            if cache_dir:
                load_kwargs['cache_dir'] = cache_dir
            
            if streaming:
                # 流式加载（不下载全部数据）
                self.dataset = load_dataset(
                    dataset_name,
                    name="default",  # 指定配置名称
                    split=hf_split,
                    streaming=True,
                    **load_kwargs
                )
                # 流式数据集需要转换为列表（但会占用内存）
                # 更好的方式是使用IterableDataset包装
                print("  Mode: Streaming (on-demand loading)")
                # 对于流式数据集，我们需要先获取长度
                # 但这可能需要遍历一次，所以先设为None
                self._length = None
                self._dataset_list = None  # 延迟加载
            else:
                # 非流式加载（会下载并缓存）
                self.dataset = load_dataset(
                    dataset_name,
                    name="default",  # 指定配置名称
                    split=hf_split,
                    **load_kwargs
                )
                self._length = len(self.dataset)
                print(f"  Loaded {self._length} samples")
        except Exception as e:
            print(f"Error loading dataset: {e}")
            print("\nPossible solutions:")
            print("1. Check internet connection")
            print("2. Run: huggingface-cli login")
            print("3. Check dataset name")
            raise
        
        # 检查数据集结构
        if not streaming and len(self.dataset) > 0:
            print(f"\nDataset structure (first sample):")
            sample = self.dataset[0]
            for key in list(sample.keys())[:10]:  # 只显示前10个字段
                value = sample[key]
                if isinstance(value, (str, int, float)):
                    print(f"  {key}: {value}")
                else:
                    print(f"  {key}: {type(value).__name__}")
        
        # 打印改进功能状态
        if num_temporal_frames > 1:
            print(f"  使用多帧时序训练: {num_temporal_frames} 帧, stride={temporal_stride}")
        if normalize_actions:
            print(f"  动作归一化: 已启用")
    
    def _get_dataset_list(self):
        """延迟加载数据集列表（用于流式数据集）"""
        if self._dataset_list is None:
            print("Converting streaming dataset to list (this may take a while)...")
            self._dataset_list = list(self.dataset)
            self._length = len(self._dataset_list)
            print(f"Loaded {self._length} samples")
        return self._dataset_list
    
    def __len__(self) -> int:
        if self.streaming:
            if self._length is None:
                # 尝试获取长度（可能需要遍历）
                try:
                    # 对于流式数据集，可能需要遍历一次
                    if self._dataset_list is None:
                        self._get_dataset_list()
                    return self._length
                except:
                    # 如果无法获取长度，返回一个估计值
                    return 67000  # 根据HuggingFace信息
            return self._length
        else:
            return len(self.dataset)
    
    def _load_image_from_bytes(self, image_bytes) -> torch.Tensor:
        """从字节数据加载图像"""
        if isinstance(image_bytes, bytes):
            image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        elif isinstance(image_bytes, Image.Image):
            image = image_bytes.convert("RGB")
        else:
            # 如果是PIL Image或numpy array
            image = Image.fromarray(np.array(image_bytes)).convert("RGB")
        
        # Resize到目标尺寸
        image = image.resize((self.image_size, self.image_size), Image.BICUBIC)
        # 转换为tensor: [3, H, W]
        image_tensor = torch.from_numpy(np.array(image)).permute(2, 0, 1).float() / 255.0
        return image_tensor
    
    def _extract_action(self, sample) -> np.ndarray:
        """从样本中提取动作（7-DOF）"""
        # physical-intelligence/libero使用'actions'字段（注意是复数）
        action_fields = ['actions', 'action', 'ee_pose', 'end_effector_pose']
        
        for field in action_fields:
            if field in sample:
                action = sample[field]
                if isinstance(action, (list, np.ndarray)):
                    action = np.array(action)
                    if len(action) >= 7:
                        return action[:7].astype(np.float32)
                    elif len(action) == 6:
                        # 如果只有6-DOF，添加gripper
                        return np.append(action, [0.0]).astype(np.float32)
        
        # 如果找不到，尝试从多个字段组合
        if 'ee_pos' in sample and 'ee_rot' in sample:
            pos = np.array(sample['ee_pos'])
            rot = np.array(sample['ee_rot'])
            gripper = sample.get('gripper', [0.0])
            action = np.concatenate([pos, rot, gripper])[:7]
            return action.astype(np.float32)
        
        # 默认返回零动作
        print(f"Warning: Could not find action in sample. Available keys: {list(sample.keys())}")
        return np.zeros(7, dtype=np.float32)
    
    def _extract_language(self, sample) -> str:
        """从样本中提取语言任务描述"""
        lang_fields = ['language_task', 'task', 'instruction', 'text', 'language']
        
        for field in lang_fields:
            if field in sample:
                lang = sample[field]
                if isinstance(lang, str):
                    return lang.strip()
                elif isinstance(lang, list) and len(lang) > 0:
                    return str(lang[0]).strip()
        
        # 如果physical-intelligence/libero没有language字段，尝试从task_index获取
        # 注意：这需要LIBERO benchmark信息，可能需要额外处理
        if 'task_index' in sample:
            task_idx = sample['task_index']
            # 可以尝试从LIBERO benchmark获取任务描述
            # 但为了简化，先返回通用描述
            return f"LIBERO Task {task_idx}"
        
        return "Unknown task"
    
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
        # 改进4: 多帧时序采样
        # HuggingFace数据集通常每个样本是一个episode的单个帧
        # 为了支持多帧时序，我们需要：
        # 1. 如果num_temporal_frames=1，使用当前实现（单帧）
        # 2. 如果num_temporal_frames>1，需要从episode中采样连续帧
        # 注意：这需要数据集包含episode信息，或者我们假设连续的样本属于同一个episode
        
        if self.num_temporal_frames > 1:
            # 多帧模式：采样连续的多帧
            # 注意：这假设连续的样本索引属于同一个episode
            # 如果数据集结构不同，可能需要调整
            all_images = []
            
            for t in range(self.num_temporal_frames):
                # 计算当前帧的索引
                frame_idx = min(idx + t * self.temporal_stride, len(self.dataset) - 1)
                
                # 获取样本
                if self.streaming:
                    dataset_list = self._get_dataset_list()
                    sample = dataset_list[frame_idx]
                else:
                    sample = self.dataset[frame_idx]
                
                # 加载当前帧的图像
                frame_images = []
                
                # 加载workspace图像
                workspace_fields = ['image', 'workspace_image', 'workspace_rgb', 'images']
                workspace_img = None
                
                for field in workspace_fields:
                    if field in sample:
                        workspace_img = sample[field]
                        break
                
                # 如果字段是列表，取第一个
                if isinstance(workspace_img, list) and len(workspace_img) > 0:
                    workspace_img = workspace_img[0]
                
                if workspace_img is not None:
                    workspace_tensor = self._load_image_from_bytes(workspace_img)
                    frame_images.append(workspace_tensor)
                else:
                    frame_images.append(torch.zeros(3, self.image_size, self.image_size))
                
                # 加载wrist图像（如果启用）
                if self.use_wrist_camera:
                    wrist_fields = ['wrist_image', 'wrist_rgb', 'wrist_camera', 'eye_in_hand_image']
                    wrist_img = None
                    
                    for field in wrist_fields:
                        if field in sample:
                            wrist_img = sample[field]
                            break
                    
                    if isinstance(wrist_img, list) and len(wrist_img) > 0:
                        wrist_img = wrist_img[0]
                    
                    if wrist_img is not None:
                        wrist_tensor = self._load_image_from_bytes(wrist_img)
                        frame_images.append(wrist_tensor)
                    else:
                        frame_images.append(torch.zeros(3, self.image_size, self.image_size))
                
                # 堆叠当前帧的多个视角: [S, 3, H, W]
                frame_stack = torch.stack(frame_images, dim=0)
                all_images.append(frame_stack)
            
            # 堆叠所有时序帧: [T, S, 3, H, W] -> [T*S, 3, H, W]
            images = torch.cat(all_images, dim=0)  # [T*S, 3, H, W]
            
            # 使用最后一帧的动作
            if self.streaming:
                dataset_list = self._get_dataset_list()
                sample = dataset_list[idx]
            else:
                sample = self.dataset[idx]
        else:
            # 单帧模式（原始实现）
            # 获取样本
            if self.streaming:
                dataset_list = self._get_dataset_list()
                sample = dataset_list[idx]
            else:
                sample = self.dataset[idx]
            
            images = []
            
            # 加载workspace图像
            workspace_fields = ['image', 'workspace_image', 'workspace_rgb', 'images']
            workspace_img = None
            
            for field in workspace_fields:
                if field in sample:
                    workspace_img = sample[field]
                    break
            
            # 如果字段是列表，取第一个
            if isinstance(workspace_img, list) and len(workspace_img) > 0:
                workspace_img = workspace_img[0]
            
            if workspace_img is not None:
                workspace_tensor = self._load_image_from_bytes(workspace_img)
                images.append(workspace_tensor)
            else:
                images.append(torch.zeros(3, self.image_size, self.image_size))
            
            # 加载wrist图像（如果启用）
            if self.use_wrist_camera:
                wrist_fields = ['wrist_image', 'wrist_rgb', 'wrist_camera', 'eye_in_hand_image']
                wrist_img = None
                
                for field in wrist_fields:
                    if field in sample:
                        wrist_img = sample[field]
                        break
                
                if isinstance(wrist_img, list) and len(wrist_img) > 0:
                    wrist_img = wrist_img[0]
                
                if wrist_img is not None:
                    wrist_tensor = self._load_image_from_bytes(wrist_img)
                    images.append(wrist_tensor)
                else:
                    images.append(torch.zeros(3, self.image_size, self.image_size))
            
            # Stack images: [S, 3, H, W]
            images = torch.stack(images, dim=0)
        
        # 提取动作
        action = self._extract_action(sample)
        action_tensor = torch.from_numpy(action).float()
        
        # 改进1: 动作归一化
        if self.normalize_actions and self.action_normalizer is not None:
            action_tensor = self.action_normalizer.normalize(action_tensor)
        
        # 提取语言任务
        language_task = self._extract_language(sample)
        
        return {
            "images": images,  # [S或T*S, 3, H, W]
            "action": action_tensor,  # [7] - 可能已归一化
            "pose": action_tensor[:6],  # [6] - 6-DOF pose
            "gripper": action_tensor[6:7],  # [1] - gripper
            "language_task": language_task,
            "episode_idx": idx,
        }
    
    @staticmethod
    def collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
        """
        Custom collate function to handle variable sequence lengths
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
