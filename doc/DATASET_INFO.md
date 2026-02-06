# LIBERO数据集信息

## 1. 数据集大小

### 训练集大小估算

根据HuggingFace信息：
- **训练样本数**: ~67,000 行
- **每个episode**: 通常包含多个时间步（frames）

### 存储空间估算

**图像数据**:
- 图像尺寸: 256×256×3 (RGB)
- 每个图像: ~200 KB (PNG压缩后)
- 每个episode: 假设50帧 × 2相机 = 100图像 = ~20 MB
- 67,000样本: 假设平均每个样本1个episode = ~1.3 TB（未压缩）

**实际压缩后大小**:
- 使用PNG/JPEG压缩: **约50-200 GB**（取决于压缩率）
- Parquet格式（如果使用）: **约10-50 GB**

**完整数据集（包含所有任务套件）**:
- LIBERO-10: ~10 GB
- LIBERO-90: ~90 GB  
- LIBERO-100 (LIBERO-10 + LIBERO-90): ~100 GB
- 所有任务套件: **约100-200 GB**

### 实际下载大小

从HuggingFace下载 `lerobot/libero_object_image`:
- **预计大小**: **20-100 GB**（取决于具体数据集）
- **缓存大小**: 可能更大（HuggingFace会缓存）

### 磁盘空间建议

- **最小**: 50 GB（仅训练集）
- **推荐**: 200 GB（训练集 + 验证集 + 缓存）
- **安全**: 500 GB（包含所有数据和中间文件）

## 2. 数据格式要求

### 目录结构

```
dataset/
├── train/
│   ├── episode_000/
│   │   ├── images/                    # 必需
│   │   │   ├── workspace_000.png     # 必需（或.jpg）
│   │   │   ├── workspace_001.png
│   │   │   ├── wrist_000.png         # 必需（如果use_wrist_camera=True）
│   │   │   ├── wrist_001.png
│   │   │   └── ...
│   │   ├── actions.parquet           # 必需（或actions.csv）
│   │   └── language_task.txt         # 必需（或metadata.json）
│   ├── episode_001/
│   └── ...
└── val/
    └── ... (相同结构)
```

### 文件格式要求

#### 1. 图像文件

**格式**: PNG 或 JPG
**命名规则**:
- Workspace相机: `workspace_XXX.png` (XXX是3位数字，如000, 001, 002...)
- Wrist相机: `wrist_XXX.png` (XXX是3位数字)

**尺寸**: 
- 原始尺寸: 256×256×3 (RGB)
- 训练时会resize到518×518（VGGT输入尺寸）

**示例**:
```
images/
├── workspace_000.png
├── workspace_001.png
├── workspace_002.png
├── wrist_000.png
├── wrist_001.png
└── wrist_002.png
```

#### 2. 动作文件（Actions）

**格式**: 
- **首选**: `actions.parquet` (Parquet格式)
- **备选**: `actions.csv` (CSV格式)

**内容**: 
- 7列数据（6-DOF pose + gripper）
- 每行对应一个时间步
- 列顺序: `[x, y, z, roll, pitch, yaw, gripper]` 或类似

**Parquet格式示例**:
```python
import pandas as pd

# actions.parquet应该包含:
# - 至少7列（6-DOF pose + gripper）
# - 行数应该 >= 图像帧数
actions_df = pd.read_parquet("actions.parquet")
print(actions_df.shape)  # (num_frames, 7)
```

**CSV格式示例**:
```csv
x,y,z,roll,pitch,yaw,gripper
0.1,0.2,0.3,0.0,0.0,0.0,0.0
0.11,0.21,0.31,0.01,0.01,0.01,0.0
...
```

#### 3. 语言任务文件

**格式**: `language_task.txt` (纯文本)

**内容**: 
- 一行文本描述任务
- 例如: "Pick up the red block"

**示例**:
```
Pick up the red block and place it in the blue container
```

**备选格式**: `metadata.json`
```json
{
  "language_task": "Pick up the red block",
  "task_id": "libero_object_0",
  ...
}
```

### 数据格式总结

| 文件类型 | 格式 | 必需 | 命名规则 |
|---------|------|------|---------|
| **Workspace图像** | PNG/JPG | ✅ | `workspace_XXX.png` |
| **Wrist图像** | PNG/JPG | ⚠️ | `wrist_XXX.png` (如果use_wrist_camera=True) |
| **动作数据** | Parquet/CSV | ✅ | `actions.parquet` 或 `actions.csv` |
| **语言任务** | TXT/JSON | ✅ | `language_task.txt` 或 `metadata.json` |

## 3. 数据格式检查

### 快速检查脚本

```python
#!/usr/bin/env python
"""检查LIBERO数据格式"""
import os
from pathlib import Path

def check_episode_format(episode_dir):
    """检查单个episode的格式"""
    episode_path = Path(episode_dir)
    
    issues = []
    
    # 检查images目录
    images_dir = episode_path / "images"
    if not images_dir.exists():
        issues.append("✗ images/ 目录不存在")
    else:
        workspace_imgs = list(images_dir.glob("workspace_*.png")) + list(images_dir.glob("workspace_*.jpg"))
        wrist_imgs = list(images_dir.glob("wrist_*.png")) + list(images_dir.glob("wrist_*.jpg"))
        
        if len(workspace_imgs) == 0:
            issues.append("✗ 没有workspace图像")
        else:
            print(f"  ✓ {len(workspace_imgs)} 个workspace图像")
        
        if len(wrist_imgs) == 0:
            issues.append("⚠ 没有wrist图像（如果不需要可以忽略）")
        else:
            print(f"  ✓ {len(wrist_imgs)} 个wrist图像")
    
    # 检查actions文件
    actions_parquet = episode_path / "actions.parquet"
    actions_csv = episode_path / "actions.csv"
    if actions_parquet.exists():
        print(f"  ✓ actions.parquet 存在")
        import pandas as pd
        df = pd.read_parquet(actions_parquet)
        print(f"    形状: {df.shape}, 列数: {len(df.columns)}")
        if len(df.columns) < 7:
            issues.append(f"⚠ actions只有{len(df.columns)}列，需要至少7列")
    elif actions_csv.exists():
        print(f"  ✓ actions.csv 存在")
        import pandas as pd
        df = pd.read_csv(actions_csv)
        print(f"    形状: {df.shape}, 列数: {len(df.columns)}")
        if len(df.columns) < 7:
            issues.append(f"⚠ actions只有{len(df.columns)}列，需要至少7列")
    else:
        issues.append("✗ actions.parquet 或 actions.csv 不存在")
    
    # 检查language_task
    lang_txt = episode_path / "language_task.txt"
    lang_json = episode_path / "metadata.json"
    if lang_txt.exists():
        print(f"  ✓ language_task.txt 存在")
    elif lang_json.exists():
        print(f"  ✓ metadata.json 存在")
    else:
        issues.append("✗ language_task.txt 或 metadata.json 不存在")
    
    return issues

# 检查数据集
dataset_dir = Path("./dataset/train")
if dataset_dir.exists():
    episodes = sorted([d for d in dataset_dir.iterdir() if d.is_dir() and d.name.startswith("episode_")])
    print(f"找到 {len(episodes)} 个episodes")
    print()
    
    # 检查前3个episodes
    for ep in episodes[:3]:
        print(f"检查 {ep.name}:")
        issues = check_episode_format(ep)
        if issues:
            for issue in issues:
                print(f"  {issue}")
        print()
else:
    print("数据集目录不存在: ./dataset/train")
```

## 4. 从HuggingFace下载后的格式转换

如果从HuggingFace下载的数据格式不同，需要转换。参考 `convert_libero_data.py`。

## 5. 最小测试数据集

如果只想测试，可以创建一个小数据集：

```bash
# 创建测试数据目录
mkdir -p dataset/train/episode_000/images

# 创建几个测试图像（使用任意256x256图像）
# 创建actions.csv
echo "x,y,z,roll,pitch,yaw,gripper" > dataset/train/episode_000/actions.csv
echo "0.1,0.2,0.3,0.0,0.0,0.0,0.0" >> dataset/train/episode_000/actions.csv

# 创建language_task.txt
echo "Pick up the red block" > dataset/train/episode_000/language_task.txt
```

## 总结

### 大小
- **训练集**: 约20-100 GB（取决于具体数据集）
- **完整数据集**: 约100-200 GB
- **推荐磁盘空间**: 200 GB+

### 格式
- **图像**: PNG/JPG, 256×256, 命名 `workspace_XXX.png`, `wrist_XXX.png`
- **动作**: Parquet或CSV, 7列（6-DOF + gripper）
- **语言**: TXT文件，一行文本描述
- **目录结构**: `dataset/train/episode_XXX/`
