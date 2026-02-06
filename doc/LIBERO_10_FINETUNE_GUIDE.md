# 使用LIBERO_10数据集进行Atlas Finetune指南

本指南详细说明如何使用LIBERO_10数据集来finetune Atlas VLA模型。

## 目录

1. [概述](#概述)
2. [数据准备](#数据准备)
3. [数据格式转换](#数据格式转换)
4. [配置训练](#配置训练)
5. [开始训练](#开始训练)
6. [常见问题](#常见问题)

---

## 概述

LIBERO_10是LIBERO-100的一个子集，包含10个操作任务，用于测试下游任务的lifelong learning性能。本指南将帮助你：

1. 下载LIBERO_100数据集（包含LIBERO_10）
2. 将LIBERO的HDF5格式转换为Atlas需要的格式
3. 配置并启动finetune训练

---

## 数据准备

### 步骤1: 下载LIBERO数据集

LIBERO_10的数据包含在LIBERO_100数据集中。首先需要下载LIBERO_100数据集。

#### 方法1: 从HuggingFace下载（推荐）

```bash
cd dataset/LIBERO
python benchmark_scripts/download_libero_datasets.py \
    --datasets libero_100 \
    --use-huggingface
```

#### 方法2: 从原始链接下载

```bash
cd dataset/LIBERO
python benchmark_scripts/download_libero_datasets.py \
    --datasets libero_100
```

**注意**: 原始链接可能已过期，推荐使用HuggingFace。

### 步骤2: 检查数据下载

下载完成后，数据会保存在LIBERO的默认数据目录。你可以检查：

```python
from libero.libero import get_libero_path
import os

data_dir = get_libero_path("datasets")
print(f"LIBERO数据目录: {data_dir}")

# 检查libero_100是否存在
libero_100_dir = os.path.join(data_dir, "libero_100")
if os.path.exists(libero_100_dir):
    print("✓ LIBERO_100数据集已下载")
    # 列出任务文件
    tasks = [f for f in os.listdir(libero_100_dir) if f.endswith('.hdf5')]
    print(f"找到 {len(tasks)} 个任务文件")
else:
    print("✗ LIBERO_100数据集未找到")
```

---

## 数据格式转换

LIBERO原始数据是HDF5格式，需要转换为Atlas需要的episode目录格式。

### 使用转换脚本

我们提供了一个转换脚本 `convert_libero_to_atlas_format.py`：

```bash
cd atlas/scripts
python convert_libero_to_atlas_format.py \
    --libero-data-dir /path/to/libero/datasets \
    --output-dir /path/to/output/atlas_format \
    --benchmark libero_10
```

### 转换后的目录结构

转换完成后，你会得到以下结构：

```
atlas_format/
├── train/
│   ├── episode_000000/
│   │   ├── images/
│   │   │   ├── workspace_000000.png
│   │   │   ├── workspace_000001.png
│   │   │   ├── wrist_000000.png
│   │   │   └── ...
│   │   ├── actions.parquet
│   │   └── language_task.txt
│   ├── episode_000001/
│   └── ...
```

### 手动转换（如果需要）

如果自动转换脚本不工作，你可以参考以下步骤手动转换：

1. **读取HDF5文件**:
   ```python
   import h5py
   with h5py.File('path/to/demo.hdf5', 'r') as f:
       demos = list(f['data'].keys())
       for demo_key in demos:
           demo = f['data'][demo_key]
           obs = demo['obs']  # 观测数据
           actions = demo['actions'][:]  # 动作数据
   ```

2. **提取图像**: 从`obs`中提取workspace和wrist相机图像
3. **保存为PNG**: 将图像保存为PNG格式
4. **保存动作**: 将actions保存为parquet或CSV格式
5. **保存语言描述**: 从benchmark获取任务描述并保存为txt文件

---

## 配置训练

### 步骤1: 更新配置文件

编辑 `atlas/configs/train_config.yaml`，更新数据路径：

```yaml
# Data configuration
data:
  data_dir: "/path/to/atlas_format"  # 转换后的数据路径
  train_split: "train"
  val_split: null  # LIBERO_10通常没有验证集，可以设为null或创建验证集
  image_size: 518  # VGGT input size
  use_wrist_camera: true
  batch_size: 8
  num_workers: 4
```

### 步骤2: 调整训练超参数（可选）

根据你的GPU内存和需求调整：

```yaml
training:
  num_epochs: 50
  learning_rate: 1e-4  # 可以尝试 5e-5 或 1e-5
  batch_size: 8  # 如果OOM，减小到4或2
  warmup_steps: 1000
  gradient_accumulation_steps: 1  # 如果batch_size小，可以增加这个值
```

### 步骤3: 模型配置

对于LIBERO_10 finetune，推荐配置：

```yaml
model:
  vggt_checkpoint: "facebook/VGGT-1B"
  lang_encoder_name: "meta-llama/Llama-2-7b-hf"
  freeze_vggt: true  # 推荐先freeze VGGT
  freeze_lang_encoder: true  # 可以尝试unfreeze以提升性能
  geom_output_dim: 512
  fusion_hidden_dim: 1024
  action_dim: 7
```

---

## 开始训练

### 单GPU训练

```bash
cd /workspace/02042026_tingting/AtlasVLA
python atlas/train.py --config atlas/configs/train_config.yaml
```

### 多GPU训练（推荐）

```bash
# 使用4个GPU
torchrun --nproc_per_node=4 \
    atlas/train.py \
    --config atlas/configs/train_config.yaml
```

### 使用特定GPU

```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3
torchrun --nproc_per_node=4 \
    atlas/train.py \
    --config atlas/configs/train_config.yaml
```

### 恢复训练

```bash
python atlas/train.py \
    --config atlas/configs/train_config.yaml \
    --resume checkpoints/checkpoint_epoch_10.pt
```

---

## 训练策略建议

### 阶段1: 快速实验（推荐开始）

- Freeze VGGT和语言编码器
- 只训练fusion层和action head
- 快速迭代，验证流程

```yaml
model:
  freeze_vggt: true
  freeze_lang_encoder: true
```

### 阶段2: 提升性能

- Unfreeze语言编码器
- Fine-tune语言理解

```yaml
model:
  freeze_vggt: true
  freeze_lang_encoder: false  # Unfreeze
```

### 阶段3: 端到端训练（可选）

- Unfreeze所有模块
- 需要更多GPU内存
- 可能提升性能但需要小心过拟合

```yaml
model:
  freeze_vggt: false  # Unfreeze
  freeze_lang_encoder: false
```

---

## 监控训练

### 使用Wandb（推荐）

1. 登录Wandb:
   ```bash
   wandb login
   ```

2. 在配置文件中启用:
   ```yaml
   wandb:
     enabled: true
     project: "atlas-libero-10"
     entity: "your-wandb-username"
   ```

### 检查点文件

训练过程中，checkpoints会保存在 `checkpoints/` 目录：
- `checkpoint_step_*.pt`: 定期保存的checkpoint
- `checkpoint_epoch_*.pt`: 每个epoch结束的checkpoint
- `best_model.pt`: 最佳验证模型（如果有验证集）

---

## 常见问题

### Q1: 数据转换失败

**问题**: 转换脚本报错找不到HDF5文件

**解决**:
1. 确认LIBERO数据集已正确下载
2. 检查数据路径是否正确
3. 确认LIBERO包已安装: `pip install -e dataset/LIBERO`

### Q2: 内存不足（OOM）

**解决**:
1. 减小batch_size（例如从8到4或2）
2. 增加gradient_accumulation_steps
3. 确保VGGT被freeze
4. 使用更少的num_workers

### Q3: 训练很慢

**解决**:
1. 确保使用GPU训练
2. 增加num_workers（但不要超过CPU核心数）
3. 使用混合精度训练（已默认启用）
4. 如果VGGT未freeze，考虑freeze它

### Q4: 损失不下降

**解决**:
1. 检查学习率（尝试更小的值如1e-5）
2. 验证数据格式是否正确
3. 检查语言任务描述是否正确加载
4. 尝试unfreeze语言编码器

### Q5: 如何创建验证集

LIBERO_10通常没有单独的验证集。你可以：

1. **从训练集分割**:
   ```python
   # 使用80%训练，20%验证
   # 在数据加载时实现train/val split
   ```

2. **使用LIBERO_90作为验证**:
   - 下载LIBERO_90数据集
   - 转换为Atlas格式
   - 在配置中设置val_split指向LIBERO_90数据

---

## 完整示例脚本

创建一个完整的训练脚本 `train_libero_10.sh`:

```bash
#!/bin/bash

# 设置GPU
export CUDA_VISIBLE_DEVICES=0,1,2,3

# 设置路径
LIBERO_DATA_DIR="/path/to/libero/datasets"
ATLAS_DATA_DIR="/path/to/atlas_format"
CONFIG_PATH="atlas/configs/train_config.yaml"

# 步骤1: 转换数据（如果还没转换）
echo "Step 1: Converting LIBERO_10 data..."
python atlas/scripts/convert_libero_to_atlas_format.py \
    --libero-data-dir $LIBERO_DATA_DIR \
    --output-dir $ATLAS_DATA_DIR \
    --benchmark libero_10

# 步骤2: 更新配置文件中的数据路径
# (手动编辑或使用sed)
sed -i "s|data_dir:.*|data_dir: $ATLAS_DATA_DIR|" $CONFIG_PATH

# 步骤3: 开始训练
echo "Step 2: Starting training..."
torchrun --nproc_per_node=4 \
    atlas/train.py \
    --config $CONFIG_PATH
```

---

## 预期结果

### 训练时间估算

- **单GPU (RTX 3090)**: 
  - LIBERO_10: ~2-3天（50 epochs）
- **多GPU (4x A100)**:
  - LIBERO_10: ~12-18小时（50 epochs）

### 性能指标

训练过程中关注：
- **Loss**: 应该逐渐下降
- **Action prediction error**: 应该减小
- **如果使用验证集**: Validation loss应该跟踪training loss

---

## 下一步

训练完成后，你可以：

1. **评估模型**: 使用 `atlas/eval.py` 评估性能
2. **测试推理**: 使用训练好的模型进行推理
3. **扩展到LIBERO_90**: 使用更大的数据集继续训练
4. **尝试不同的策略**: 调整freeze策略、学习率等

---

## 参考资源

- [LIBERO官方文档](https://lifelong-robot-learning.github.io/LIBERO/)
- [LIBERO论文](https://arxiv.org/pdf/2306.03310.pdf)
- [Atlas训练文档](atlas/README_TRAINING.md)
- [数据集信息](DATASET_INFO.md)

---

## 获取帮助

如果遇到问题：

1. 检查日志文件
2. 查看常见问题部分
3. 检查数据格式是否正确
4. 验证LIBERO和Atlas安装是否正确

祝训练顺利！🚀
