# 使用HuggingFace数据训练指南

## 🎯 快速开始

### 步骤1: 更新配置文件

编辑 `atlas/configs/train_config.yaml`，确保使用HuggingFace数据：

```yaml
data:
  use_huggingface: true  # ✅ 启用HuggingFace数据
  hf_dataset_name: "physical-intelligence/libero"
  streaming: false  # false = 下载并缓存（推荐）
```

### 步骤2: 开始训练

```bash
# 单GPU
python atlas/train.py --config atlas/configs/train_config.yaml

# 多GPU
torchrun --nproc_per_node=4 atlas/train.py --config atlas/configs/train_config.yaml
```

**首次运行会自动下载数据**到 `~/.cache/huggingface/datasets/`（约35GB）

---

## 📋 配置选项说明

### HuggingFace数据配置

```yaml
data:
  use_huggingface: true  # 启用HuggingFace数据源
  hf_dataset_name: "physical-intelligence/libero"  # 数据集名称
  streaming: false  # 是否流式加载
  hf_cache_dir: null  # 缓存目录（可选）
```

**streaming选项**:
- `false`（推荐）: 下载完整数据集并缓存到本地，后续训练无需网络
- `true`: 流式加载，按需下载，节省磁盘但需要稳定网络

### 本地数据配置（如果use_huggingface=false）

```yaml
data:
  use_huggingface: false
  data_dir: "./dataset/libero_10_atlas_format"  # 转换后的数据路径
  train_split: "train"
  val_split: "val"
```

---

## 🔍 数据加载流程

### HuggingFace数据（use_huggingface=true）

1. 首次运行：
   - 自动从HuggingFace下载数据
   - 缓存到 `~/.cache/huggingface/datasets/physical-intelligence___libero/`
   - 约35GB，根据网速需要数小时

2. 后续运行：
   - 直接使用缓存数据
   - 无需网络连接
   - 加载速度快

### 本地数据（use_huggingface=false）

1. 需要先运行转换脚本：
   ```bash
   python atlas/scripts/convert_libero_to_atlas_format.py \
       --output-dir ./dataset/atlas_format
   ```

2. 然后训练：
   ```bash
   python atlas/train.py --config atlas/configs/train_config.yaml
   ```

---

## ✅ 优势对比

| 特性 | HuggingFace数据 | 本地HDF5数据 |
|------|----------------|-------------|
| **包含LIBERO_10** | ✅ 是 | ❌ 否（需要下载libero_100） |
| **需要转换** | ❌ 否 | ✅ 是 |
| **首次使用** | 自动下载 | 需要转换脚本 |
| **后续使用** | 直接使用缓存 | 直接使用 |
| **网络需求** | 首次需要 | 不需要 |

---

## 🚀 推荐配置

```yaml
# atlas/configs/train_config.yaml
data:
  use_huggingface: true  # ✅ 使用HuggingFace数据
  hf_dataset_name: "physical-intelligence/libero"
  streaming: false  # 下载并缓存（推荐）
  train_split: "train"
  val_split: null  # HuggingFace数据通常只有train
  image_size: 518
  use_wrist_camera: true
  batch_size: 8
```

---

## 📝 注意事项

1. **首次下载**: 首次运行会下载约35GB数据，确保：
   - 有足够的磁盘空间（推荐50GB+）
   - 稳定的网络连接
   - 耐心等待下载完成

2. **缓存位置**: 数据会缓存在：
   ```
   ~/.cache/huggingface/datasets/physical-intelligence___libero/
   ```

3. **验证集**: HuggingFace数据通常只有train split，如果需要验证集：
   - 可以手动分割train数据
   - 或者跳过验证（val_split设为null）

4. **网络问题**: 如果下载失败：
   - 检查网络连接
   - 尝试设置代理
   - 或者使用本地数据（需要先转换）

---

## 🔧 故障排除

### 问题1: 下载失败

```bash
# 检查网络连接
ping huggingface.co

# 设置HuggingFace缓存目录（如果需要）
export HF_HOME=/path/to/custom/cache
```

### 问题2: 磁盘空间不足

```bash
# 检查可用空间
df -h ~/.cache/huggingface/

# 清理旧的HuggingFace缓存
rm -rf ~/.cache/huggingface/datasets/physical-intelligence___libero
```

### 问题3: 数据加载错误

检查数据集字段是否匹配：
```python
from datasets import load_dataset
ds = load_dataset("physical-intelligence/libero", split="train")
print(ds[0].keys())  # 查看可用字段
```

---

## 📊 预期结果

- **数据集大小**: 273,465 行
- **包含任务**: LIBERO-Spatial, LIBERO-Object, LIBERO-Goal, LIBERO-10
- **数据格式**: Parquet（已转换好）
- **字段**: image, wrist_image, actions, state, task_index等

---

## 🎉 开始训练！

配置完成后，直接运行：

```bash
python atlas/train.py --config atlas/configs/train_config.yaml
```

训练脚本会自动：
1. 检测配置中的`use_huggingface`选项
2. 使用相应的数据集加载器
3. 开始训练

无需手动切换代码！
