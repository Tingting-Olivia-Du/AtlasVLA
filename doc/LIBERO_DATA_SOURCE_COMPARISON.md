# LIBERO数据源对比和使用指南

## 📊 两种数据源对比

### 1. 你已下载的数据（原始LIBERO格式）

**位置**: `/workspace/1228_tingting/libero_data/datasets/`

**格式**: 
- **HDF5文件** (`.hdf5`)
- 每个任务一个HDF5文件
- 包含：`libero_object`, `libero_goal`, `libero_spatial` (已完成)

**特点**:
- ✅ 已下载到本地（约13GB）
- ❌ 需要转换为Atlas格式才能使用
- ❌ 不包含LIBERO_10（需要下载libero_100）

**使用方式**: 
需要运行转换脚本 `convert_libero_to_atlas_format.py`

---

### 2. HuggingFace上的数据（physical-intelligence/libero）

**位置**: https://huggingface.co/datasets/physical-intelligence/libero

**格式**:
- **Parquet文件** (已转换好的格式)
- 包含所有四个数据集：Spatial, Object, Goal, **LIBERO-10**
- 273k行数据，34.9 GB

**特点**:
- ✅ **已转换为标准格式**，可直接使用
- ✅ **包含LIBERO_10**
- ✅ 可以直接用Atlas的`LIBEROHFDataset`加载
- ⚠️ 需要网络连接（首次下载会缓存）
- ⚠️ 会占用HuggingFace缓存空间（~35GB）

**使用方式**: 
直接使用 `LIBEROHFDataset`，无需转换！

---

## 🎯 推荐方案：使用HuggingFace数据（更简单）

### 为什么推荐？

1. **无需转换** - 数据已经是标准格式
2. **包含LIBERO_10** - 你需要的子集已经包含
3. **代码已支持** - Atlas已经有`LIBEROHFDataset`
4. **更省事** - 不需要写转换脚本

### 如何使用

#### 方法1: 直接使用LIBEROHFDataset（推荐）

修改 `atlas/train.py` 或创建新的训练脚本：

```python
# 替换原来的 LIBERODataset
from atlas.src.data import LIBEROHFDataset  # 使用HF版本

# 使用HuggingFace数据集
train_dataset = LIBEROHFDataset(
    dataset_name="physical-intelligence/libero",  # HuggingFace数据集名称
    split="train",
    image_size=518,
    use_wrist_camera=True,
    streaming=False,  # False = 下载并缓存，True = 流式加载
)
```

#### 方法2: 修改配置文件支持HF数据集

更新 `atlas/configs/train_config.yaml`:

```yaml
data:
  # 方式1: 使用本地HDF5数据（需要转换）
  # data_dir: "./dataset/libero_10_atlas_format"
  
  # 方式2: 使用HuggingFace数据（推荐，无需转换）
  use_huggingface: true
  hf_dataset_name: "physical-intelligence/libero"
  train_split: "train"
  image_size: 518
  use_wrist_camera: true
  batch_size: 8
```

---

## 🔄 如果要用已下载的数据

### 需要转换

你下载的HDF5数据需要转换为Atlas格式：

```bash
cd atlas/scripts
python convert_libero_to_atlas_format.py \
    --libero-data-dir /workspace/1228_tingting/libero_data/datasets \
    --output-dir /path/to/output/atlas_format \
    --benchmark libero_object  # 或 libero_goal, libero_spatial
```

**注意**: 
- 你下载的数据**不包含LIBERO_10**
- 要使用LIBERO_10，需要下载`libero_100`数据集

---

## 📋 详细对比表

| 特性 | 已下载的HDF5数据 | HuggingFace Parquet数据 |
|------|-----------------|------------------------|
| **格式** | HDF5 | Parquet |
| **位置** | `/workspace/1228_tingting/libero_data/datasets/` | HuggingFace Hub |
| **大小** | ~13 GB (已下载) | ~35 GB (需下载) |
| **包含LIBERO_10** | ❌ 否 | ✅ 是 |
| **需要转换** | ✅ 是 | ❌ 否 |
| **使用难度** | 中等（需转换） | 简单（直接使用） |
| **网络需求** | 无（已下载） | 首次需要 |
| **缓存位置** | 本地目录 | `~/.cache/huggingface/` |

---

## 🚀 快速开始：使用HuggingFace数据

### 步骤1: 安装依赖

```bash
pip install datasets  # 如果还没安装
```

### 步骤2: 修改训练代码

编辑 `atlas/train.py`，找到数据集加载部分（约137行）：

```python
# 原来的代码（使用本地HDF5转换后的数据）
train_dataset = LIBERODataset(
    data_dir=data_config["data_dir"],
    split=data_config["train_split"],
    image_size=data_config["image_size"],
    use_wrist_camera=data_config["use_wrist_camera"],
)
```

**替换为**（使用HuggingFace数据）：

```python
# 使用HuggingFace数据（无需转换）
from atlas.src.data import LIBEROHFDataset

train_dataset = LIBEROHFDataset(
    dataset_name="physical-intelligence/libero",
    split="train",
    image_size=data_config["image_size"],
    use_wrist_camera=data_config["use_wrist_camera"],
    streaming=False,  # False = 下载并缓存到本地
)
```

### 步骤3: 开始训练

```bash
python atlas/train.py --config atlas/configs/train_config.yaml
```

**首次运行会**:
- 自动从HuggingFace下载数据
- 缓存到 `~/.cache/huggingface/datasets/`
- 后续运行直接使用缓存

---

## 💡 两种方式的选择建议

### 使用HuggingFace数据（推荐）✅

**适合**:
- ✅ 想快速开始训练
- ✅ 需要LIBERO_10数据
- ✅ 不想写转换脚本
- ✅ 有稳定的网络连接

**优点**:
- 无需转换，直接可用
- 包含LIBERO_10
- 代码已支持

**缺点**:
- 首次需要下载（~35GB）
- 需要网络连接

---

### 使用已下载的HDF5数据

**适合**:
- ✅ 网络不稳定
- ✅ 想完全离线使用
- ✅ 只需要libero_object/goal/spatial（不需要LIBERO_10）

**优点**:
- 已下载，无需网络
- 完全本地

**缺点**:
- 需要转换脚本
- 不包含LIBERO_10
- 需要更多步骤

---

## 🎯 针对你的情况

**你已下载**: `libero_object`, `libero_goal`, `libero_spatial`

**你需要**: LIBERO_10 来finetune

**推荐方案**: 

**选项1（最简单）**: 使用HuggingFace的`physical-intelligence/libero`
- 直接包含LIBERO_10
- 无需转换
- 代码已支持

**选项2**: 下载`libero_100`然后转换
- 包含LIBERO_10
- 需要转换脚本
- 完全本地

---

## 📝 快速测试HuggingFace数据

创建一个测试脚本 `test_hf_dataset.py`:

```python
from atlas.src.data import LIBEROHFDataset

# 测试加载HuggingFace数据
print("Loading HuggingFace LIBERO dataset...")
dataset = LIBEROHFDataset(
    dataset_name="physical-intelligence/libero",
    split="train",
    image_size=518,
    use_wrist_camera=True,
    streaming=False  # 下载并缓存
)

print(f"Dataset size: {len(dataset)}")

# 测试加载一个样本
sample = dataset[0]
print(f"Sample keys: {sample.keys()}")
print(f"Images shape: {sample['images'].shape}")
print(f"Action shape: {sample['action'].shape}")
print(f"Language: {sample['language_task']}")
```

运行测试：
```bash
python test_hf_dataset.py
```

---

## 🔍 数据格式对比

### HuggingFace格式（physical-intelligence/libero）

```
字段:
- image: [256, 256, 3] - workspace图像
- wrist_image: [256, 256, 3] - wrist图像  
- state: [8] - 状态
- actions: [7] - 动作（6-DOF + gripper）
- timestamp: float32
- frame_index: int64
- episode_index: int64
- task_index: int64
```

### 你下载的HDF5格式

```
结构:
data/
  demo_0/
    obs/
      agentview_image: [T, H, W, 3]
      eye_in_hand_image: [T, H, W, 3]
    actions: [T, 7]
  demo_1/
    ...
```

---

## ✅ 总结

**推荐使用HuggingFace数据** (`physical-intelligence/libero`):
1. ✅ 包含LIBERO_10
2. ✅ 无需转换
3. ✅ 代码已支持
4. ✅ 更简单快速

**你已下载的数据可以**:
- 保留作为备份
- 或者用于其他LIBERO相关实验
- 如果不需要LIBERO_10，也可以转换使用

---

## 🚀 下一步

1. **测试HuggingFace数据加载**（运行上面的测试脚本）
2. **修改训练代码**使用`LIBEROHFDataset`
3. **开始训练**！

需要我帮你修改训练代码来使用HuggingFace数据吗？
