# HuggingFace数据集支持说明

## 📊 问题解答

### Q: 能否使用 `physical-intelligence/libero` 数据训练？

**A: 可以！** 现在 `LIBEROHFDataset` 已经完全支持所有改进功能，包括：
- ✅ 多帧时序训练
- ✅ 动作归一化
- ✅ 所有其他架构改进

### Q: 是否需要下载原始数据？

**A: 不需要！** 可以直接使用HuggingFace上的 `physical-intelligence/libero` 数据集，无需下载原始HDF5格式数据。

---

## 🎯 使用 `physical-intelligence/libero` 的优势

### 优点

1. **无需下载** - 数据直接从HuggingFace加载，节省本地磁盘空间
2. **自动缓存** - HuggingFace会自动缓存数据，后续加载更快
3. **包含LIBERO_10** - 包含完整的LIBERO-10子集
4. **标准格式** - 数据已经是标准格式，无需转换
5. **支持所有改进** - 现在完全支持多帧时序和动作归一化

### 缺点

1. **需要网络** - 首次加载需要网络连接（后续使用缓存）
2. **缓存空间** - 数据会缓存在 `~/.cache/huggingface/datasets/`（约35GB）

---

## 🚀 使用方法

### 配置示例

```yaml
# atlas/configs/train_config.yaml
data:
  use_huggingface: true  # 使用HuggingFace数据
  hf_dataset_name: "physical-intelligence/libero"
  streaming: false  # false=下载并缓存，true=流式加载
  
  # 改进4: 多帧时序训练（现在支持！）
  num_temporal_frames: 4  # 使用4帧时序
  temporal_stride: 1
  
  # 改进1: 动作归一化（现在支持！）
  normalize_actions: true
  action_stats_path: null  # null=自动计算
```

### 开始训练

```bash
python atlas/train.py --config atlas/configs/train_config.yaml
```

---

## 📝 实现细节

### 多帧时序支持

`LIBEROHFDataset` 现在支持多帧时序训练：

```python
# 单帧模式（默认）
dataset = LIBEROHFDataset(num_temporal_frames=1)

# 多帧时序模式
dataset = LIBEROHFDataset(
    num_temporal_frames=4,  # 使用4帧
    temporal_stride=1  # 帧之间步长为1
)
```

**工作原理**：
- HuggingFace数据集中的每个样本是一个episode的单个帧
- 多帧模式会采样连续的样本作为时序帧
- 假设连续的样本索引属于同一个episode（对于大多数数据集结构都适用）

### 动作归一化支持

`LIBEROHFDataset` 现在完全支持动作归一化：

```python
dataset = LIBEROHFDataset(
    normalize_actions=True,
    action_stats_path="action_stats.pt"  # 或null自动计算
)
```

**工作原理**：
- 使用 `ActionNormalizer` 类进行归一化
- 可以预先计算统计信息，或让dataset自动计算
- 归一化在 `__getitem__` 时自动应用

---

## ⚠️ 注意事项

### 1. 多帧时序的限制

**当前实现假设**：
- 连续的样本索引属于同一个episode
- 这对于大多数数据集结构都适用

**如果数据集结构不同**：
- 如果每个样本是独立的（不按episode组织），多帧时序可能采样到不同episode的帧
- 这种情况下，建议使用 `num_temporal_frames=1`（单帧模式）
- 或者转换为本地格式，使用 `LIBERODataset`（支持episode级别的多帧采样）

### 2. 动作统计信息

**自动计算**：
- 如果 `action_stats_path` 为 `null`，dataset会在首次使用时计算统计信息
- 这可能需要遍历部分数据，可能较慢

**预先计算**（推荐）：
```python
from atlas.src.data.action_normalizer import ActionNormalizer
import numpy as np

# 收集动作数据
actions = []
dataset = LIBEROHFDataset(normalize_actions=False)  # 先不归一化
for i in range(min(10000, len(dataset))):  # 采样部分数据
    sample = dataset[i]
    actions.append(sample["action"].numpy())

# 计算统计信息
actions_array = np.array(actions)
normalizer = ActionNormalizer()
stats = normalizer.compute_stats(actions_array)
normalizer.save_stats("action_stats.pt")

# 然后使用
dataset = LIBEROHFDataset(
    normalize_actions=True,
    action_stats_path="action_stats.pt"
)
```

### 3. 缓存管理

HuggingFace数据会缓存在：
```
~/.cache/huggingface/datasets/physical-intelligence___libero/
```

**清理缓存**（如果需要）：
```bash
rm -rf ~/.cache/huggingface/datasets/physical-intelligence___libero
```

---

## 🔄 与本地数据的对比

| 特性 | HuggingFace数据 | 本地HDF5数据 |
|------|----------------|-------------|
| **下载需求** | ❌ 不需要 | ✅ 需要下载（~100GB） |
| **磁盘空间** | ~35GB缓存 | ~100GB原始数据 |
| **多帧时序** | ✅ 支持（已实现） | ✅ 支持 |
| **动作归一化** | ✅ 支持（已实现） | ✅ 支持 |
| **包含LIBERO_10** | ✅ 是 | ❌ 需要单独下载 |
| **网络需求** | ✅ 首次需要 | ❌ 不需要 |
| **数据转换** | ❌ 不需要 | ✅ 需要转换 |

---

## ✅ 推荐方案

### 方案1: 使用HuggingFace数据（推荐）⭐

**适用场景**：
- 首次训练
- 需要LIBERO_10
- 磁盘空间有限
- 有稳定的网络连接

**配置**：
```yaml
data:
  use_huggingface: true
  hf_dataset_name: "physical-intelligence/libero"
  num_temporal_frames: 4  # 支持多帧
  normalize_actions: true  # 支持归一化
```

### 方案2: 使用本地数据

**适用场景**：
- 无网络环境
- 需要精确的episode级别多帧采样
- 已有本地HDF5数据

**配置**：
```yaml
data:
  use_huggingface: false
  data_dir: "./dataset/libero_converted"
  num_temporal_frames: 4
  normalize_actions: true
```

---

## 🎓 总结

**现在可以直接使用 `physical-intelligence/libero` 进行训练！**

- ✅ 支持所有架构改进
- ✅ 无需下载原始数据
- ✅ 包含LIBERO_10
- ✅ 使用简单，配置即可

只需要在配置文件中设置：
```yaml
data:
  use_huggingface: true
  hf_dataset_name: "physical-intelligence/libero"
  num_temporal_frames: 4  # 可选：多帧时序
  normalize_actions: true  # 可选：动作归一化
```

---

**最后更新**: 2026-02-07
**作者**: Atlas VLA Team
