# 统一训练脚本快速开始

## 🎯 一句话开始

```bash
python atlas/train.py --config atlas/configs/train_config.yaml
```

训练脚本会自动根据配置文件选择数据源！

---

## 📋 配置数据源

### 方式1: 使用HuggingFace数据（推荐）✅

编辑 `atlas/configs/train_config.yaml`:

```yaml
data:
  use_huggingface: true  # ✅ 启用HuggingFace
  hf_dataset_name: "physical-intelligence/libero"
  streaming: false
```

**优点**:
- ✅ 包含LIBERO_10
- ✅ 无需转换
- ✅ 自动下载和缓存

### 方式2: 使用本地数据

```yaml
data:
  use_huggingface: false  # 使用本地数据
  data_dir: "./dataset/atlas_format"  # 转换后的数据路径
```

**需要先转换**:
```bash
python atlas/scripts/convert_libero_to_atlas_format.py \
    --output-dir ./dataset/atlas_format
```

---

## 🚀 完整训练命令

### 单GPU

```bash
python atlas/train.py --config atlas/configs/train_config.yaml
```

### 多GPU

```bash
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

## 📝 配置文件示例

### HuggingFace数据（推荐）

```yaml
# atlas/configs/train_config.yaml
data:
  use_huggingface: true
  hf_dataset_name: "physical-intelligence/libero"
  streaming: false
  train_split: "train"
  val_split: null  # HuggingFace数据通常只有train
  image_size: 518
  use_wrist_camera: true
  batch_size: 8
```

### 本地数据

```yaml
data:
  use_huggingface: false
  data_dir: "./dataset/atlas_format"
  train_split: "train"
  val_split: "val"
  image_size: 518
  use_wrist_camera: true
  batch_size: 8
```

---

## ✅ 统一后的优势

1. **一个脚本** - `atlas/train.py` 支持两种数据源
2. **配置驱动** - 通过YAML配置切换，无需改代码
3. **自动适配** - 自动选择正确的数据集类和collate函数
4. **向后兼容** - 仍然支持本地HDF5转换后的数据

---

## 🔍 如何验证配置

运行时会显示使用的数据源：

```
Loading datasets...
  Using HuggingFace dataset: physical-intelligence/libero
  Streaming mode: False
```

或

```
Loading datasets...
  Using local dataset from: ./dataset/atlas_format
```

---

## 📚 详细文档

- HuggingFace数据使用: `atlas/USING_HUGGINGFACE_DATA.md`
- 数据源对比: `atlas/LIBERO_DATA_SOURCE_COMPARISON.md`
- LIBERO_10训练指南: `atlas/LIBERO_10_FINETUNE_GUIDE.md`

---

## 🎉 开始训练！

配置完成后，直接运行：

```bash
python atlas/train.py --config atlas/configs/train_config.yaml
```

就这么简单！
