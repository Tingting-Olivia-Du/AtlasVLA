# LIBERO_10 快速开始指南

## 三步开始训练

### 1. 下载LIBERO_100数据

```bash
cd dataset/LIBERO
python benchmark_scripts/download_libero_datasets.py \
    --datasets libero_100 \
    --use-huggingface
```

### 2. 转换数据格式

```bash
cd atlas/scripts
python convert_libero_to_atlas_format.py \
    --output-dir ../../dataset/libero_10_atlas_format \
    --benchmark libero_10
```

### 3. 开始训练

**方法1: 使用快速启动脚本（推荐）**

```bash
cd atlas/scripts
# 编辑脚本中的路径配置
./train_libero_10.sh
```

**方法2: 手动训练**

```bash
# 更新配置文件中的数据路径
# 编辑 atlas/configs/train_config.yaml，设置 data.data_dir

# 单GPU训练
python atlas/train.py --config atlas/configs/train_config.yaml

# 多GPU训练
torchrun --nproc_per_node=4 atlas/train.py --config atlas/configs/train_config.yaml
```

## 详细文档

查看完整指南: [LIBERO_10_FINETUNE_GUIDE.md](LIBERO_10_FINETUNE_GUIDE.md)

## 常见问题

**Q: 数据转换失败？**
- 确保LIBERO包已安装: `pip install -e dataset/LIBERO`
- 检查数据是否已下载

**Q: 内存不足？**
- 减小batch_size（在config中设置）
- 确保freeze_vggt: true

**Q: 训练很慢？**
- 使用GPU训练
- 增加num_workers
- 确保VGGT被freeze

## 配置示例

```yaml
# atlas/configs/train_config.yaml
data:
  data_dir: "./dataset/libero_10_atlas_format"  # 转换后的数据路径
  train_split: "train"
  val_split: null  # LIBERO_10通常没有验证集
  batch_size: 8

model:
  freeze_vggt: true  # 推荐先freeze
  freeze_lang_encoder: true
```

## 预期训练时间

- 单GPU (RTX 3090): ~2-3天（50 epochs）
- 4x GPU (A100): ~12-18小时（50 epochs）
