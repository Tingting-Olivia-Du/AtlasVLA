# 训练配置快速参考

## 核心概念速查

### 1. Warmup（学习率预热）
- **作用**: 训练初期逐步增加学习率，稳定训练
- **配置**: `warmup_steps: 1000` (前1000步)
- **效果**: 避免初期梯度爆炸，提高收敛稳定性

### 2. 梯度累积（Gradient Accumulation）
- **作用**: 模拟更大的batch size，节省显存
- **配置**: `gradient_accumulation_steps: 4`
- **计算**: 有效batch = batch_size × num_gpus × accumulation_steps
- **示例**: batch=8, 8GPU, accum=4 → 有效batch=256

### 3. 学习率调整
- **单GPU**: `lr = 1e-4`
- **8GPU (线性缩放)**: `lr = 1e-4 × 8 = 8e-4` (自动)
- **8GPU (平方根缩放)**: `lr = 1e-4 × √8 ≈ 2.8e-4` (需修改代码)

### 4. 验证频率
- **作用**: 每隔N步验证一次模型性能
- **配置**: `val_interval: 1000` (每1000步)
- **建议**: 
  - 小数据集: 500步
  - 中等数据集: 1000步
  - 大数据集: 2000-5000步

### 5. 多GPU训练
- **命令**: `torchrun --nproc_per_node=8 atlas/train.py --config atlas/configs/train_config.yaml`
- **自动功能**: 
  - 学习率缩放
  - 数据分布式采样
  - 梯度同步

## 配置模板

### 单GPU训练
```yaml
data:
  batch_size: 16

training:
  learning_rate: 1e-4
  warmup_steps: 1000
  gradient_accumulation_steps: 1
  val_interval: 1000
```

### 8GPU训练（推荐）
```yaml
data:
  batch_size: 8  # 每个GPU的batch size
  # 总batch = 8 × 8 = 64

training:
  learning_rate: 1e-4  # 会自动缩放为 8e-4
  warmup_steps: 1000
  gradient_accumulation_steps: 2  # 有效batch = 8×8×2 = 128
  val_interval: 1000
```

### 8GPU + 大有效batch
```yaml
data:
  batch_size: 8

training:
  learning_rate: 1e-4
  warmup_steps: 1000
  gradient_accumulation_steps: 4  # 有效batch = 8×8×4 = 256
  val_interval: 2000  # 减少验证频率
```

## 训练命令

### 单GPU
```bash
python atlas/train.py --config atlas/configs/train_config.yaml
```

### 8GPU
```bash
torchrun --nproc_per_node=8 atlas/train.py --config atlas/configs/train_config.yaml
```

### 4GPU（如果只有4个GPU）
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 atlas/train.py --config atlas/configs/train_config.yaml
```

### 从checkpoint恢复
```bash
torchrun --nproc_per_node=8 atlas/train.py \
  --config atlas/configs/train_config.yaml \
  --resume checkpoints/checkpoint_epoch_10.pt
```

## 常见问题速查

| 问题 | 解决方案 |
|------|---------|
| **OOM (内存不足)** | 1. 减少batch_size<br>2. 增加gradient_accumulation_steps<br>3. 冻结更多模块 |
| **训练太慢** | 1. 增加batch_size<br>2. 减少gradient_accumulation_steps<br>3. 使用更多GPU |
| **验证太慢** | 增加val_interval (如2000或5000) |
| **Loss不下降** | 1. 降低学习率<br>2. 检查数据质量<br>3. 增加warmup_steps |
| **Loss震荡** | 1. 降低学习率<br>2. 增加batch_size或gradient_accumulation_steps |

## 性能优化检查清单

- [ ] Batch size合理（每个GPU 4-16）
- [ ] 学习率已根据GPU数量调整（自动）
- [ ] Warmup steps设置（1000步推荐）
- [ ] 梯度累积设置（如果需要大batch）
- [ ] 验证频率合理（不会太频繁）
- [ ] 使用混合精度训练（已启用）
- [ ] 使用pin_memory（已启用）

## 监控训练

```bash
# 查看GPU使用
watch -n 1 nvidia-smi

# 查看训练日志（rank 0输出）
tail -f checkpoints/train.log  # 如果有日志文件
```

## 相关文档

- 详细概念说明: `TRAINING_CONCEPTS.md`
- 多GPU训练指南: `MULTI_GPU_TRAINING.md`
- 训练配置: `atlas/configs/train_config.yaml`
