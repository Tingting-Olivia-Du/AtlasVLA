# 多GPU训练指南

## 快速开始（8个GPU）

### 方法1：使用 torchrun（推荐，PyTorch 2.0+）

```bash
# 使用8个GPU训练
torchrun --nproc_per_node=8 atlas/train.py --config atlas/configs/train_config.yaml
```

### 方法2：使用 torch.distributed.launch（兼容旧版本）

```bash
python -m torch.distributed.launch \
  --nproc_per_node=8 \
  --master_port=29500 \
  atlas/train.py --config atlas/configs/train_config.yaml
```

### 方法3：使用环境变量（手动设置）

```bash
# 设置环境变量
export MASTER_ADDR=localhost
export MASTER_PORT=29500
export WORLD_SIZE=8
export RANK=0  # 每个进程设置不同的RANK (0-7)
export LOCAL_RANK=0  # 每个进程设置不同的LOCAL_RANK (0-7)

# 然后运行8个进程（每个GPU一个）
# GPU 0
CUDA_VISIBLE_DEVICES=0 python atlas/train.py --config atlas/configs/train_config.yaml &
# GPU 1
CUDA_VISIBLE_DEVICES=1 python atlas/train.py --config atlas/configs/train_config.yaml &
# ... 等等（不推荐，太麻烦）
```

## 配置说明

### 1. Batch Size 配置

```yaml
data:
  batch_size: 8  # 这是每个GPU的batch size
```

**实际总batch size计算**：
- 单GPU: `batch_size = 8`
- 8个GPU: `总batch_size = 8 × 8 = 64`
- 如果使用梯度累积: `有效batch_size = 8 × 8 × gradient_accumulation_steps`

### 2. 学习率调整

代码会自动根据GPU数量调整学习率（线性缩放规则）：

```python
# 在 train.py 中
if is_distributed:
    effective_lr = base_lr * world_size
    # base_lr = 1e-4, world_size = 8
    # effective_lr = 1e-4 * 8 = 8e-4
```

**如果你想使用平方根缩放**（更保守），可以修改 `train.py`：

```python
# 替换线性缩放为平方根缩放
effective_lr = base_lr * math.sqrt(world_size)
# effective_lr = 1e-4 * sqrt(8) ≈ 2.8e-4
```

### 3. 梯度累积

```yaml
training:
  gradient_accumulation_steps: 4  # 累积4步再更新
```

**效果**：
- 每个GPU: `batch_size = 8`
- 梯度累积: `4步`
- 8个GPU
- **有效batch size = 8 × 8 × 4 = 256**

### 4. Warmup Steps

```yaml
training:
  warmup_steps: 1000  # 前1000步进行学习率预热
```

**注意**：Warmup steps是全局的，不是每个GPU的。

## 完整配置示例

```yaml
# atlas/configs/train_config.yaml

data:
  batch_size: 8  # 每个GPU的batch size
  # 实际总batch = 8 × 8 GPUs = 64

training:
  learning_rate: 1e-4  # 基础学习率
  # 实际学习率 = 1e-4 × 8 = 8e-4 (自动调整)
  warmup_steps: 1000
  gradient_accumulation_steps: 2  # 累积2步
  # 有效batch size = 8 × 8 × 2 = 128
  val_interval: 1000  # 每1000步验证一次
```

## 训练命令示例

### 基本训练（8个GPU）

```bash
torchrun --nproc_per_node=8 atlas/train.py \
  --config atlas/configs/train_config.yaml
```

### 从checkpoint恢复

```bash
torchrun --nproc_per_node=8 atlas/train.py \
  --config atlas/configs/train_config.yaml \
  --resume checkpoints/checkpoint_epoch_10.pt
```

### 指定特定GPU（如果不想用全部8个）

```bash
# 只使用GPU 0, 1, 2, 3
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 atlas/train.py \
  --config atlas/configs/train_config.yaml
```

## 监控训练

### 查看GPU使用情况

```bash
# 在另一个终端运行
watch -n 1 nvidia-smi
```

### 查看训练日志

训练日志会从rank 0（主进程）输出。如果使用wandb，只有rank 0会记录。

## 常见问题

### Q1: 如何知道训练是否使用了所有GPU？

**A**: 检查日志输出：
```
Training setup:
  Distributed: True
  World size: 8
  Rank: 0, Local rank: 0
```

或者运行 `nvidia-smi` 查看所有GPU的使用情况。

### Q2: 为什么学习率变大了？

**A**: 这是正常的！代码使用线性缩放规则：
- 单GPU: `lr = 1e-4`
- 8个GPU: `lr = 1e-4 × 8 = 8e-4`

这是因为总batch size变大了，需要相应调整学习率。

### Q3: 如何调整学习率缩放策略？

**A**: 编辑 `atlas/train.py`，找到学习率计算部分：

```python
# 当前（线性缩放）
effective_lr = base_lr * world_size

# 改为平方根缩放（更保守）
effective_lr = base_lr * math.sqrt(world_size)

# 或者不缩放（需要手动调整）
effective_lr = base_lr
```

### Q4: 验证很慢怎么办？

**A**: 增加 `val_interval`：

```yaml
training:
  val_interval: 2000  # 改为每2000步验证一次
```

### Q5: 内存不足（OOM）怎么办？

**A**: 有几个选项：

1. **减少batch size**：
```yaml
data:
  batch_size: 4  # 从8减到4
```

2. **使用梯度累积**：
```yaml
training:
  gradient_accumulation_steps: 4  # 累积4步
  # 有效batch = 4 × 8 × 4 = 128 (和原来batch=8, 8GPU一样)
```

3. **冻结更多模块**：
```yaml
model:
  freeze_vggt: true
  freeze_lang_encoder: true  # 也冻结语言编码器
```

### Q6: 如何只使用部分GPU？

**A**: 使用 `CUDA_VISIBLE_DEVICES`：

```bash
# 只使用GPU 0, 1, 2, 3
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 atlas/train.py \
  --config atlas/configs/train_config.yaml
```

## 性能优化建议

1. **Batch Size**: 每个GPU的batch size建议在4-16之间
2. **Gradient Accumulation**: 如果内存不够，用梯度累积模拟大batch
3. **Num Workers**: 根据CPU核心数调整 `num_workers`（通常4-8）
4. **Mixed Precision**: 已经启用，可以节省内存和加速训练
5. **Pin Memory**: 已经启用，可以加速数据传输

## 检查清单

训练前检查：

- [ ] 确认所有8个GPU都可用：`nvidia-smi`
- [ ] 确认数据路径正确：`data_dir` 在config中
- [ ] 确认batch size合理：每个GPU的batch size × 8不会OOM
- [ ] 确认学习率：代码会自动缩放，但可以检查日志
- [ ] 确认checkpoint目录可写：`save_dir` 在config中

## 示例：完整训练命令

```bash
# 1. 激活环境（如果有conda/virtualenv）
# conda activate your_env

# 2. 设置wandb（可选）
export WANDB_API_KEY=your_key

# 3. 开始训练（8个GPU）
torchrun --nproc_per_node=8 atlas/train.py \
  --config atlas/configs/train_config.yaml

# 4. 监控（另一个终端）
watch -n 1 nvidia-smi
```

## 参考

- PyTorch Distributed Training: https://pytorch.org/tutorials/intermediate/ddp_tutorial.html
- Learning Rate Scaling: https://arxiv.org/abs/1706.02677
- Gradient Accumulation: https://pytorch.org/docs/stable/notes/amp_examples.html#gradient-accumulation
