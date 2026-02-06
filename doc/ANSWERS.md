# 你的问题解答

## 1. Warmup 是什么？

**Warmup（学习率预热）**是在训练初期逐渐增加学习率的过程。

### 为什么需要？
- 模型参数初始化是随机的，一开始就用大学习率可能导致训练不稳定
- 大模型（如LLaMA）在训练初期容易产生大的梯度
- 逐步增加学习率让模型有"适应期"

### 工作原理
```
步数 0:   学习率 = 0
步数 250: 学习率 = 1e-4 * 0.25 = 2.5e-5
步数 500: 学习率 = 1e-4 * 0.5  = 5e-5
步数 1000: 学习率 = 1e-4 (达到目标值)
之后: 使用余弦衰减
```

### 在你的代码中
- **配置**: `warmup_steps: 1000` (在 `train_config.yaml` 中)
- **已实现**: ✅ 代码已支持warmup，使用 `LambdaLR` scheduler

---

## 2. 梯度累积是什么？

**梯度累积**是将多个小batch的梯度累积起来，然后一次性更新参数。

### 为什么需要？
- **节省显存**: GPU显存不够时，无法使用大batch size
- **模拟大batch**: 累积4个batch_size=8的梯度 = 实际batch_size=32的效果
- **更稳定的训练**: 大batch通常训练更稳定

### 工作原理
```
正常训练（batch_size=32）:
  前向传播 → 计算loss → 反向传播 → 更新参数

梯度累积（accumulation_steps=4, batch_size=8）:
  Step 1: 前向传播(batch=8) → 反向传播 → 累积梯度（不更新）
  Step 2: 前向传播(batch=8) → 反向传播 → 累积梯度（不更新）
  Step 3: 前向传播(batch=8) → 反向传播 → 累积梯度（不更新）
  Step 4: 前向传播(batch=8) → 反向传播 → 累积梯度 → 更新参数（相当于batch=32）
```

### 在你的代码中
- **配置**: `gradient_accumulation_steps: 1` (在 `train_config.yaml` 中)
- **已实现**: ✅ 代码已支持梯度累积
- **使用**: 设置为2-4可以模拟更大的batch size

### 计算有效batch size
```
有效batch size = batch_size × num_gpus × gradient_accumulation_steps

示例：
- batch_size=8, 8GPU, accum=1 → 有效batch=64
- batch_size=8, 8GPU, accum=4 → 有效batch=256
```

---

## 3. 学习率怎么调整？

### 自动调整（已实现）

代码会根据GPU数量自动调整学习率（线性缩放规则）：

```python
# 在 train.py 中
if is_distributed:
    effective_lr = base_lr * world_size
    # base_lr = 1e-4, world_size = 8
    # effective_lr = 1e-4 * 8 = 8e-4
```

### 手动调整策略

#### 1. 调整基础学习率
```yaml
training:
  learning_rate: 1e-4  # 修改这个值
```

**推荐值**：
- 小模型: `1e-3` 到 `1e-4`
- 大模型（LLaMA）: `1e-4` 到 `5e-5`
- 冻结backbone时: `1e-3` 到 `1e-4`

#### 2. 修改缩放策略（如果需要）

编辑 `atlas/train.py`，找到：
```python
effective_lr = base_lr * world_size  # 线性缩放
```

可以改为：
```python
# 平方根缩放（更保守）
effective_lr = base_lr * math.sqrt(world_size)

# 或不缩放（需要手动调整）
effective_lr = base_lr
```

#### 3. 不同模块不同学习率（高级）

可以在 `train.py` 中为不同模块设置不同学习率：
```python
# VGGT (冻结): 0
# Language Encoder: 1e-5 (小学习率)
# Fusion + Action Head: 1e-4 (新模块，可以用大学习率)
```

### 调整建议

| 情况 | 操作 |
|------|------|
| Loss不下降 | 降低学习率（除以2或10） |
| Loss震荡 | 降低学习率 |
| Loss下降太慢 | 可以尝试增加学习率（但要小心） |
| 训练初期不稳定 | 增加warmup_steps |

---

## 4. 验证频率是什么？

**验证频率**决定每隔多少训练步进行一次验证。

### 作用
- **监控过拟合**: 检查模型在验证集上的表现
- **选择最佳模型**: 保存验证loss最低的模型
- **调整训练策略**: 如果验证loss不降，可能需要调整

### 配置
```yaml
training:
  val_interval: 1000  # 每1000步验证一次
```

### 如何设置？

**根据数据集大小**：
- 小数据集（<10k样本）: `val_interval: 500`
- 中等数据集（10k-100k）: `val_interval: 1000` ✅ **推荐**
- 大数据集（>100k）: `val_interval: 2000-5000`

**根据训练时间**：
- 如果验证很慢：减少验证频率（如5000步一次）
- 如果验证很快：可以增加频率（如500步一次）

### 注意
- 验证会暂停训练
- 如果验证集很大，验证可能很慢
- 建议根据实际情况调整

---

## 5. 8个GPU怎么配置？

### ✅ 已实现：代码已支持多GPU训练！

### 使用方法

#### 方法1：使用 torchrun（推荐）

```bash
# 使用8个GPU训练
torchrun --nproc_per_node=8 atlas/train.py --config atlas/configs/train_config.yaml
```

#### 方法2：使用 torch.distributed.launch

```bash
python -m torch.distributed.launch \
  --nproc_per_node=8 \
  --master_port=29500 \
  atlas/train.py --config atlas/configs/train_config.yaml
```

### 自动功能

代码会自动：
1. ✅ **检测分布式环境**: 自动检测是否在多GPU模式下运行
2. ✅ **包装模型**: 使用 `DistributedDataParallel` (DDP) 包装模型
3. ✅ **数据采样**: 使用 `DistributedSampler` 分配数据
4. ✅ **学习率缩放**: 根据GPU数量自动调整学习率
5. ✅ **梯度同步**: 自动同步所有GPU的梯度

### 配置说明

```yaml
data:
  batch_size: 8  # 这是每个GPU的batch size
  # 实际总batch = 8 × 8 GPUs = 64

training:
  learning_rate: 1e-4  # 基础学习率
  # 实际学习率 = 1e-4 × 8 = 8e-4 (自动调整)
```

### 验证是否使用多GPU

训练开始时会输出：
```
Training setup:
  Distributed: True
  World size: 8
  Rank: 0, Local rank: 0
  Device: cuda:0
```

或者运行 `nvidia-smi` 查看所有GPU的使用情况。

### 只使用部分GPU

```bash
# 只使用GPU 0, 1, 2, 3
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 atlas/train.py \
  --config atlas/configs/train_config.yaml
```

### 完整示例

```bash
# 1. 激活环境
# conda activate your_env

# 2. 开始训练（8个GPU）
torchrun --nproc_per_node=8 atlas/train.py \
  --config atlas/configs/train_config.yaml

# 3. 监控GPU使用（另一个终端）
watch -n 1 nvidia-smi
```

---

## 总结

| 功能 | 状态 | 配置位置 |
|------|------|---------|
| **Warmup** | ✅ 已实现 | `training.warmup_steps` |
| **梯度累积** | ✅ 已实现 | `training.gradient_accumulation_steps` |
| **学习率调整** | ✅ 已实现（自动） | `training.learning_rate` |
| **验证频率** | ✅ 已实现 | `training.val_interval` |
| **多GPU训练** | ✅ 已实现 | 使用 `torchrun --nproc_per_node=8` |

所有功能都已实现并可以使用！🎉

## 下一步

1. 查看详细文档：
   - `TRAINING_CONCEPTS.md` - 详细概念说明
   - `MULTI_GPU_TRAINING.md` - 多GPU训练指南
   - `QUICK_REFERENCE.md` - 快速参考

2. 开始训练：
   ```bash
   torchrun --nproc_per_node=8 atlas/train.py --config atlas/configs/train_config.yaml
   ```

3. 根据实际情况调整配置（batch size, learning rate等）
