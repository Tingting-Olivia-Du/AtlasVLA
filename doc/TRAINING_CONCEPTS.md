# 训练概念详解

## 1. Warmup（学习率预热）是什么？

### 概念
**Warmup** 是在训练初期逐渐增加学习率的过程，而不是一开始就使用目标学习率。

### 为什么需要Warmup？
- **稳定训练**：模型参数在初始化时是随机的，如果一开始就用大学习率，可能导致训练不稳定
- **避免梯度爆炸**：大模型（如LLaMA）在训练初期容易产生大的梯度，Warmup可以平滑这个过程
- **更好的收敛**：逐步增加学习率让模型有"适应期"

### 示例
```
没有Warmup:
学习率: 1e-4 → 1e-4 → 1e-4 → ... (直接跳到目标值)

有Warmup (1000步):
步数 0:   学习率 = 0
步数 250: 学习率 = 1e-4 * 0.25 = 2.5e-5
步数 500: 学习率 = 1e-4 * 0.5  = 5e-5
步数 750: 学习率 = 1e-4 * 0.75 = 7.5e-5
步数 1000: 学习率 = 1e-4 (达到目标值)
之后: 使用CosineAnnealing衰减
```

### 配置
在你的 `train_config.yaml` 中：
```yaml
training:
  warmup_steps: 1000  # 前1000步进行warmup
```

---

## 2. 梯度累积（Gradient Accumulation）是什么？

### 概念
**梯度累积** 是将多个小batch的梯度累积起来，然后一次性更新参数。这样可以模拟更大的batch size。

### 为什么需要梯度累积？
- **内存限制**：GPU显存不够时，无法使用大batch size
- **模拟大batch**：累积4个batch_size=8的梯度 = 实际batch_size=32的效果
- **更稳定的训练**：大batch通常训练更稳定

### 工作原理
```
正常训练（batch_size=32）:
  前向传播 → 计算loss → 反向传播 → 更新参数

梯度累积（accumulation_steps=4, batch_size=8）:
  Step 1: 前向传播(batch=8) → 计算loss → 反向传播 → 累积梯度（不更新）
  Step 2: 前向传播(batch=8) → 计算loss → 反向传播 → 累积梯度（不更新）
  Step 3: 前向传播(batch=8) → 计算loss → 反向传播 → 累积梯度（不更新）
  Step 4: 前向传播(batch=8) → 计算loss → 反向传播 → 累积梯度 → 更新参数（相当于batch=32）
```

### 配置
```yaml
training:
  gradient_accumulation_steps: 4  # 累积4步再更新
  batch_size: 8  # 每个GPU的batch size
  # 实际有效batch size = 8 * 4 = 32
```

---

## 3. 学习率怎么调整？

### 学习率调度策略

#### 1. Warmup + CosineAnnealing（推荐）
```python
# 前warmup_steps步：线性增长
# 之后：余弦衰减
学习率 = base_lr * (1 + cos(π * (step - warmup_steps) / (total_steps - warmup_steps))) / 2
```

#### 2. 学习率调整建议

**初始学习率选择**：
- 小模型：1e-3 到 1e-4
- 大模型（LLaMA）：1e-4 到 5e-5
- 冻结backbone时：可以稍大，如 1e-3

**不同组件不同学习率**（推荐）：
```python
# VGGT (冻结): 0
# Language Encoder: 1e-5 (小学习率，因为是预训练模型)
# Fusion + Action Head: 1e-4 (新模块，可以用大学习率)
```

**调整策略**：
- 如果loss不下降：降低学习率（除以2或10）
- 如果loss震荡：降低学习率
- 如果loss下降太慢：可以尝试增加学习率（但要小心）

### 配置示例
```yaml
training:
  learning_rate: 1e-4  # 基础学习率
  warmup_steps: 1000
  # 不同模块的学习率（在代码中实现）
  lr_multipliers:
    lang_encoder: 0.1  # 语言编码器用0.1倍学习率
    fusion: 1.0        # 融合模块用1.0倍
    action_head: 1.0   # 动作头用1.0倍
```

---

## 4. 验证频率（Validation Interval）是什么？

### 概念
**验证频率** 决定每隔多少训练步进行一次验证。

### 为什么重要？
- **监控过拟合**：定期检查模型在验证集上的表现
- **选择最佳模型**：保存验证loss最低的模型
- **调整训练策略**：如果验证loss不降，可能需要调整

### 如何设置？

**根据数据集大小**：
```yaml
# 小数据集（<10k样本）
val_interval: 500   # 每500步验证一次

# 中等数据集（10k-100k样本）
val_interval: 1000  # 每1000步验证一次（推荐）

# 大数据集（>100k样本）
val_interval: 2000  # 每2000步验证一次
```

**根据训练时间**：
- 如果验证很慢：减少验证频率（如5000步一次）
- 如果验证很快：可以增加频率（如500步一次）

### 当前配置
```yaml
training:
  val_interval: 1000  # 每1000个训练步验证一次
```

**注意**：验证会暂停训练，如果验证集很大，验证可能很慢。

---

## 5. 多GPU训练配置（8个GPU）

### 分布式训练方式

#### 方式1：使用 torchrun（推荐，PyTorch 2.0+）
```bash
# 使用8个GPU训练
torchrun --nproc_per_node=8 atlas/train.py --config atlas/configs/train_config.yaml
```

#### 方式2：使用 torch.distributed.launch（旧版本）
```bash
python -m torch.distributed.launch \
  --nproc_per_node=8 \
  --master_port=29500 \
  atlas/train.py --config atlas/configs/train_config.yaml
```

### 关键概念

**Local Rank vs Global Rank**：
- `local_rank`: 在当前机器上的GPU编号（0-7）
- `global_rank`: 所有机器上的全局编号（0-7，单机时等于local_rank）
- `world_size`: 总GPU数量（8）

**Batch Size分配**：
```yaml
# 配置文件中
data:
  batch_size: 8  # 这是每个GPU的batch size

# 实际总batch size = batch_size * num_gpus
# 8个GPU × 8 = 64 (每个GPU处理8个样本)
```

**注意事项**：
1. 每个GPU会处理 `batch_size` 个样本
2. 总的有效batch size = `batch_size × num_gpus × gradient_accumulation_steps`
3. 学习率可能需要根据总batch size调整（线性缩放规则）

### 学习率缩放规则
```
如果单GPU batch_size=8, lr=1e-4
那么8GPU batch_size=8, 总batch=64
建议学习率 = 1e-4 * 8 = 8e-4 (线性缩放)
或者 = 1e-4 * sqrt(8) = 2.8e-4 (平方根缩放，更保守)
```

---

## 总结

| 概念 | 作用 | 推荐值 |
|------|------|--------|
| **Warmup** | 稳定训练初期 | 1000步 |
| **梯度累积** | 模拟大batch | 2-4步 |
| **学习率** | 控制更新幅度 | 1e-4 (基础) |
| **验证频率** | 监控性能 | 1000步 |
| **多GPU** | 加速训练 | 8个GPU |
