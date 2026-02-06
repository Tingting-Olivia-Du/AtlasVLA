# 显存优化指南

## 当前GPU状态分析

从你的 `nvidia-smi` 输出看：
- **GPU型号**: NVIDIA RTX 6000 Ada Generation（8个GPU）
- **总显存**: 49140 MiB ≈ **49 GB**
- **已使用**: ~20790 MiB ≈ **20 GB**
- **可用显存**: ~28350 MiB ≈ **28 GB**

**注意**: 虽然总显存是49GB，但已有约20GB被占用，实际可用约28GB。

## 显存需求估算

### 模型组件显存需求

#### 1. VGGT-1B Backbone（冻结时）
- **模型权重**: ~4 GB（如果冻结，不存储梯度）
- **激活值**: ~2-4 GB（取决于batch size和图像大小）

#### 2. LLaMA-2-7B Encoder（如果微调）
- **模型权重**: ~14 GB（FP16）
- **梯度**: ~14 GB（如果训练）
- **优化器状态**: ~28 GB（AdamW需要2倍参数空间）
- **激活值**: ~2-4 GB

#### 3. 融合模块 + 动作头
- **模型权重**: ~0.5 GB
- **梯度**: ~0.5 GB
- **优化器状态**: ~1 GB
- **激活值**: ~1-2 GB

#### 4. 数据（Batch Size = 8, Image Size = 518）
- **输入图像**: 8 × 2 × 3 × 518 × 518 × 4 bytes ≈ **50 MB**
- **中间特征**: ~2-4 GB

### 总显存需求估算

#### 场景1: VGGT冻结 + LLaMA冻结（推荐开始）
```
VGGT (冻结):        4 GB
LLaMA (冻结):       14 GB
融合+动作头:        2 GB
数据+激活值:        4 GB
系统开销:           2 GB
─────────────────────────
总计:              ~26 GB ✅ 可用
```

#### 场景2: VGGT冻结 + LLaMA微调
```
VGGT (冻结):        4 GB
LLaMA (训练):       14 GB (权重) + 14 GB (梯度) + 28 GB (优化器) = 56 GB
融合+动作头:        2 GB
数据+激活值:        4 GB
系统开销:           2 GB
─────────────────────────
总计:              ~68 GB ❌ 超出！
```

#### 场景3: 全部训练（End-to-End）
```
VGGT (训练):        4 GB + 4 GB (梯度) + 8 GB (优化器) = 16 GB
LLaMA (训练):       56 GB
融合+动作头:        2 GB
数据+激活值:        4 GB
系统开销:           2 GB
─────────────────────────
总计:              ~80 GB ❌ 超出！
```

## 优化策略

### 策略1: 减少Batch Size（最简单）

```yaml
data:
  batch_size: 4  # 从8减少到4，显存需求减半
```

**效果**: 
- 数据+激活值显存: 4 GB → 2 GB
- 总显存需求: ~26 GB → ~24 GB ✅

### 策略2: 使用梯度累积（推荐）

```yaml
data:
  batch_size: 4  # 每个GPU的batch size

training:
  gradient_accumulation_steps: 2  # 累积2步
  # 有效batch size = 4 × 8 GPUs × 2 = 64（和原来batch=8, 8GPU一样）
```

**效果**:
- 显存需求: ~24 GB
- 训练效果: 和batch=8相同

### 策略3: 冻结更多模块（最安全）

```yaml
model:
  freeze_vggt: true  # 冻结VGGT
  freeze_lang_encoder: true  # 也冻结LLaMA（推荐开始）
```

**效果**:
- 总显存需求: ~26 GB ✅
- 只训练融合模块和动作头

### 策略4: 使用混合精度训练（已启用）

代码已经启用了 `torch.cuda.amp`，可以：
- 减少约50%的激活值显存
- 减少约50%的梯度显存（如果使用FP16）

### 策略5: 减少图像尺寸（不推荐）

```yaml
data:
  image_size: 384  # 从518减少到384
```

**效果**: 显存减少约30%，但可能影响性能

### 策略6: 使用梯度检查点（Gradient Checkpointing）

如果LLaMA需要微调，可以使用梯度检查点：
- 显存减少约50%
- 训练速度降低约20%

## 推荐配置（28GB可用显存）

### 配置1: 保守配置（推荐开始）

```yaml
model:
  freeze_vggt: true
  freeze_lang_encoder: true  # 冻结LLaMA

data:
  batch_size: 8
  image_size: 518

training:
  gradient_accumulation_steps: 1
```

**显存需求**: ~26 GB ✅
**训练内容**: 只训练融合模块和动作头

### 配置2: 平衡配置

```yaml
model:
  freeze_vggt: true
  freeze_lang_encoder: false  # 微调LLaMA

data:
  batch_size: 2  # 减少batch size
  image_size: 518

training:
  gradient_accumulation_steps: 4  # 累积4步
  # 有效batch = 2 × 8 × 4 = 64
```

**显存需求**: ~28 GB ✅（接近极限）
**训练内容**: 微调LLaMA + 训练融合模块

### 配置3: 最大化利用（如果清理了其他占用）

如果能够释放更多显存（清理其他进程），可以：

```yaml
model:
  freeze_vggt: true
  freeze_lang_encoder: false

data:
  batch_size: 4
  image_size: 518

training:
  gradient_accumulation_steps: 2
  # 有效batch = 4 × 8 × 2 = 64
```

**显存需求**: ~30-32 GB（需要释放一些显存）

## 检查当前显存占用

### 1. 查看哪些进程占用显存

```bash
nvidia-smi
```

### 2. 查看详细进程信息

```bash
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv
```

### 3. 清理不需要的进程

```bash
# 找到占用显存的进程PID
nvidia-smi

# 终止进程（谨慎使用）
kill <PID>
```

## 训练前检查清单

- [ ] 确认可用显存 ≥ 26 GB（保守配置）
- [ ] 清理不必要的GPU进程
- [ ] 设置合适的batch size
- [ ] 启用梯度累积（如果需要）
- [ ] 确认混合精度训练已启用（已默认启用）
- [ ] 冻结不需要训练的模块

## 实际测试建议

### 步骤1: 先用最小配置测试

```yaml
model:
  freeze_vggt: true
  freeze_lang_encoder: true

data:
  batch_size: 4  # 小batch size测试
```

运行几个batch，检查显存使用：
```bash
watch -n 1 nvidia-smi
```

### 步骤2: 逐步增加

如果显存充足，逐步增加：
1. batch_size: 4 → 6 → 8
2. 解冻LLaMA（如果显存允许）
3. 增加gradient_accumulation_steps（如果需要更大有效batch）

### 步骤3: 监控显存使用

训练时监控：
```bash
# 另一个终端
watch -n 1 nvidia-smi
```

确保显存使用 < 90%，留一些余量避免OOM。

## 如果仍然OOM（Out of Memory）

### 选项1: 进一步减少batch size

```yaml
data:
  batch_size: 2  # 甚至1
```

### 选项2: 增加梯度累积

```yaml
training:
  gradient_accumulation_steps: 8  # 累积更多步
```

### 选项3: 使用更小的图像

```yaml
data:
  image_size: 384  # 或更小
```

### 选项4: 只训练部分模块

```yaml
model:
  freeze_vggt: true
  freeze_lang_encoder: true
  # 只训练融合和动作头
```

## 总结

**对于28GB可用显存**：

✅ **推荐配置**:
- VGGT冻结 + LLaMA冻结
- Batch size = 8
- 显存需求: ~26 GB

⚠️ **如果微调LLaMA**:
- VGGT冻结 + LLaMA微调
- Batch size = 2-4
- Gradient accumulation = 4-8
- 显存需求: ~28-30 GB（需要清理其他占用）

❌ **不推荐**:
- End-to-end训练（需要~80GB）
- 大batch size + LLaMA微调

## 快速配置模板

### 模板1: 安全配置（26GB）

```yaml
model:
  freeze_vggt: true
  freeze_lang_encoder: true

data:
  batch_size: 8
  image_size: 518

training:
  gradient_accumulation_steps: 1
```

### 模板2: 微调LLaMA（28GB）

```yaml
model:
  freeze_vggt: true
  freeze_lang_encoder: false

data:
  batch_size: 2
  image_size: 518

training:
  gradient_accumulation_steps: 4
```

### 模板3: 最大化利用（需要清理显存）

```yaml
model:
  freeze_vggt: true
  freeze_lang_encoder: false

data:
  batch_size: 4
  image_size: 518

training:
  gradient_accumulation_steps: 2
```
