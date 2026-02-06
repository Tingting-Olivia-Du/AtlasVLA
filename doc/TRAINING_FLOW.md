# AtlasVLA 训练流程详解

## 训练流程概览

```
┌─────────────────────────────────────────────────────────────┐
│                    AtlasVLA 训练流程                          │
└─────────────────────────────────────────────────────────────┘

1. 环境准备与检查
   ├─ 检查Python环境
   ├─ 检查GPU可用性
   ├─ 检查必要的Python包
   └─ 检查配置文件

2. 配置加载
   ├─ 读取YAML配置文件
   ├─ 解析模型配置
   ├─ 解析数据配置
   └─ 解析训练超参数

3. 分布式训练设置（多GPU模式）
   ├─ 初始化进程组
   ├─ 设置本地rank和全局rank
   └─ 配置NCCL后端

4. 模型初始化
   ├─ 加载VGGT预训练模型
   ├─ 加载LLaMA语言编码器
   ├─ 初始化融合模块
   ├─ 初始化动作预测头
   └─ 设置冻结/解冻策略

5. 数据加载
   ├─ 加载训练集（LIBERO）
   ├─ 加载验证集（可选）
   ├─ 创建DataLoader
   └─ 设置分布式采样器（多GPU）

6. Trainer初始化
   ├─ 设置优化器（AdamW）
   ├─ 设置学习率调度器（Warmup + Cosine）
   ├─ 设置损失函数
   ├─ 设置混合精度训练
   └─ 设置日志记录（Wandb）

7. 从Checkpoint恢复（可选）
   ├─ 加载模型权重
   ├─ 加载优化器状态
   ├─ 加载学习率调度器状态
   └─ 恢复训练步数和epoch

8. 训练循环
   │
   ├─ 对于每个Epoch:
   │   │
   │   ├─ 设置分布式采样器的epoch（多GPU）
   │   │
   │   ├─ 训练阶段（train_epoch）:
   │   │   ├─ 对于每个Batch:
   │   │   │   ├─ 1. 数据加载到GPU
   │   │   │   ├─ 2. 前向传播（混合精度）
   │   │   │   │   ├─ VGGT提取3D几何特征
   │   │   │   │   ├─ LLaMA编码语言指令
   │   │   │   │   ├─ 多模态融合（Cross-Attention）
   │   │   │   │   └─ 动作预测（6-DOF pose + gripper）
   │   │   │   ├─ 3. 计算损失（Pose Loss + Gripper Loss）
   │   │   │   ├─ 4. 反向传播（累积梯度）
   │   │   │   └─ 5. 每N步更新参数（梯度累积）
   │   │   │       ├─ 梯度裁剪
   │   │   │       ├─ 优化器更新
   │   │   │       ├─ 学习率调度
   │   │   │       └─ 更新全局步数
   │   │   │
   │   │   ├─ 定期日志记录（每log_interval步）
   │   │   │   ├─ 记录训练loss
   │   │   │   ├─ 记录学习率
   │   │   │   └─ 记录到Wandb（如果启用）
   │   │   │
   │   │   ├─ 定期验证（每val_interval步）
   │   │   │   ├─ 切换到评估模式
   │   │   │   ├─ 在验证集上评估
   │   │   │   ├─ 计算验证指标
   │   │   │   ├─ 保存最佳模型
   │   │   │   └─ 切换回训练模式
   │   │   │
   │   │   └─ 定期保存checkpoint（每save_interval步）
   │   │       ├─ 保存模型权重
   │   │       ├─ 保存优化器状态
   │   │       ├─ 保存调度器状态
   │   │       └─ 保存训练状态
   │   │
   │   └─ Epoch结束:
   │       ├─ 最终验证（如果有验证集）
   │       └─ 保存Epoch checkpoint
   │
   └─ 训练完成:
       ├─ 保存最终模型
       ├─ 清理分布式训练环境
       └─ 输出训练总结

9. 后处理
   ├─ 保存最佳模型
   ├─ 生成训练报告
   └─ 清理临时文件
```

## 详细步骤说明

### 步骤1: 环境准备与检查

**目的**: 确保训练环境正确配置

**检查项**:
- Python版本（>=3.8）
- PyTorch安装和CUDA可用性
- 必要的Python包（yaml, tqdm等）
- GPU数量和可用性
- 配置文件存在性

**代码位置**: `train.sh` 中的 `check_python()`, `check_gpu()`, `check_config()`

---

### 步骤2: 配置加载

**目的**: 从YAML文件加载所有训练配置

**加载内容**:
```yaml
model:
  - VGGT checkpoint路径
  - 语言编码器名称
  - 冻结策略
  - 模型维度配置

data:
  - 数据路径
  - Batch size
  - 图像尺寸
  - 数据增强设置

training:
  - 学习率
  - 训练轮数
  - Warmup步数
  - 梯度累积步数
  - 损失权重

checkpoint:
  - 保存目录
  - 恢复路径
```

**代码位置**: `train.py` 中的 `load_config()`

---

### 步骤3: 分布式训练设置（多GPU）

**目的**: 初始化多GPU分布式训练环境

**过程**:
1. 检测环境变量（RANK, WORLD_SIZE, LOCAL_RANK）
2. 初始化NCCL进程组
3. 设置每个进程的GPU设备
4. 返回分布式训练信息

**代码位置**: `train.py` 中的 `setup_distributed()`

---

### 步骤4: 模型初始化

**目的**: 创建并配置VGGTVLA模型

**组件**:
1. **VGGT Backbone**: 
   - 从预训练checkpoint加载
   - 提取3D几何特征（深度图、点云、相机姿态）
   - 可冻结或可训练

2. **Language Encoder (LLaMA)**:
   - 加载LLaMA 2编码器
   - 编码自然语言指令
   - 可冻结或可训练

3. **Geometry Feature Extractor**:
   - 从VGGT输出提取特征
   - 可选：PointNet编码点云
   - 可选：编码相机姿态

4. **Multimodal Fusion**:
   - Cross-Attention机制
   - 融合语言和几何特征

5. **Action Head**:
   - 预测6-DOF末端执行器姿态
   - 预测夹爪动作

**代码位置**: `train.py` 中的模型初始化部分

---

### 步骤5: 数据加载

**目的**: 准备训练和验证数据

**过程**:
1. 创建LIBERODataset实例
2. 加载episode元数据
3. 创建DataLoader（支持分布式采样）
4. 设置collate函数处理batch

**数据格式**:
- 输入: RGB图像 [B, S, 3, H, W] + 语言指令
- 输出: 动作 [B, 7] (6-DOF pose + gripper)

**代码位置**: `atlas/src/data/libero_dataset.py`

---

### 步骤6: Trainer初始化

**目的**: 设置训练所需的所有组件

**组件**:
1. **优化器 (AdamW)**:
   - 只优化可训练参数
   - 学习率和权重衰减

2. **学习率调度器**:
   - Warmup阶段：线性增长
   - 主训练阶段：余弦衰减

3. **损失函数 (VLALoss)**:
   - Pose Loss (SmoothL1)
   - Gripper Loss (L1)
   - 加权组合

4. **混合精度训练**:
   - GradScaler用于FP16训练
   - 加速训练并节省显存

5. **日志记录**:
   - Wandb集成（可选）
   - 控制台输出

**代码位置**: `atlas/src/training/trainer.py` 中的 `__init__()`

---

### 步骤7: 从Checkpoint恢复（可选）

**目的**: 从之前的checkpoint继续训练

**恢复内容**:
- 模型权重
- 优化器状态
- 学习率调度器状态
- 训练步数和epoch
- 最佳验证loss

**代码位置**: `trainer.py` 中的 `load_checkpoint()`

---

### 步骤8: 训练循环

这是训练的核心部分，包含多个嵌套循环：

#### 8.1 Epoch循环

**目的**: 遍历整个训练集多次

**过程**:
- 对于每个epoch，遍历所有训练数据
- 在每个epoch开始时设置分布式采样器的epoch（确保数据shuffle）

#### 8.2 Batch循环（训练阶段）

**对于每个batch**:

1. **数据准备**:
   ```python
   images = batch["images"].to(device)  # [B, S, 3, H, W]
   language_tasks = batch["language_task"]  # List[str]
   ```

2. **前向传播（混合精度）**:
   ```python
   with torch.cuda.amp.autocast():
       outputs = model(images, language_tasks)
       # outputs包含: pose [B, 6], gripper [B, 1]
   ```

3. **计算损失**:
   ```python
   loss_dict = criterion(
       predictions={"pose": outputs["pose"], "gripper": outputs["gripper"]},
       targets={"pose": batch["pose"], "gripper": batch["gripper"]}
   )
   ```

4. **反向传播（梯度累积）**:
   ```python
   # 损失除以累积步数（因为会累积多次）
   loss = loss_dict["total_loss"] / gradient_accumulation_steps
   scaler.scale(loss).backward()  # 累积梯度
   ```

5. **参数更新（每N步）**:
   ```python
   if (batch_idx + 1) % gradient_accumulation_steps == 0:
       # 梯度裁剪
       scaler.unscale_(optimizer)
       clip_grad_norm_(model.parameters(), max_norm=1.0)
       
       # 优化器更新
       scaler.step(optimizer)
       scaler.update()
       scheduler.step()  # 更新学习率
       optimizer.zero_grad()
       
       global_step += 1
   ```

#### 8.3 定期操作

**日志记录（每log_interval步）**:
- 记录训练loss（平均）
- 记录学习率
- 记录到Wandb（如果启用）

**验证（每val_interval步）**:
- 切换到eval模式
- 在验证集上评估
- 计算验证指标
- 保存最佳模型（如果验证loss更低）
- 切换回train模式

**保存Checkpoint（每save_interval步）**:
- 保存模型权重
- 保存优化器状态
- 保存调度器状态
- 保存训练状态

#### 8.4 Epoch结束

- 执行最终验证（如果有验证集）
- 保存Epoch checkpoint
- 更新epoch计数

**代码位置**: `trainer.py` 中的 `train_epoch()` 和 `train()`

---

### 步骤9: 后处理

**目的**: 训练完成后的清理和总结

**操作**:
1. 保存最终模型
2. 清理分布式训练环境
3. 输出训练总结（总时间、最佳loss等）
4. 关闭日志记录

**代码位置**: `train.py` 中的 `cleanup_distributed()`

---

## 关键概念

### 梯度累积

**目的**: 模拟更大的batch size

**工作原理**:
```
正常: batch_size=32 → 每次更新
累积: batch_size=8, accum=4 → 累积4次后更新（相当于batch=32）
```

**代码实现**:
- 损失除以累积步数
- 多次backward累积梯度
- 每N步才更新参数

### Warmup + Cosine调度

**Warmup阶段** (前warmup_steps步):
```
lr = base_lr * (current_step / warmup_steps)
```

**Cosine阶段** (warmup之后):
```
progress = (step - warmup_steps) / (total_steps - warmup_steps)
lr = base_lr * 0.5 * (1 + cos(π * progress))
```

### 分布式训练

**数据并行**:
- 每个GPU处理不同的数据batch
- 梯度在所有GPU间同步
- 有效batch size = batch_size × num_gpus

**学习率缩放**:
- 线性缩放: lr = base_lr × num_gpus
- 因为batch size增大了，学习率也需要相应增大

---

## 训练时间线示例

```
时间轴: 0 ──────────────────────────────────────────> 完成

步骤:
├─ 0s:     环境检查
├─ 10s:    加载配置
├─ 30s:    初始化模型（下载预训练权重可能需要更久）
├─ 60s:    加载数据
├─ 90s:    初始化Trainer
├─ 100s:   开始训练
│
│ 训练循环（每个epoch）:
│ ├─ Epoch 0:  [████████████] 100%
│ ├─ Epoch 1:  [████████████] 100%
│ ├─ ...
│ └─ Epoch 49: [████████████] 100%
│
└─ 完成:    保存最终模型，输出总结
```

---

## 监控训练

### 关键指标

1. **训练Loss**: 应该逐渐下降
2. **验证Loss**: 应该下降，如果上升可能过拟合
3. **学习率**: Warmup阶段上升，之后余弦衰减
4. **GPU利用率**: 应该接近100%
5. **内存使用**: 不应该OOM

### 检查点

- 每100步: 检查训练loss是否正常
- 每1000步: 检查验证loss
- 每5000步: 检查checkpoint是否保存
- 每个epoch: 检查整体进度

---

## 常见问题

### Q: 训练很慢怎么办？
A: 
- 检查GPU利用率（应该>90%）
- 减少数据加载的num_workers
- 确保使用混合精度训练
- 检查是否有数据加载瓶颈

### Q: 内存不足（OOM）？
A:
- 减少batch_size
- 增加gradient_accumulation_steps
- 冻结更多模块
- 减少图像尺寸

### Q: Loss不下降？
A:
- 检查学习率（可能需要降低）
- 检查数据质量
- 检查模型是否冻结过多
- 增加warmup_steps

### Q: 验证loss上升？
A:
- 可能过拟合，考虑：
  - 增加正则化（weight_decay）
  - 早停（early stopping）
  - 数据增强
  - 减少模型容量
