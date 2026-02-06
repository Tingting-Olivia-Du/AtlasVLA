# 指定GPU使用指南

## 功能说明

现在你可以通过 `--gpu-ids` 参数指定使用哪些GPU，而不是只能指定GPU数量。

## 使用方法

### 1. 指定连续的GPU（例如：使用GPU 0,1,2,3）

```bash
./train.sh --mode multi --gpu-ids 0,1,2,3
```

### 2. 指定不连续的GPU（例如：使用GPU 0,2,4,6，跳过奇数GPU）

```bash
./train.sh --mode multi --gpu-ids 0,2,4,6
```

### 3. 只使用前4个GPU

```bash
./train.sh --mode multi --gpu-ids 0,1,2,3
```

### 4. 使用特定的GPU（例如：只使用GPU 1和3）

```bash
./train.sh --mode multi --gpu-ids 1,3
```

### 5. 单GPU模式下指定GPU

```bash
# 使用GPU 2进行单GPU训练
./train.sh --mode single --gpu-ids 2
```

## 工作原理

### CUDA_VISIBLE_DEVICES

脚本使用 `CUDA_VISIBLE_DEVICES` 环境变量来限制可见的GPU：

```bash
# 当你指定 --gpu-ids 0,2,4,6 时
export CUDA_VISIBLE_DEVICES=0,2,4,6

# PyTorch会将这些GPU重新编号为 0,1,2,3
# 物理GPU 0 → 逻辑GPU 0
# 物理GPU 2 → 逻辑GPU 1
# 物理GPU 4 → 逻辑GPU 2
# 物理GPU 6 → 逻辑GPU 3
```

### 自动验证

脚本会自动：
- ✅ 验证指定的GPU ID是否有效
- ✅ 检查GPU是否存在
- ✅ 显示指定GPU的详细信息（名称、显存等）
- ✅ 自动计算GPU数量

## 使用场景

### 场景1: 避免使用被占用的GPU

```bash
# 假设GPU 0和1被其他任务占用，使用GPU 2,3,4,5
./train.sh --mode multi --gpu-ids 2,3,4,5
```

### 场景2: 使用特定性能的GPU

```bash
# 假设你的服务器有不同型号的GPU，只使用高性能的GPU
./train.sh --mode multi --gpu-ids 0,1,2,3  # 使用前4个高性能GPU
```

### 场景3: 测试特定GPU配置

```bash
# 测试使用不同GPU组合的效果
./train.sh --mode multi --gpu-ids 0,1 --log logs/test_2gpu.log
./train.sh --mode multi --gpu-ids 0,1,2,3 --log logs/test_4gpu.log
```

### 场景4: 多用户共享服务器

```bash
# 用户A使用GPU 0,1,2,3
./train.sh --mode multi --gpu-ids 0,1,2,3

# 用户B使用GPU 4,5,6,7
./train.sh --mode multi --gpu-ids 4,5,6,7
```

## 完整示例

### 示例1: 使用GPU 0,2,4,6进行8GPU训练

```bash
./train.sh \
  --mode multi \
  --gpu-ids 0,2,4,6 \
  --wandb \
  --log logs/train_gpu_0246.log
```

### 示例2: 使用GPU 1,3,5,7，从checkpoint恢复

```bash
./train.sh \
  --mode multi \
  --gpu-ids 1,3,5,7 \
  --resume checkpoints/checkpoint_epoch_10.pt \
  --log logs/resume_gpu_1357.log
```

### 示例3: 单GPU训练，使用GPU 5

```bash
./train.sh \
  --mode single \
  --gpu-ids 5 \
  --log logs/single_gpu5.log
```

## 参数优先级

如果同时指定了 `--gpus` 和 `--gpu-ids`：

```bash
# --gpu-ids 会覆盖 --gpus
./train.sh --mode multi --gpus 8 --gpu-ids 0,1,2,3
# 实际会使用4个GPU（0,1,2,3），而不是8个
```

**推荐**: 如果指定了 `--gpu-ids`，就不需要再指定 `--gpus`，脚本会自动计算GPU数量。

## 查看GPU信息

在运行训练前，脚本会自动显示：

1. **所有GPU信息**（如果未指定GPU IDs）:
   ```
   [INFO] 所有GPU信息：
   0, NVIDIA A100-SXM4-40GB, 40536 MiB, 1024 MiB
   1, NVIDIA A100-SXM4-40GB, 40536 MiB, 2048 MiB
   ...
   ```

2. **指定GPU信息**（如果指定了GPU IDs）:
   ```
   [INFO] 指定GPU的详细信息：
   0, NVIDIA A100-SXM4-40GB, 40536 MiB, 1024 MiB
   2, NVIDIA A100-SXM4-40GB, 40536 MiB, 512 MiB
   4, NVIDIA A100-SXM4-40GB, 40536 MiB, 2048 MiB
   6, NVIDIA A100-SXM4-40GB, 40536 MiB, 1024 MiB
   ```

## 验证GPU选择

训练开始后，你可以通过以下方式验证：

### 1. 查看nvidia-smi

```bash
# 在另一个终端运行
watch -n 1 nvidia-smi
```

你应该只看到指定的GPU有活动。

### 2. 查看训练日志

训练日志会显示：
```
[INFO] 指定使用的GPU: 0,2,4,6
[SUCCESS] 将使用 4 个GPU: 0,2,4,6
[INFO] 设置 CUDA_VISIBLE_DEVICES=0,2,4,6
```

### 3. 在代码中验证

PyTorch代码中：
```python
import torch
print(torch.cuda.device_count())  # 应该等于你指定的GPU数量
print(torch.cuda.get_device_name(0))  # 显示第一个可见GPU的名称
```

## 常见问题

### Q: 如何查看所有可用的GPU？

```bash
nvidia-smi --query-gpu=index,name,memory.total,memory.used --format=csv
```

### Q: 指定的GPU ID无效怎么办？

脚本会自动检查并报错：
```
[ERROR] GPU ID 8 无效（可用范围: 0-7）
```

### Q: 可以混合使用不同型号的GPU吗？

可以，但不推荐。不同型号的GPU可能有不同的性能，可能导致训练不稳定。

### Q: 指定GPU后，GPU编号会改变吗？

是的。PyTorch会将可见的GPU重新编号为0,1,2,3...，但这是正常的，不影响训练。

### Q: 如何知道哪些GPU被占用了？

```bash
# 查看GPU使用情况
nvidia-smi

# 查看特定GPU
nvidia-smi -i 0  # 查看GPU 0
```

### Q: 可以动态改变使用的GPU吗？

不可以。必须在训练开始前指定。如果需要改变，需要停止训练并重新启动。

## 最佳实践

1. **先查看GPU状态**:
   ```bash
   nvidia-smi
   ```

2. **选择空闲的GPU**:
   ```bash
   # 使用显存使用率低的GPU
   ./train.sh --mode multi --gpu-ids 2,3,4,5
   ```

3. **记录使用的GPU**:
   ```bash
   # 在日志文件名中包含GPU信息
   ./train.sh --mode multi --gpu-ids 0,1,2,3 --log logs/train_gpu0123.log
   ```

4. **多用户环境**:
   - 与团队成员协调GPU使用
   - 使用 `nvidia-smi` 查看GPU占用情况
   - 在共享服务器上使用特定的GPU范围

## 与配置文件的关系

`--gpu-ids` 参数只影响运行时可见的GPU，不影响配置文件中的其他设置。

配置文件中的 `batch_size` 仍然是每个GPU的batch size：
```yaml
data:
  batch_size: 8  # 每个GPU的batch size
```

如果使用 `--gpu-ids 0,1,2,3`（4个GPU），总batch size = 8 × 4 = 32。
