# 分布式训练调试指南

## 🔍 段错误（Segmentation Fault）常见原因

### 1. NCCL初始化问题

**症状**: 在 `dist.init_process_group()` 时崩溃

**解决方案**:
- ✅ 确保在初始化前设置CUDA设备: `torch.cuda.set_device(local_rank)`
- ✅ 添加超时设置避免死锁
- ✅ 检查NCCL环境变量

### 2. CUDA设备问题

**检查**:
```bash
# 检查CUDA是否可用
python3 -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.device_count())"

# 检查GPU可见性
echo $CUDA_VISIBLE_DEVICES

# 检查NCCL
python3 -c "import torch.distributed; print('NCCL available:', torch.distributed.is_nccl_available())"
```

### 3. 内存问题

**症状**: OOM或段错误

**解决方案**:
- 减少batch_size
- 减少num_workers
- 使用gradient checkpointing
- 检查GPU内存: `nvidia-smi`

### 4. 环境变量问题

**检查必要的环境变量**:
```bash
# torchrun会自动设置这些，但可以手动检查
echo $RANK
echo $WORLD_SIZE
echo $LOCAL_RANK
echo $MASTER_ADDR
echo $MASTER_PORT
```

## 🛠️ 调试步骤

### 步骤1: 单GPU测试

首先确保单GPU训练正常：

```bash
CUDA_VISIBLE_DEVICES=0 python3 atlas/train.py --config atlas/configs/train_config.yaml
```

### 步骤2: 双GPU测试

如果单GPU正常，测试双GPU：

```bash
CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 atlas/train.py --config atlas/configs/train_config.yaml
```

### 步骤3: 添加调试输出

在代码中添加更多print语句来定位问题：

```python
print(f"[Rank {rank}] Before model init")
model = VGGTVLA(...)
print(f"[Rank {rank}] After model init")
```

### 步骤4: 检查日志

查看日志文件中的错误信息：

```bash
tail -f logs/train_*.log
```

## 🔧 修复后的改进

### 1. 改进的分布式初始化

```python
# 在init_process_group之前设置设备
torch.cuda.set_device(local_rank)

# 添加超时避免死锁
dist.init_process_group(
    backend='nccl',
    init_method='env://',
    timeout=timedelta(seconds=1800)  # 30分钟
)
```

### 2. 错误处理

添加了try-catch块来捕获和记录错误：

```python
try:
    dist.init_process_group(...)
except Exception as e:
    logging.error(f"Error: {e}")
    raise
```

### 3. DDP优化

```python
model = DDP(
    model,
    device_ids=[local_rank],
    output_device=local_rank,
    find_unused_parameters=True,
    broadcast_buffers=True,
    gradient_as_bucket_view=True  # 更节省内存
)
```

## 📋 常见错误和解决方案

### 错误1: "NCCL error: unhandled system error"

**原因**: NCCL通信问题

**解决**:
```bash
# 设置NCCL调试
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=ALL

# 或使用TCP后端（如果NCCL有问题）
# 修改代码: backend='gloo'  # 但gloo不支持CUDA，只用于调试
```

### 错误2: "CUDA out of memory"

**解决**:
- 减少batch_size
- 减少num_workers
- 使用gradient accumulation
- 检查是否有其他进程占用GPU

### 错误3: "Address already in use"

**原因**: MASTER_PORT被占用

**解决**:
```bash
# 使用不同的端口
export MASTER_PORT=29501

# 或让torchrun自动选择
# torchrun会自动处理端口冲突
```

## 🚀 推荐的训练命令

### 单GPU（调试用）

```bash
CUDA_VISIBLE_DEVICES=0 python3 atlas/train.py --config atlas/configs/train_config.yaml
```

### 多GPU（生产用）

```bash
# 使用train.sh脚本（推荐）
CUDA_VISIBLE_DEVICES=0,1,2,3 ./atlas/scripts/train.sh

# 或直接使用torchrun
torchrun --nproc_per_node=4 atlas/train.py --config atlas/configs/train_config.yaml
```

### 带调试信息

```bash
# 启用NCCL调试
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=ALL

# 运行训练
torchrun --nproc_per_node=4 atlas/train.py --config atlas/configs/train_config.yaml
```

## 📊 监控训练

### 实时监控GPU

```bash
# 另一个终端
watch -n 1 nvidia-smi
```

### 查看进程

```bash
# 查看训练进程
ps aux | grep train.py

# 查看GPU进程
nvidia-smi
```

## 💡 最佳实践

1. **先单GPU测试**: 确保代码逻辑正确
2. **逐步增加GPU**: 从2个GPU开始，逐步增加到8个
3. **监控资源**: 使用nvidia-smi监控GPU使用
4. **保存日志**: 启用日志保存以便调试
5. **使用wandb**: 可视化训练过程

## 🔗 相关文档

- PyTorch分布式训练: https://pytorch.org/tutorials/intermediate/ddp_tutorial.html
- NCCL文档: https://docs.nvidia.com/deeplearning/nccl/
- 训练脚本使用: `atlas/scripts/train.sh`
