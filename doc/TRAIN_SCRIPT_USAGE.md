# 训练脚本使用指南

## 快速开始

### 1. 基本使用（8个GPU）

```bash
# 使用默认配置（8个GPU）
./train.sh

# 或者明确指定
./train.sh --mode multi --gpus 8
```

### 2. 单GPU训练

```bash
./train.sh --mode single
```

### 3. 使用4个GPU

```bash
./train.sh --mode multi --gpus 4
```

### 4. 从checkpoint恢复训练

```bash
./train.sh --resume checkpoints/checkpoint_epoch_10.pt
```

### 5. 启用wandb记录

```bash
./train.sh --wandb
```

### 6. 保存训练日志

```bash
./train.sh --log logs/train_$(date +%Y%m%d_%H%M%S).log
```

### 7. 后台运行

```bash
./train.sh --background --log logs/train.log
```

## 完整示例

### 示例1: 标准8GPU训练（推荐）

```bash
# 使用8个GPU，启用wandb，保存日志
./train.sh \
  --mode multi \
  --gpus 8 \
  --wandb \
  --log logs/train_$(date +%Y%m%d_%H%M%S).log
```

### 示例2: 从checkpoint恢复

```bash
# 从第10个epoch的checkpoint恢复，使用4个GPU
./train.sh \
  --mode multi \
  --gpus 4 \
  --resume checkpoints/checkpoint_epoch_10.pt \
  --log logs/resume_train.log
```

### 示例3: 单GPU调试

```bash
# 单GPU训练，用于调试
./train.sh \
  --mode single \
  --log logs/debug.log
```

### 示例4: 后台训练

```bash
# 在后台运行训练，日志保存到文件
./train.sh \
  --mode multi \
  --gpus 8 \
  --background \
  --log logs/train_background.log

# 查看训练日志
tail -f logs/train_background.log

# 查看GPU使用情况（另一个终端）
watch -n 1 nvidia-smi
```

## 参数说明

| 参数 | 说明 | 默认值 | 示例 |
|------|------|--------|------|
| `--config` | 配置文件路径 | `atlas/configs/train_config.yaml` | `--config my_config.yaml` |
| `--mode` | 训练模式 | `multi` | `--mode single` 或 `--mode multi` |
| `--gpus` | GPU数量 | `8` | `--gpus 4` |
| `--resume` | Checkpoint路径 | 无 | `--resume checkpoints/best_model.pt` |
| `--wandb` | 启用wandb | `false` | `--wandb` |
| `--log` | 日志文件路径 | 无（输出到控制台） | `--log logs/train.log` |
| `--background` | 后台运行 | `false` | `--background` |
| `-h, --help` | 显示帮助 | - | `--help` |

## 脚本功能

### 自动检查

脚本会自动检查：
- ✅ Python环境
- ✅ PyTorch和CUDA
- ✅ GPU可用性
- ✅ 必要的Python包
- ✅ 配置文件存在性
- ✅ 数据路径配置

### 自动配置

脚本会自动：
- ✅ 根据GPU数量调整学习率（多GPU模式）
- ✅ 创建checkpoint和日志目录
- ✅ 设置分布式训练环境（多GPU模式）
- ✅ 显示训练信息摘要

### 错误处理

- ✅ 遇到错误立即退出（`set -e`）
- ✅ 使用未定义变量时报错（`set -u`）
- ✅ 检查命令是否存在
- ✅ 检查文件/目录是否存在

## 训练流程

脚本执行以下步骤：

1. **环境检查** - 检查Python、GPU、依赖包
2. **配置检查** - 验证配置文件和数据路径
3. **目录设置** - 创建checkpoint和日志目录
4. **Wandb设置** - 如果启用，检查wandb配置
5. **信息摘要** - 显示训练参数摘要
6. **开始训练** - 执行训练命令

## 监控训练

### 查看实时日志

如果使用 `--log` 参数：
```bash
tail -f logs/train.log
```

如果后台运行：
```bash
# 查看进程输出
ps aux | grep train.sh
tail -f /proc/<PID>/fd/1
```

### 查看GPU使用

```bash
# 实时监控
watch -n 1 nvidia-smi

# 或者
nvidia-smi -l 1
```

### 查看训练进度

训练过程中会显示：
- 当前epoch
- 当前step
- 训练loss
- 学习率
- 验证loss（如果启用验证）

## 常见问题

### Q: 脚本没有执行权限？

```bash
chmod +x train.sh
```

### Q: 找不到配置文件？

确保在项目根目录运行脚本，或者使用 `--config` 指定完整路径：
```bash
./train.sh --config /full/path/to/config.yaml
```

### Q: GPU不可用？

检查：
1. GPU驱动是否安装：`nvidia-smi`
2. PyTorch CUDA是否可用：`python -c "import torch; print(torch.cuda.is_available())"`
3. 如果使用Docker，确保GPU已映射

### Q: 内存不足（OOM）？

修改配置文件中的batch_size：
```yaml
data:
  batch_size: 4  # 从8减少到4
```

或者使用梯度累积：
```yaml
training:
  gradient_accumulation_steps: 4  # 累积4步
```

### Q: 如何停止训练？

如果在前台运行：`Ctrl+C`

如果在后台运行：
```bash
# 找到进程ID
ps aux | grep train.sh

# 终止进程
kill <PID>

# 或者强制终止
kill -9 <PID>
```

### Q: 如何查看训练历史？

检查checkpoint目录：
```bash
ls -lh checkpoints/
```

查看日志文件（如果使用了 `--log`）：
```bash
cat logs/train.log
```

## 最佳实践

1. **首次训练**: 使用单GPU测试配置是否正确
   ```bash
   ./train.sh --mode single
   ```

2. **正式训练**: 使用多GPU加速
   ```bash
   ./train.sh --mode multi --gpus 8 --wandb --log logs/train.log
   ```

3. **长时间训练**: 使用后台运行和日志
   ```bash
   ./train.sh --mode multi --gpus 8 --background --log logs/train.log
   ```

4. **恢复训练**: 始终指定checkpoint路径
   ```bash
   ./train.sh --resume checkpoints/best_model.pt
   ```

5. **实验管理**: 为每次实验创建不同的日志文件
   ```bash
   ./train.sh --log logs/exp_$(date +%Y%m%d_%H%M%S).log
   ```

## 与配置文件配合使用

脚本使用 `atlas/configs/train_config.yaml` 作为默认配置。你可以：

1. **修改默认配置**: 直接编辑 `atlas/configs/train_config.yaml`

2. **使用自定义配置**: 
   ```bash
   ./train.sh --config my_custom_config.yaml
   ```

3. **覆盖特定参数**: 修改配置文件中的对应项

## 输出说明

### 成功输出示例

```
[INFO] ==========================================
[INFO]     AtlasVLA 训练脚本
[INFO] ==========================================

[INFO] 训练配置：
  配置文件: atlas/configs/train_config.yaml
  训练模式: multi
  GPU数量: 8
  ...

[INFO] ==========================================
[INFO] 步骤 1/6: 环境检查
[INFO] ==========================================
[INFO] 检查Python环境...
[SUCCESS] Python版本: 3.9.0
[SUCCESS] PyTorch CUDA可用，版本: 11.8
...

[SUCCESS] ==========================================
[SUCCESS] 训练完成！
[SUCCESS] ==========================================
[SUCCESS] 总耗时: 2小时 30分钟 15秒
```

### 错误输出示例

```
[ERROR] Python未安装
或
[ERROR] 配置文件不存在: atlas/configs/train_config.yaml
或
[WARNING] 数据目录不存在: /path/to/libero/data
```

## 相关文档

- `TRAINING_FLOW.md` - 详细的训练流程说明
- `TRAINING_CONCEPTS.md` - 训练概念详解
- `MULTI_GPU_TRAINING.md` - 多GPU训练指南
- `QUICK_REFERENCE.md` - 快速参考
