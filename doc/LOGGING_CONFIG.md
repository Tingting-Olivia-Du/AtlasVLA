# 日志配置说明

## 📋 配置选项

在 `atlas/configs/train_config.yaml` 中可以配置日志保存：

```yaml
# Logging configuration
logging:
  # 是否保存训练日志到文件
  save_to_file: true
  # 日志文件保存路径（null = 自动生成，格式: logs/train_YYYYMMDD_HHMMSS.log）
  log_file: null  # 例如: "logs/train.log" 或 null 自动生成
  # 日志目录（如果log_file为null，会在此目录下自动生成日志文件）
  log_dir: "./logs"
  # 是否同时输出到控制台
  console_output: true
```

## 🎯 使用方式

### 方式1: 自动生成日志文件名（推荐）

```yaml
logging:
  save_to_file: true
  log_file: null  # 自动生成，例如: logs/train_20260205_051930.log
  log_dir: "./logs"
  console_output: true
```

**优点**: 每次训练都会生成新的日志文件，不会覆盖之前的日志

### 方式2: 指定固定日志文件名

```yaml
logging:
  save_to_file: true
  log_file: "logs/train.log"  # 固定文件名
  console_output: true
```

**注意**: 如果文件已存在，日志会追加到文件末尾

### 方式3: 禁用文件日志（仅控制台输出）

```yaml
logging:
  save_to_file: false
  console_output: true
```

### 方式4: 仅文件日志（不输出到控制台）

```yaml
logging:
  save_to_file: true
  log_file: "logs/train.log"
  console_output: false
```

## 📝 日志内容

日志文件会记录：

- ✅ 训练配置信息（GPU、分布式设置、数据源等）
- ✅ 模型信息（参数量、可训练参数等）
- ✅ 训练进度（每个log_interval的loss、pose_loss、gripper_loss）
- ✅ 验证结果（每个epoch的验证指标）
- ✅ Checkpoint保存信息
- ✅ 训练完成信息

## 🔍 查看日志

### 实时查看

```bash
# 查看最新日志
tail -f logs/train_20260205_051930.log

# 查看最后100行
tail -n 100 logs/train_20260205_051930.log
```

### 搜索日志

```bash
# 搜索错误
grep -i error logs/train_*.log

# 搜索特定step的loss
grep "Step 1000" logs/train_*.log

# 搜索checkpoint保存
grep "Saved checkpoint" logs/train_*.log
```

## 🚀 分布式训练

在分布式训练中，**只有rank 0进程会保存日志文件**，避免多个进程同时写入造成冲突。

## 📊 日志格式

```
2026-02-05 05:19:30 - INFO - Training setup:
2026-02-05 05:19:30 - INFO -   Distributed: False
2026-02-05 05:19:30 - INFO -   Device: cuda
2026-02-05 05:19:30 - INFO -   Log file: logs/train_20260205_051930.log
2026-02-05 05:19:30 - INFO - Initializing model...
2026-02-05 05:19:31 - INFO - Model initialized. Total parameters: 1,234,567,890
2026-02-05 05:19:31 - INFO - Trainable parameters: 123,456,789
2026-02-05 05:19:31 - INFO - Step 100: Loss=0.1234, Pose=0.1000, Gripper=0.0234
...
```

## 💡 最佳实践

1. **使用自动生成文件名**: 便于追踪不同训练实验
2. **保留日志目录**: 定期清理旧日志，但保留重要实验的日志
3. **结合wandb**: 文件日志 + wandb 可以更好地追踪实验

## 🔧 故障排除

### 问题1: 日志文件没有创建

检查：
- `save_to_file` 是否设置为 `true`
- `log_dir` 目录是否有写权限
- 磁盘空间是否充足

### 问题2: 日志文件为空

检查：
- 训练是否正常启动
- 是否有错误导致训练提前退出
- 查看控制台输出是否有错误信息

### 问题3: 日志文件太大

可以：
- 减少 `log_interval`（减少日志频率）
- 定期清理旧日志文件
- 使用日志轮转工具（如logrotate）
