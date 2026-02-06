# Wandb 设置和使用指南

## 什么是 Wandb？

Weights & Biases (wandb) 是一个机器学习实验跟踪工具，可以帮助你：
- 📊 可视化训练过程（loss、学习率等）
- 📈 比较不同实验
- 💾 自动保存代码和配置
- 🔍 调试训练问题
- 📝 记录实验笔记

## 快速开始

### 1. 安装 Wandb

```bash
# 方法1: 使用pip安装
pip install wandb

# 方法2: 使用项目的可选依赖
pip install -e ".[wandb]"
```

### 2. 登录 Wandb

```bash
# 方法1: 交互式登录（推荐）
wandb login

# 方法2: 使用API Key
export WANDB_API_KEY=your_api_key_here
wandb login

# 方法3: 在代码中设置（不推荐，安全性较低）
export WANDB_API_KEY=your_api_key_here
```

**获取API Key**:
1. 访问 https://wandb.ai/
2. 注册/登录账号
3. 进入 Settings → API keys
4. 复制你的API key

### 3. 启用 Wandb

#### 方法1: 使用训练脚本（推荐）

```bash
# 使用 --wandb 参数
./train.sh --wandb

# 完整示例
./train.sh \
  --mode multi \
  --gpus 8 \
  --wandb \
  --log logs/train.log
```

#### 方法2: 修改配置文件

编辑 `atlas/configs/train_config.yaml`:

```yaml
wandb:
  enabled: true  # 改为 true
  project: "atlas-vla"
  entity: "your-username"  # 可选：你的wandb用户名
  name: "experiment-1"  # 可选：实验名称
  tags: ["baseline", "vggt-frozen"]  # 可选：标签
  notes: "First experiment with frozen VGGT"  # 可选：备注
```

然后运行：
```bash
./train.sh
```

## 配置选项详解

### 基本配置

```yaml
wandb:
  enabled: true  # 是否启用wandb
  project: "atlas-vla"  # 项目名称（会在wandb网站创建/使用这个项目）
  entity: null  # 用户名或团队名（留空使用默认）
```

### 高级配置

```yaml
wandb:
  enabled: true
  project: "atlas-vla"
  entity: "my-team"  # 团队名称（如果有）
  
  # 实验名称（留空自动生成）
  name: "vggt-frozen-baseline"
  
  # 标签（用于分类和组织实验）
  tags: 
    - "baseline"
    - "vggt-frozen"
    - "libero-dataset"
  
  # 实验备注/描述
  notes: "First experiment with frozen VGGT backbone, training fusion and action head only"
  
  # 是否保存代码到wandb（推荐开启）
  save_code: true
  
  # 如果实验名称已存在，如何处理
  # "allow": 允许（创建新运行）
  # "must": 必须存在（恢复运行）
  # "never": 不允许（报错）
  # "auto": 自动处理
  resume: "allow"
```

## 使用示例

### 示例1: 基本使用

```bash
# 启用wandb，使用默认设置
./train.sh --wandb
```

这会在wandb网站创建/使用项目 `atlas-vla`，实验名称自动生成。

### 示例2: 自定义项目名称

修改配置文件：
```yaml
wandb:
  enabled: true
  project: "atlas-vla-libero"  # 自定义项目名
```

### 示例3: 添加实验标签和备注

修改配置文件：
```yaml
wandb:
  enabled: true
  project: "atlas-vla"
  name: "experiment-vggt-frozen"
  tags: 
    - "baseline"
    - "vggt-frozen"
    - "batch-size-8"
  notes: "Testing with frozen VGGT, batch size 8, learning rate 1e-4"
```

### 示例4: 团队协作

```yaml
wandb:
  enabled: true
  project: "atlas-vla"
  entity: "my-research-team"  # 团队名称
  name: "experiment-1"
```

## 查看实验结果

### 1. 在浏览器中查看

训练开始后，wandb会自动打开浏览器，或显示一个URL：
```
https://wandb.ai/your-username/atlas-vla/runs/xxxxx
```

### 2. 命令行查看

```bash
# 查看最近的运行
wandb status

# 查看所有项目
wandb projects

# 查看特定项目的运行
wandb runs atlas-vla
```

### 3. 在wandb网站查看

访问 https://wandb.ai/ 登录后可以看到：
- 📊 **实时图表**: Loss、学习率、验证指标等
- 📋 **系统指标**: GPU使用率、内存使用等
- 📝 **日志**: 训练日志输出
- 💾 **文件**: 保存的checkpoint、代码等
- 🔍 **配置**: 所有超参数

## 记录的内容

Wandb会自动记录：

### 训练指标
- `train/loss` - 训练总损失
- `train/pose_loss` - 姿态损失
- `train/gripper_loss` - 夹爪损失
- `train/lr` - 学习率
- `train/epoch` - 当前epoch
- `train/step` - 当前步数

### 验证指标
- `val/loss` - 验证总损失
- `val/pose_loss` - 验证姿态损失
- `val/gripper_loss` - 验证夹爪损失
- `val/pose_l2_error` - 姿态L2误差
- `val/pose_l1_error` - 姿态L1误差
- `val/gripper_error` - 夹爪误差

### 系统指标
- GPU使用率
- GPU内存使用
- CPU使用率
- 内存使用

### 配置信息
- 所有超参数（从配置文件读取）
- 模型参数数量
- 训练配置

## 比较实验

### 在wandb网站

1. 选择多个运行（实验）
2. 点击 "Compare" 按钮
3. 查看对比图表和表格

### 使用Python API

```python
import wandb

# 初始化API
api = wandb.Api()

# 获取项目
runs = api.runs("your-username/atlas-vla")

# 比较运行
for run in runs:
    print(f"Run: {run.name}")
    print(f"Final Loss: {run.summary.get('val/loss')}")
    print(f"Config: {run.config}")
```

## 恢复实验

### 从wandb恢复checkpoint

如果启用了 `save_code: true`，代码会被保存到wandb。可以：

1. 在wandb网站下载checkpoint
2. 或使用wandb API下载

```python
import wandb

run = wandb.init(id="run-id", resume="must")
# 继续训练...
```

## 常见问题

### Q: wandb登录失败？

**A**: 检查：
1. API key是否正确
2. 网络连接是否正常
3. 是否在代理环境中（需要设置代理）

```bash
# 设置代理（如果需要）
export https_proxy=http://proxy.example.com:8080
export http_proxy=http://proxy.example.com:8080
```

### Q: 如何离线使用wandb？

**A**: 设置离线模式：

```bash
export WANDB_MODE=offline
./train.sh --wandb
```

训练结束后，同步到wandb：
```bash
wandb sync wandb/offline-run-xxxxx
```

### Q: 如何禁用wandb？

**A**: 

方法1: 不使用 `--wandb` 参数
```bash
./train.sh  # 不启用wandb
```

方法2: 修改配置文件
```yaml
wandb:
  enabled: false
```

### Q: 多GPU训练时wandb会记录多次吗？

**A**: 不会。只有rank 0（主进程）会记录到wandb，避免重复记录。

### Q: 如何更改wandb项目名称？

**A**: 修改配置文件中的 `project` 字段：

```yaml
wandb:
  enabled: true
  project: "my-new-project-name"
```

### Q: 如何添加自定义指标？

**A**: 在训练代码中添加：

```python
# 在trainer.py中
if self.use_wandb:
    wandb.log({
        "custom/metric1": value1,
        "custom/metric2": value2,
    }, step=self.global_step)
```

### Q: wandb占用太多空间？

**A**: 可以限制保存的内容：

```yaml
wandb:
  enabled: true
  save_code: false  # 不保存代码
```

或在代码中设置：
```python
wandb.init(..., settings=wandb.Settings(_disable_stats=True))
```

## 最佳实践

1. **为每次实验命名**: 使用有意义的名称
   ```yaml
   name: "vggt-frozen-lr1e4-bs8"
   ```

2. **使用标签分类**: 方便后续查找和比较
   ```yaml
   tags: ["baseline", "vggt-frozen", "libero"]
   ```

3. **添加实验备注**: 记录实验目的和关键信息
   ```yaml
   notes: "Testing different learning rates, baseline experiment"
   ```

4. **定期检查**: 训练过程中定期查看wandb网站，及时发现问题

5. **保存重要checkpoint**: 虽然wandb可以保存，但重要的checkpoint建议本地也保存一份

6. **使用团队功能**: 如果是团队协作，使用 `entity` 指定团队名称

## 高级功能

### 1. 超参数扫描（Sweep）

```python
# 创建sweep配置
sweep_config = {
    "method": "grid",
    "parameters": {
        "learning_rate": {"values": [1e-4, 5e-5, 1e-5]},
        "batch_size": {"values": [4, 8, 16]},
    }
}

sweep_id = wandb.sweep(sweep_config, project="atlas-vla")
wandb.agent(sweep_id, train_function)
```

### 2. 自定义可视化

```python
import wandb

# 记录图像
wandb.log({"predictions": wandb.Image(image)})

# 记录表格
wandb.log({"table": wandb.Table(data=...)}))

# 记录视频
wandb.log({"video": wandb.Video(video_path)})
```

### 3. 报告生成

在wandb网站创建报告，汇总多个实验的结果。

## 相关资源

- Wandb官方文档: https://docs.wandb.ai/
- Wandb Python API: https://docs.wandb.ai/ref/python/api
- 示例项目: https://wandb.ai/examples

## 总结

使用wandb可以大大提升实验管理的效率：

✅ **自动记录**: 无需手动记录指标  
✅ **可视化**: 直观的图表和对比  
✅ **协作**: 团队共享实验结果  
✅ **可复现**: 自动保存代码和配置  
✅ **调试**: 快速定位训练问题  

开始使用wandb，让你的训练过程更加专业和高效！
