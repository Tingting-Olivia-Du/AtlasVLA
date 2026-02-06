# Wandb 快速开始指南

## 3步启用 Wandb

### 步骤1: 安装 Wandb

```bash
pip install wandb
```

### 步骤2: 登录 Wandb

```bash
wandb login
```

会提示输入API key，在 https://wandb.ai/settings 获取。

### 步骤3: 启用 Wandb

#### 方法A: 使用脚本参数（最简单）

```bash
./train.sh --wandb
```

#### 方法B: 修改配置文件

编辑 `atlas/configs/train_config.yaml`:

```yaml
wandb:
  enabled: true  # 改为 true
```

然后运行：
```bash
./train.sh
```

## 完整示例

```bash
# 8GPU训练，启用wandb，保存日志
./train.sh \
  --mode multi \
  --gpus 8 \
  --wandb \
  --log logs/train.log
```

## 查看结果

训练开始后，wandb会自动：
1. 打开浏览器显示训练页面
2. 或在终端显示URL，例如：
   ```
   https://wandb.ai/your-username/atlas-vla/runs/xxxxx
   ```

## 自定义配置

编辑 `atlas/configs/train_config.yaml`:

```yaml
wandb:
  enabled: true
  project: "atlas-vla"  # 项目名称
  entity: "my-team"  # 可选：团队名
  name: "experiment-1"  # 可选：实验名称
  tags: ["baseline"]  # 可选：标签
  notes: "First experiment"  # 可选：备注
```

## 常见问题

**Q: 如何获取API key？**
A: 访问 https://wandb.ai/settings → API keys

**Q: 不想每次登录？**
A: API key会保存在 `~/.netrc`，只需登录一次

**Q: 如何禁用wandb？**
A: 不使用 `--wandb` 参数，或设置 `enabled: false`

**Q: 多GPU训练会重复记录吗？**
A: 不会，只有rank 0会记录

## 详细文档

查看 `WANDB_SETUP.md` 获取完整文档。
