# HuggingFace Token 配置指南

## 🔑 Token配置方式

### 方式1: 配置文件（推荐）✅

在 `atlas/configs/train_config.yaml` 中配置：

```yaml
# HuggingFace authentication
huggingface:
  token: "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"  # 你的token
```

**优点**: 
- 集中管理
- 版本控制友好（如果使用git，建议添加到.gitignore）

### 方式2: 环境变量

```bash
export HF_TOKEN="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"
# 或
export HUGGINGFACE_TOKEN="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"
```

### 方式3: HuggingFace CLI登录

```bash
huggingface-cli login
# 然后输入你的token
```

## 📋 当前配置

已更新配置文件使用：
- **模型**: `meta-llama/Meta-Llama-3-8B`
- **Token**: 已在配置文件中设置

## 🔒 安全建议

### 1. 不要提交token到Git

如果配置文件包含token，确保添加到 `.gitignore`:

```bash
# 添加到 .gitignore
echo "atlas/configs/train_config.yaml" >> .gitignore
```

或者使用环境变量方式，不将token写入配置文件。

### 2. 使用环境变量（生产环境推荐）

```yaml
# config文件中
huggingface:
  token: null  # 使用环境变量
```

然后在运行前设置：
```bash
export HF_TOKEN="your_token_here"
python atlas/train.py --config atlas/configs/train_config.yaml
```

## 🚀 使用方法

### 直接运行（token已在配置文件中）

```bash
python atlas/train.py --config atlas/configs/train_config.yaml
```

### 使用环境变量覆盖

```bash
export HF_TOKEN="your_token"
python atlas/train.py --config atlas/configs/train_config.yaml
```

## ✅ 验证Token

测试token是否有效：

```python
from transformers import AutoModel
import os

token = "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"
os.environ['HF_TOKEN'] = token

# 测试加载模型
model = AutoModel.from_pretrained(
    "meta-llama/Meta-Llama-3-8B",
    token=token
)
print("Token有效！")
```

## 🔍 故障排除

### 问题1: "Repository not found"

**原因**: Token无效或没有访问权限

**解决**:
1. 检查token是否正确
2. 确认你有访问该模型的权限
3. 在HuggingFace网站上验证token权限

### 问题2: "401 Client Error"

**原因**: Token认证失败

**解决**:
```bash
# 重新登录
huggingface-cli login

# 或检查环境变量
echo $HF_TOKEN
```

### 问题3: Token在配置文件中但不起作用

**检查**:
1. 配置文件路径是否正确
2. YAML格式是否正确（注意缩进）
3. Token字符串是否正确（没有多余空格）

## 📝 代码中的Token优先级

代码会按以下顺序查找token：

1. **函数参数** (`hf_token`参数)
2. **配置文件** (`huggingface.token`)
3. **环境变量** (`HF_TOKEN` 或 `HUGGINGFACE_TOKEN`)

## 🎯 当前模型配置

```yaml
model:
  lang_encoder_name: "meta-llama/Meta-Llama-3-8B"

huggingface:
  token: "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"
```

现在可以直接运行训练，token会自动用于模型和数据集加载！
