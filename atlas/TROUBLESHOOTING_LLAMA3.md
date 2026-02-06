# Llama-3 模型加载故障排除

## 🔍 错误: KeyError: 'llama'

### 问题描述

加载 `meta-llama/Meta-Llama-3-8B` 时出现错误：
```
KeyError: 'llama'
```

### 可能原因

1. **transformers版本过旧** - 需要 >= 4.30.0
2. **Token未正确传递** - 需要确保token正确设置
3. **模型访问权限** - 需要确保有访问权限

### 解决方案

#### 1. 检查transformers版本

```bash
python3 -c "import transformers; print(transformers.__version__)"
```

如果版本 < 4.30.0，需要更新：
```bash
pip install --upgrade transformers>=4.30.0
```

#### 2. 验证Token和模型访问

```python
from transformers import AutoConfig
import os

token = "your_token_here"
os.environ['HF_TOKEN'] = token

# 测试加载config
config = AutoConfig.from_pretrained(
    "meta-llama/Meta-Llama-3-8B",
    token=token
)
print("Config loaded:", config.model_type)
```

#### 3. 使用huggingface-cli登录

```bash
huggingface-cli login
# 输入你的token
```

#### 4. 检查配置文件

确保 `atlas/configs/train_config.yaml` 中：
```yaml
model:
  lang_encoder_name: "meta-llama/Meta-Llama-3-8B"

huggingface:
  token: "your_token_here"
```

### 已验证的工作方式

以下方式已验证可以工作：

```python
from transformers import AutoModel
import os

token = "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"
os.environ['HF_TOKEN'] = token

model = AutoModel.from_pretrained(
    "meta-llama/Meta-Llama-3-8B",
    token=token,
    trust_remote_code=True
)
```

### 当前代码修复

代码已更新为：
1. ✅ 正确设置环境变量
2. ✅ 显式传递token参数
3. ✅ 添加config预加载用于调试
4. ✅ 改进错误处理

### 如果仍然失败

1. **更新transformers**:
   ```bash
   pip install --upgrade transformers
   ```

2. **清除缓存**:
   ```bash
   rm -rf ~/.cache/huggingface/hub/models--meta-llama--Meta-Llama-3-8B
   ```

3. **使用环境变量**:
   ```bash
   export HF_TOKEN="your_token"
   python atlas/train.py --config atlas/configs/train_config.yaml
   ```

4. **检查网络连接**:
   确保可以访问 huggingface.co

### 调试信息

代码现在会输出：
- Token是否设置
- Config加载状态
- 模型类型信息

查看日志文件获取详细错误信息：
```bash
tail -f logs/train_*.log
```
