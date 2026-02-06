# Token传递问题调试

## 🔍 问题分析

从错误信息看：
```
401 Client Error: Repository not found for url: https://huggingface.co/meta-llama/Meta-Llama-3-8B/resolve/main/config.json
```

这说明token没有正确传递到config加载阶段。

## ✅ 已添加的调试信息

代码现在会输出：
1. Token是否从config读取
2. Token的前15个字符（用于验证）
3. Token长度
4. load_kwargs的内容
5. 环境变量状态

## 🛠️ 验证步骤

运行训练时，查看日志输出：

```bash
python atlas/train.py --config atlas/configs/train_config.yaml 2>&1 | grep -i token
```

应该看到：
- "HuggingFace token loaded and set in environment"
- "Token (first 15 chars): hf_EhHKcijCcxnL..."
- "Token length: 37"
- "Token added to load_kwargs"

## 🔧 如果仍然失败

### 1. 检查配置文件

```bash
python3 -c "import yaml; config = yaml.safe_load(open('atlas/configs/train_config.yaml')); print(config.get('huggingface', {}).get('token'))"
```

应该输出完整的token。

### 2. 手动设置环境变量测试

```bash
export HF_TOKEN="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"
python atlas/train.py --config atlas/configs/train_config.yaml
```

### 3. 使用huggingface-cli登录

```bash
huggingface-cli login
# 输入token: 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
```

### 4. 检查token权限

确保token有访问 `meta-llama/Meta-Llama-3-8B` 的权限。

## 📝 当前代码流程

1. **train.py**: 从config读取token → 设置环境变量 → 传递给VGGTVLA
2. **vggt_vla.py**: 接收token参数 → 设置环境变量 → 添加到load_kwargs → 传递给from_pretrained

## 🎯 下一步

运行训练并查看详细的调试输出，这将帮助我们定位token传递的问题。
