# LIBERO Eval 失败原因分析报告

## 📋 执行摘要

所有 eval 任务失败的**根本原因**：**模型权重加载失败导致模型无法进行有效推理**

---

## 🔴 核心问题：语言模型配置不匹配

### 问题描述

当运行 eval 时，模型权重加载失败，产生大量维度不匹配错误：

```
size mismatch for language_encoder.language_model.layers.0.self_attn.q_proj.weight:
  copying a param with shape torch.Size([2048, 1024]) from checkpoint,
  the shape in current model is torch.Size([896, 896]).
```

### 根本原因

**Checkpoint 使用的是 Qwen3-0.6B，但当前环境中 Qwen3 不可用，代码 fallback 到 Qwen2-0.5B**

| 模型 | Hidden Size | 状态 |
|------|-------------|------|
| Qwen3-0.6B (Checkpoint中) | 1024 | ❌ 不可用（Transformers不支持） |
| Qwen2-0.5B (Fallback) | 896 | ✅ 可用（当前环境中使用） |

### 为什么会这样？

1. **Checkpoint 保存时**（训练时）：使用了 Qwen3-0.6B
   - 语言模型 hidden_size: 1024
   - 投影层输入维度: 1024
   - 权重形状都是基于 1024 的

2. **Checkpoint 加载时**（eval 时）：
   - 尝试加载 Qwen3-0.6B，失败（提示"qwen3 architecture not recognized"）
   - 代码 fallback 到 Qwen2-0.5B
   - Qwen2-0.5B hidden_size: 896
   - 创建的投影层输入维度: 896
   - **权重形状不匹配** → `load_state_dict(strict=True)` 失败

### 错误日志位置

文件：`/workspace/02042026_tingting/AtlasVLA/vggt_vla/eval/debug_eval.py`
输出：`/root/.claude/projects/-workspace-02042026-tingting-AtlasVLA/f99044e9-69d9-4f87-a1ac-946470eddc40/tool-results/be0da36.txt`

---

## 📊 配置信息

### Checkpoint 中的配置
```
language_model: Qwen/Qwen3-0.6B-Base
hidden_dim: 768
vggt.embed_dim: 768
language.output_dim: 768
```

### 当前代码中的配置（fallback后）
```
language_model: Qwen/Qwen2-0.5B  (fallback)
language_hidden_size: 896 (实际的Qwen2-0.5B)
target output_dim: 768
```

### 投影层尺寸不匹配
- Checkpoint中的投影层：`Linear(1024 → 768)`
- 当前代码中创建的投影层：`Linear(896 → 768)`

---

## 🔧 解决方案

### 方案 1：更新 Transformers 版本（推荐）
如果 Qwen3 是新的模型，需要升级 Transformers 来支持它。

```bash
pip install --upgrade transformers
```

**优点**：
- 恢复原始训练配置
- 权重完全匹配
- 模型性能最佳

**可能的风险**：
- 可能影响其他依赖

### 方案 2：使用兼容 Checkpoint（快速修复）
重新用 Qwen2-0.5B 训练模型，或转换 checkpoint。

**流程**：
1. 修改 checkpoint 中的配置，将语言模型改为 Qwen2-0.5B
2. 使用特殊的加载逻辑处理维度不匹配

**实现代码**（在 `eval_vla_libero.py` 中）：
```python
# 修改 load_model_and_config 函数
try:
    model.load_state_dict(state_dict, strict=True)
except RuntimeError as e:
    if 'size mismatch' in str(e) and 'language' in str(e):
        # 尝试使用 strict=False 加载，然后手动处理
        print("Language model dimension mismatch detected.")
        print("Loading with strict=False and re-initializing mismatched layers...")
        model.load_state_dict(state_dict, strict=False)
        # 重新初始化被忽略的层，但这可能影响性能
```

### 方案 3：检查是否有 Qwen3 的轻量级替代品

Qwen3-0.6B 如果不可用，可以寻找其他类似规模的模型：
- `meta-llama/Llama-2-7b` (大一点)
- `mistralai/Mistral-7B` (大一点)
- `Qwen/Qwen2-1.5B` (大一点)

---

## 📝 相关代码位置

### 语言编码器（有 fallback 逻辑）
文件：[vggt_vla/models/language_encoder.py:30-41](vggt_vla/models/language_encoder.py#L30-L41)

```python
try:
    self.language_model = AutoModel.from_pretrained(
        config.model_name,
        trust_remote_code=True
    )
except Exception as e:
    print(f"Warning: Could not load model from {config.model_name}: {e}")
    print("Falling back to Qwen2-0.5B...")
    self.language_model = AutoModel.from_pretrained(
        "Qwen/Qwen2-0.5B",
        trust_remote_code=True
    )
```

### Checkpoint 加载逻辑（strict=True）
文件：[vggt_vla/eval/eval_vla_libero.py:148](vggt_vla/eval/eval_vla_libero.py#L148)

```python
model.load_state_dict(state_dict, strict=True)  # 严格模式导致失败
```

---

## ✅ 建议的快速修复步骤

1. **立即**：使用 `strict=False` 加载权重
2. **测试**：评估模型性能是否降低
3. **长期**：
   - 升级 Transformers
   - 或重新用 Qwen2-0.5B 训练模型

---

## 🧪 验证方法

运行调试脚本验证修复是否成功：
```bash
python vggt_vla/eval/debug_eval.py
```

预期输出应该包含：
```
✓ Model loaded to cuda:0
✓ Forward pass successful
✓ Actions are deterministic
✓ Task completed at step X!
```

---

## 📌 总结

| 问题 | 原因 | 影响 | 修复 |
|------|------|------|------|
| 模型权重加载失败 | Qwen3 不可用，fallback 到 Qwen2 | 所有任务失败 | 升级 Transformers |
| 语言模型维度不匹配 | 1024 vs 896 | 权重无法加载 | strict=False 或重训 |
| Eval 成功率 0% | 模型推理无效 | 无法评估模型 | 解决加载问题 |
