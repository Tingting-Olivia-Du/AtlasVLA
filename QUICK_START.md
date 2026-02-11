# 快速开始 - facebook/vggt + Qwen3-0.6B-Base

## 🎯 你的三个要求

✅ **1. 用原始的 vggt** - 从 HuggingFace 加载 facebook/vggt  
✅ **2. 单帧输入给 vggt** - VGGTAdapter 专门处理单帧  
✅ **3. Qwen3-0.6B-Base** - 作为语言 encoder

---

## ⚡ 3步开始训练

### Step 1: 安装

```bash
cd /workspace/tingting/AtlasVLA/vggt_vla
pip install -r requirements.txt
pip install -U "transformers>=4.51.0"  # Qwen3 需要
```

### Step 2: 测试

```bash
python scripts/test_vggt_qwen3.py
```

看到这些输出就OK：
- ✓ facebook/vggt loaded
- ✓ Qwen3-0.6B-Base integrated  
- ✓ Single frame input working

### Step 3: 训练

```bash
# 基础配置 (推荐)
bash scripts/quick_start.sh configs/train_vggt_qwen3.yaml

# 或完整配置 (+ DINOv2)
bash scripts/quick_start.sh configs/train_vggt_qwen3_dinov2.yaml
```

---

## 📊 监控

```bash
tensorboard --logdir logs
# 访问 http://localhost:6006
```

---

## 📖 详细文档

- 🎯 **你的配置指南**: [VGGT_QWEN3_GUIDE.md](./vggt_vla/VGGT_QWEN3_GUIDE.md)
- 📋 **需求实现总结**: [USER_REQUIREMENTS_SUMMARY.md](./USER_REQUIREMENTS_SUMMARY.md)
- 📖 **架构分析**: [vggt_vla/ARCHITECTURE_ANALYSIS.md](./vggt_vla/ARCHITECTURE_ANALYSIS.md)

---

## 🐛 常见问题

### Q: 无法加载 facebook/vggt？
```bash
# 方案1: 安装本地 vggt
cd /workspace/tingting/AtlasVLA/vggt
pip install -e .

# 方案2: 会自动 fallback 到本地实现
```

### Q: Qwen3 加载失败？
```bash
# 更新 transformers
pip install -U "transformers>=4.51.0"

# 或会自动 fallback 到 Qwen2-0.5B
```

### Q: 内存不足？
```bash
# 修改配置文件
batch_size: 8  # 改小
freeze_vggt: true  # 确保冻结
freeze_language: true
```

---

## 🎉 配置说明

### configs/train_vggt_qwen3.yaml (基础)
- ✅ facebook/vggt (冻结)
- ✅ Qwen3-0.6B-Base (冻结)
- ✅ 单帧输入
- Direct patch embedding
- Batch size: 16
- ~50M 训练参数

### configs/train_vggt_qwen3_dinov2.yaml (完整)
- ✅ facebook/vggt (冻结)
- ✅ Qwen3-0.6B-Base (冻结)  
- ✅ 单帧输入
- DINOv2 vision tower (冻结)
- Batch size: 12
- ~80M 训练参数

---

**准备好了吗？开始训练！** 🚀

```bash
cd /workspace/tingting/AtlasVLA/vggt_vla
python scripts/test_vggt_qwen3.py && \
bash scripts/quick_start.sh configs/train_vggt_qwen3.yaml
```
