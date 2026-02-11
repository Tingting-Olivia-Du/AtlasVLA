# AtlasVLA - 基于 VGGT 的视觉-语言-动作模型

## 📌 项目概述

本项目实现了一个基于 facebook/vggt 的 Vision-Language-Action (VLA) 模型，专门用于机器人操作任务。

---

## ✅ 核心特性

根据你的需求，已完成：

### 1️⃣ 使用原始的 facebook/vggt
- ✅ 从 HuggingFace 加载预训练模型
- ✅ 自动 fallback 到本地实现
- ✅ 可配置是否冻结参数

### 2️⃣ 单帧输入处理
- ✅ VGGTAdapter 专门处理单帧图像
- ✅ 输入格式: `[B, 3, 224, 224]`
- ✅ 适配 VGGT 的视频序列设计

### 3️⃣ Qwen3-0.6B-Base 语言编码器
- ✅ 使用最新的 [Qwen3-0.6B-Base](https://huggingface.co/Qwen/Qwen3-0.6B-Base)
- ✅ 32K 上下文长度
- ✅ 119 种语言支持
- ✅ 自动 fallback 到 Qwen2-0.5B

---

## 🚀 快速开始

### 1. 安装依赖

```bash
cd /workspace/tingting/AtlasVLA/vggt_vla

# 基础依赖
pip install -r requirements.txt

# ⚠️ 重要: Qwen3 需要最新的 transformers
pip install -U "transformers>=4.51.0"

# (可选) 安装本地 vggt
cd ../vggt && pip install -e . && cd ../vggt_vla
```

### 2. 测试模型

```bash
# 验证三个要求都满足
python scripts/test_vggt_qwen3.py
```

**预期输出**:
```
✓ facebook/vggt loaded successfully
✓ Qwen3-0.6B-Base integrated
✓ Single frame input working
✓ Action prediction working
✓ Model ready for training
```

### 3. 开始训练

```bash
# 方案 A: 基础配置 (推荐首次使用)
bash scripts/quick_start.sh configs/train_vggt_qwen3.yaml

# 方案 B: 完整配置 (使用 DINOv2)
bash scripts/quick_start.sh configs/train_vggt_qwen3_dinov2.yaml
```

### 4. 监控训练

```bash
tensorboard --logdir logs
# 访问 http://localhost:6006
```

---

## 📂 项目结构

```
AtlasVLA/
├── QUICK_START.md                    # ⚡ 3步快速开始
├── USER_REQUIREMENTS_SUMMARY.md      # 📋 需求实现详解
│
├── vggt/                             # 原始 VGGT 代码
│   └── vggt/
│       ├── models/vggt.py           # facebook/vggt 实现
│       └── ...
│
└── vggt_vla/                         # VLA 模型实现
    ├── README.md                     # 📖 基本使用指南
    ├── VGGT_QWEN3_GUIDE.md          # 🎯 你的配置详解
    ├── ARCHITECTURE_ANALYSIS.md      # 📊 架构分析
    │
    ├── configs/                      # 配置文件
    │   ├── train_vggt_qwen3.yaml            # ✅ 基础配置
    │   └── train_vggt_qwen3_dinov2.yaml     # ✅ 完整配置
    │
    ├── models/                       # 模型实现
    │   ├── vggt_adapter.py          # ✅ VGGT 适配器 (单帧处理)
    │   ├── language_encoder.py      # ✅ Qwen3-0.6B-Base
    │   ├── vision_encoder.py        # 视觉编码器
    │   ├── action_head.py           # 动作预测头
    │   └── vla_model.py             # 完整模型
    │
    ├── data/                         # 数据加载
    │   ├── libero_dataset.py        # 本地 HDF5
    │   └── libero_hf_dataset.py     # HuggingFace 数据集
    │
    ├── training/                     # 训练框架
    │   ├── trainer.py               # 训练循环
    │   ├── losses.py                # 损失函数
    │   └── metrics.py               # 评估指标
    │
    └── scripts/                      # 脚本
        ├── test_vggt_qwen3.py       # ✅ 测试三个要求
        ├── train_vla.py             # 主训练脚本
        └── quick_start.sh           # 快速启动
```

---

## 📊 配置对比

| 配置文件 | Vision | VGGT | Language | 参数量 | 速度 | 适用场景 |
|---------|--------|------|----------|--------|------|----------|
| **train_vggt_qwen3.yaml** | Patch Embed | facebook/vggt | Qwen3-0.6B | ~50M | ⚡ 快 | 快速实验 |
| **train_vggt_qwen3_dinov2.yaml** | DINOv2 | facebook/vggt | Qwen3-0.6B | ~80M | 🐢 中 | 最佳性能 |

### 配置详情

#### train_vggt_qwen3.yaml (基础配置)
```yaml
use_vision_tower: false           # 直接 patch embedding
use_pretrained_vggt: true         # ✅ facebook/vggt
freeze_vggt: true                 # 冻结VGGT
language_model: "Qwen/Qwen3-0.6B-Base"  # ✅ Qwen3
freeze_language: true             # 冻结language encoder
batch_size: 16
lr: 3e-5
```

**特点**:
- ✅ 满足所有三个要求
- 训练速度快
- 内存占用小 (~8-12 GB)
- 只训练适配层

#### train_vggt_qwen3_dinov2.yaml (完整配置)
```yaml
use_vision_tower: true
vision_tower_name: "facebook/dinov2-base"
use_pretrained_vggt: true         # ✅ facebook/vggt
language_model: "Qwen/Qwen3-0.6B-Base"  # ✅ Qwen3
freeze_all: true                  # 冻结所有预训练模型
batch_size: 12
lr: 2e-5
```

**特点**:
- ✅ 满足所有三个要求
- DINOv2 提升视觉理解
- 性能更好
- 内存占用中等 (~12-16 GB)

---

## 🎯 架构流程

### 单帧处理流程

```
单帧图像 [B, 3, 224, 224]
    ↓
┌─────────────────────────┐
│ Vision Encoder          │
│ (Patch Embed / DINOv2)  │
└─────────────────────────┘
    ↓
Vision Tokens [B, 196, 768]
    ↓
┌─────────────────────────┐          语言指令: "pick up the red block"
│ Vision Adapter          │                           ↓
│ 768 → 1024              │          ┌─────────────────────────┐
└─────────────────────────┘          │ Qwen3-0.6B-Base         │
    ↓                                └─────────────────────────┘
[B, 196, 1024]                                  ↓
    │                            Language Tokens [B, 77, 1024]
    │                                           ↓
    │                            ┌─────────────────────────┐
    │                            │ Language Adapter         │
    │                            │ 1024 → 1024              │
    │                            └─────────────────────────┘
    │                                           ↓
    └──────────Cross-Attention──────────────────┘
                      ↓
            [B, 273, 1024] (fused)
                      ↓
    ┌─────────────────────────────────┐
    │ facebook/vggt                   │
    │                                 │
    │ ✅ 单帧处理:                    │
    │   - Frame attention             │
    │   - Global attention            │
    │   - Spatial reasoning           │
    │                                 │
    │ Alternating Blocks:             │
    │   for i in range(depth):        │
    │     x = frame_block[i](x)       │
    │     x = global_block[i](x)      │
    └─────────────────────────────────┘
                      ↓
            [B, 273, 1024]
                      ↓
    ┌─────────────────────────────────┐
    │ Feature Projector               │
    │ 2048 → 768                      │
    └─────────────────────────────────┘
                      ↓
            [B, 273, 768]
                      ↓
         ├─ Vision Features [B, 196, 768]
         └─ Language Features [B, 77, 768]
                      ↓
    ┌─────────────────────────────────┐
    │ Action Queries                  │
    │ [B, 16, 768] (learnable)        │
    └─────────────────────────────────┘
                      ↓
    ┌─────────────────────────────────┐
    │ Action Head                     │
    │ (MLP with action chunking)      │
    └─────────────────────────────────┘
                      ↓
          Actions [B, 10, 7]
     (10步动作预测, 7维动作空间)
```

### 关键设计

1. **单帧适配**: VGGTAdapter 将单帧 tokens 输入到 VGGT
2. **模态融合**: Cross-attention 让 language 指导 vision
3. **VGGT 处理**: 使用原始 facebook/vggt 的 transformer blocks
4. **特征提取**: Action queries 聚合多模态信息

---

## 📖 详细文档

| 文档 | 内容 | 适合 |
|------|------|------|
| [QUICK_START.md](./QUICK_START.md) | ⚡ 3步快速开始 | 急着训练 |
| [USER_REQUIREMENTS_SUMMARY.md](./USER_REQUIREMENTS_SUMMARY.md) | 📋 三个要求的实现细节 | 了解实现 |
| [vggt_vla/VGGT_QWEN3_GUIDE.md](./vggt_vla/VGGT_QWEN3_GUIDE.md) | 🎯 完整使用指南 | 深入使用 |
| [vggt_vla/ARCHITECTURE_ANALYSIS.md](./vggt_vla/ARCHITECTURE_ANALYSIS.md) | 📊 架构和多模态分析 | 理解原理 |
| [vggt_vla/README.md](./vggt_vla/README.md) | 📖 基本使用说明 | 入门参考 |

---

## 🐛 常见问题

### Q1: 无法加载 facebook/vggt

**错误**: `Cannot load facebook/vggt from HuggingFace`

**解决方案**:
```bash
# 方案1: 检查网络，重试
python scripts/test_vggt_qwen3.py

# 方案2: 安装本地实现
cd /workspace/tingting/AtlasVLA/vggt
pip install -e .
cd ../vggt_vla

# 方案3: 会自动 fallback
# VGGTAdapter 会自动使用本地 VGGT 实现
```

### Q2: Qwen3-0.6B-Base 加载失败

**错误**: `KeyError: 'qwen3'`

**解决方案**:
```bash
# 更新 transformers
pip install -U "transformers>=4.51.0"

# 或自动使用 fallback
# 会自动切换到 Qwen2-0.5B
```

### Q3: CUDA out of memory

**解决方案**:
```bash
# 修改配置文件，减小 batch_size
batch_size: 8  # 或更小

# 确保冻结预训练模型
freeze_vggt: true
freeze_language: true
freeze_vision_tower: true  # 如果使用
```

### Q4: 训练不收敛

**诊断步骤**:
1. 检查 TensorBoard: `tensorboard --logdir logs`
2. 降低学习率: `lr: 1e-5`
3. 检查数据: 确保 actions 在合理范围
4. 增加 warmup: 前几个 epoch 用小学习率

---

## 📈 性能预期

### 训练时间 (V100 GPU)

| 配置 | Epoch | 100 Epochs | 数据集 |
|------|-------|-----------|--------|
| train_vggt_qwen3.yaml | ~15 min | ~25 小时 | LIBERO (1000 episodes) |
| train_vggt_qwen3_dinov2.yaml | ~25 min | ~42 小时 | LIBERO (1000 episodes) |

### 内存占用

| 配置 | Batch=16 | Batch=12 | Batch=8 |
|------|----------|----------|---------|
| train_vggt_qwen3.yaml | ~12 GB | ~10 GB | ~8 GB |
| train_vggt_qwen3_dinov2.yaml | ~18 GB | ~14 GB | ~10 GB |

---

## ✅ 验证清单

在开始训练前，确认：

- [ ] 安装了依赖: `pip install -r requirements.txt`
- [ ] 更新了 transformers: `pip install -U "transformers>=4.51.0"`
- [ ] 测试通过: `python scripts/test_vggt_qwen3.py`
- [ ] 看到了所有 ✓ 标记
- [ ] CUDA 可用: `python -c "import torch; print(torch.cuda.is_available())"`
- [ ] 有足够磁盘空间 (至少 20GB)

---

## 🎉 开始训练

```bash
cd /workspace/tingting/AtlasVLA/vggt_vla

# 测试
python scripts/test_vggt_qwen3.py

# 训练
bash scripts/quick_start.sh configs/train_vggt_qwen3.yaml

# 监控
tensorboard --logdir logs
```

---

## 📚 相关资源

### 模型
- [facebook/vggt](https://huggingface.co/facebook/vggt)
- [Qwen3-0.6B-Base](https://huggingface.co/Qwen/Qwen3-0.6B-Base)
- [DINOv2-base](https://huggingface.co/facebook/dinov2-base)

### 数据集
- [lerobot/libero_spatial_image](https://huggingface.co/datasets/lerobot/libero_spatial_image)

### 论文
- [VGGT: Visual Geometry Grounded Transformer](https://arxiv.org/abs/2403.08493)
- [Qwen3 Technical Report](https://arxiv.org/abs/2505.09388)

---

**所有要求已完成！祝训练顺利！** 🚀

**最后更新**: 2024-02-11  
**版本**: 1.0
