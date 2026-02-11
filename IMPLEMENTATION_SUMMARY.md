# VLA-VGGT 实现总结

## 📌 项目概述

本项目实现了一个基于 VGGT 的 Vision-Language-Action (VLA) 模型，用于机器人操作任务。

**关键改进**:
1. ✅ 支持从 HuggingFace 加载 `facebook/vggt`
2. ✅ 支持 Qwen3-0.6B-Base 语言模型
3. ✅ 灵活的 vision encoder (直接 patch embedding 或预训练 vision tower)
4. ✅ 改进的多模态融合策略
5. ✅ HuggingFace LIBERO 数据集集成

---

## 📂 项目结构

```
vggt_vla/
├── configs/                      # 配置文件
│   ├── model_config.py          # 模型配置类
│   ├── train_simple.yaml        # 简单配置
│   ├── train_with_dinov2.yaml   # 使用 DINOv2
│   └── train_full.yaml          # 完整配置
│
├── models/                       # 模型实现
│   ├── vision_encoder.py        # 视觉编码器 (支持多种方案)
│   ├── language_encoder.py      # 语言编码器 (Qwen3-0.6B)
│   ├── vggt_adapter.py          # VGGT 适配器 (HF + 简化版)
│   ├── vggt_backbone.py         # 原始 VGGT backbone
│   ├── action_head.py           # 动作预测头
│   ├── vla_model.py             # 完整 VLA 模型
│   └── components/              # VGGT 组件
│       ├── vggt_layers.py       # VGGT 层
│       ├── token_fusion.py      # Token 融合
│       └── graph_builder.py     # 图构建
│
├── data/                         # 数据加载
│   ├── libero_dataset.py        # LIBERO 本地数据集
│   └── libero_hf_dataset.py     # LIBERO HuggingFace 数据集
│
├── training/                     # 训练相关
│   ├── trainer.py               # 训练循环
│   ├── losses.py                # 损失函数
│   └── metrics.py               # 评估指标
│
├── scripts/                      # 脚本
│   ├── train_vla.py             # 主训练脚本
│   ├── test_model.py            # 模型测试
│   ├── eval.py                  # 评估脚本
│   └── quick_start.sh           # 快速启动
│
├── README.md                     # 用户文档
├── ARCHITECTURE_ANALYSIS.md      # 架构分析
└── requirements.txt              # 依赖
```

---

## 🔧 核心组件说明

### 1. Vision Encoder (`models/vision_encoder.py`)

**功能**: 将图像编码为 token 序列

**支持的方案**:
- **方案 A**: 直接 patch embedding (无预训练)
  ```python
  VisionConfig(
      use_vision_tower=False,
      img_size=224,
      patch_size=16
  )
  ```

- **方案 B**: 预训练 vision tower
  ```python
  VisionConfig(
      use_vision_tower=True,
      vision_tower_name="facebook/dinov2-base",  # 或 CLIP, SigLIP
      freeze_vision_tower=True
  )
  ```

**输出**: `[B, N_patches, 768]` + spatial_info

### 2. Language Encoder (`models/language_encoder.py`)

**功能**: 将文本指令编码为 token 序列

**配置**:
```python
LanguageConfig(
    model_name="Qwen/Qwen3-0.6B-Base",  # 自动 fallback 到 Qwen2-0.5B
    max_length=77,
    freeze_encoder=True,
    output_dim=768
)
```

**输出**: `[B, L, 768]` + language_info

### 3. VGGT Backbone

**两种实现**:

#### A. VGGTAdapter (`models/vggt_adapter.py`)
- 从 HuggingFace 加载 `facebook/vggt`
- 添加适配层处理输入输出
- 支持多模态 token 注入

```python
VGGTConfig(
    use_pretrained_vggt=True,
    freeze_vggt=True,  # 只训练适配层
    embed_dim=768
)
```

#### B. SimpleVGGTBackbone (`models/vggt_backbone.py`)
- 简化实现: Graph Conv + Self-Attention
- 快速训练和实验
- 完全可训练

```python
VGGTConfig(
    use_pretrained_vggt=False,
    depth=6,
    num_heads=12,
    graph_type='grid'
)
```

**输出**: 
- `vision_features`: `[B, N_v, 768]`
- `language_features`: `[B, N_l, 768]`
- `global_features`: `[B, 16, 768]` (action queries)

### 4. Action Head (`models/action_head.py`)

**功能**: 从全局特征预测动作序列

```python
ActionHeadConfig(
    input_dim=768,
    action_dim=7,  # (x, y, z, quat, gripper)
    action_horizon=10,  # 预测未来 10 步
    use_action_chunking=True
)
```

**输出**: `[B, T, action_dim]`

---

## 🚀 使用指南

### 快速测试模型

```bash
cd vggt_vla

# 测试简单配置
python scripts/test_model.py --config simple

# 测试 DINOv2 配置
python scripts/test_model.py --config dinov2
```

### 训练模型

#### 方式 1: 使用预定义配置

```bash
# 简单配置 (推荐首次使用)
bash scripts/quick_start.sh configs/train_simple.yaml

# DINOv2 配置
bash scripts/quick_start.sh configs/train_with_dinov2.yaml

# 完整配置
bash scripts/quick_start.sh configs/train_full.yaml
```

#### 方式 2: 命令行参数

```bash
python scripts/train_vla.py \
  --dataset_repo lerobot/libero_spatial_image \
  --use_vision_tower \
  --vision_tower_name facebook/dinov2-base \
  --freeze_vision_tower \
  --language_model Qwen/Qwen3-0.6B-Base \
  --freeze_language \
  --use_pretrained_vggt \
  --freeze_vggt \
  --batch_size 16 \
  --num_epochs 100 \
  --lr 3e-5 \
  --log_dir ./logs \
  --exp_name my_experiment
```

### 监控训练

```bash
# 启动 TensorBoard
tensorboard --logdir logs

# 浏览器访问
# http://localhost:6006
```

---

## 📊 配置推荐

### 场景 1: 快速实验和调试
```yaml
配置文件: configs/train_simple.yaml

特点:
- 直接 patch embedding
- 简化版 VGGT
- Qwen2-0.5B
- 快速训练 (~50M 参数)

适合:
- 验证数据 pipeline
- 快速迭代
- 资源有限环境
```

### 场景 2: 平衡性能和速度
```yaml
配置文件: configs/train_with_dinov2.yaml

特点:
- DINOv2 vision tower
- 简化版 VGGT
- Qwen3-0.6B
- 中等训练时间 (~200M 参数)

适合:
- 正式实验
- 论文基线
- 大多数应用
```

### 场景 3: 最佳性能
```yaml
配置文件: configs/train_full.yaml

特点:
- DINOv2 vision tower
- facebook/vggt (预训练)
- Qwen3-0.6B
- 慢速训练 (~500M 参数)

适合:
- 最终模型
- 竞赛提交
- 充足计算资源
```

---

## 🔍 多模态处理详解

### 问题分析

原始 `vggt_vla` 实现的主要问题:

1. **VGGT 不是从 HuggingFace 加载**
   - 现有: 自己实现的简化版
   - 缺失: facebook/vggt 的预训练权重和完整架构

2. **输入输出格式不匹配**
   - facebook/vggt 期望: `[B, S, 3, H, W]` (视频)
   - VLA 任务: `[B, 3, H, W]` (单帧)
   - 需要适配层

3. **Language 注入方式**
   - 原始 VGGT 无 language 输入
   - 需要设计多模态融合策略

### 解决方案

#### 1. VGGT Adapter
```python
# models/vggt_adapter.py

class VGGTAdapter:
    - 加载 facebook/vggt
    - 适配单帧输入
    - 注入 language tokens
    - 添加 action queries
    - 特征投影层
```

#### 2. Token Fusion
```python
Vision tokens: [B, 196, 768]
Language tokens: [B, 77, 768]
  ↓
Token Type Embeddings:
  - Vision: type=0
  - Language: type=1
  ↓
Concat: [B, 273, 768]
  ↓
VGGT Processing:
  - Graph Conv (intra-modal)
  - Self-Attention (cross-modal)
```

#### 3. Graph Structure
```python
Vision: Grid graph (2D spatial)
  - 14×14 patches
  - 4-connectivity

Language: Chain graph (sequential)
  - 1D sequence
  - Bidirectional

Cross-modal: Through attention
  - No graph edges
  - Full attention matrix
```

---

## 📈 性能优化建议

### 1. 数据加载
```python
# 增加 workers
num_workers=8

# Pin memory
pin_memory=True

# Prefetch
persistent_workers=True
```

### 2. 训练优化
```python
# 混合精度
from torch.cuda.amp import autocast, GradScaler

# 梯度累积
accumulation_steps=4

# 梯度检查点 (节省内存)
use_gradient_checkpointing=True
```

### 3. 模型优化
```python
# 编译模型 (PyTorch 2.0+)
model = torch.compile(model)

# 冻结预训练模型
freeze_vision_tower=True
freeze_language=True
freeze_vggt=True
```

---

## 🐛 常见问题和解决方案

### Q1: CUDA Out of Memory

**解决方案**:
```bash
# 方法 1: 减小 batch size
--batch_size 8

# 方法 2: 使用梯度累积
--batch_size 8 --accumulation_steps 4  # 等效 batch_size=32

# 方法 3: 使用更小的模型
--use_vision_tower false

# 方法 4: 冻结更多参数
--freeze_vision_tower --freeze_language --freeze_vggt
```

### Q2: 无法加载 Qwen3-0.6B

**自动 Fallback**:
```python
# language_encoder.py 中已实现自动 fallback
try:
    model = AutoModel.from_pretrained("Qwen/Qwen3-0.6B-Base")
except:
    model = AutoModel.from_pretrained("Qwen/Qwen2-0.5B")
```

**手动指定**:
```bash
--language_model Qwen/Qwen2-0.5B
```

### Q3: 训练不收敛

**诊断步骤**:
```bash
# 1. 检查数据
python -c "from data.libero_hf_dataset import *; ..."

# 2. 降低学习率
--lr 1e-5

# 3. 检查梯度
# 在 trainer.py 中添加:
grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
print(f"Grad norm: {grad_norm}")

# 4. 可视化
tensorboard --logdir logs
```

### Q4: facebook/vggt 加载失败

**Fallback 到简化版**:
```python
# vggt_adapter.py 中已实现
try:
    vggt = AutoModel.from_pretrained("facebook/vggt")
except:
    from vggt.models.vggt import VGGT
    vggt = VGGT(...)  # 使用本地实现
```

或直接使用简化版:
```bash
--use_pretrained_vggt false
```

---

## 📦 依赖安装

### 基础依赖
```bash
pip install -r vggt_vla/requirements.txt
```

### 可选依赖

#### 1. 原始 VGGT
```bash
cd vggt
pip install -e .
```

#### 2. Vision Towers
```bash
# DINOv2
pip install timm

# CLIP/SigLIP
# (已包含在 transformers 中)
```

#### 3. 分布式训练
```bash
pip install accelerate
```

---

## 🎯 下一步

### 短期目标
1. ✅ 完成架构实现
2. ✅ 验证模型可以运行
3. ⏳ 在 LIBERO 数据集上训练
4. ⏳ 评估性能和对比 baseline

### 中期目标
1. 探索更多融合策略 (cross-attention, gating)
2. 支持更多 vision towers (SAM, EVA-CLIP)
3. 多任务学习
4. 模型压缩和量化

### 长期目标
1. 真实机器人评估
2. 开源预训练模型
3. 论文发表
4. 社区贡献

---

## 📚 参考资料

### 论文
- [VGGT: Visual Geometry Grounded Transformer](https://arxiv.org/abs/2403.08493)
- [LIBERO: Benchmarking Knowledge Transfer for Lifelong Robot Learning](https://arxiv.org/abs/2306.03310)

### 代码
- [facebook/vggt](https://huggingface.co/facebook/vggt)
- [LIBERO Dataset](https://huggingface.co/datasets/lerobot/libero_spatial_image)
- [Qwen3](https://huggingface.co/Qwen/Qwen3-0.6B-Base)

### 文档
- [vggt_vla/README.md](./vggt_vla/README.md) - 用户文档
- [vggt_vla/ARCHITECTURE_ANALYSIS.md](./vggt_vla/ARCHITECTURE_ANALYSIS.md) - 架构分析

---

## 👥 贡献

欢迎提交 Issue 和 Pull Request!

---

## 📄 License

See LICENSE file.

---

**最后更新**: 2024-02-11  
**版本**: 1.0  
**作者**: VLA-VGGT Team
