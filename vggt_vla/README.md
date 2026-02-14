# VLA-VGGT: Vision-Language-Action Model with VGGT Backbone

Vision-Language-Action model for robotic manipulation using VGGT (Vision GNN Transformer) as the backbone.

> 🎯 **快速开始**: 使用 facebook/vggt + Qwen3-0.6B-Base？查看 [VGGT_QWEN3_GUIDE.md](./VGGT_QWEN3_GUIDE.md)

> 📖 **完整文档**: 查看 [ARCHITECTURE_ANALYSIS.md](./ARCHITECTURE_ANALYSIS.md) 了解详细的架构分析和多模态处理说明

## ✨ 核心特性

- **灵活的 Vision Encoder**: 
  - 直接 patch embedding (快速实验)
  - 预训练 vision tower (DINOv2/CLIP/SigLIP)
- **Language Encoder**: Qwen3-0.6B-Base (with fallback to Qwen2-0.5B)
- **VGGT Backbone**: 
  - facebook/vggt from HuggingFace (预训练权重)
  - 简化版 VGGT (快速训练)
- **Action Head**: MLP with action chunking
- **数据集**: HuggingFace LIBERO datasets (lerobot/libero_spatial_image)

## 多模态 Token 处理

### Vision Tokens
- Image [B,3,224,224] → Patch Embedding → [B,196,768]
- 196 = 14×14 patches (每个 patch 是 16×16 pixels)
- 虽然变成 1D sequence，但通过 2D positional encoding 和 grid graph 保留空间信息

### Language Tokens
- Text → Qwen2 → [B,L,1024] → Projector → [B,L,768]
- 1D sequence structure
- Chain graph 连接

### Token Fusion
- 拼接: [Language Tokens | Vision Tokens]
- Token Type Embeddings 区分模态
- Graph Structure:
  - Language: Chain graph (sequential)
  - Vision: Grid graph (spatial 2D)
  - Cross-modal: Through attention, not graph edges

## 📦 安装

```bash
# 1. 进入目录
cd vggt_vla

# 2. 安装依赖
pip install -r requirements.txt

# 3. 更新 transformers (Qwen3 需要)
pip install -U "transformers>=4.51.0"

# 4. (可选) 安装原始 VGGT (用于本地 fallback)
cd ../vggt
pip install -e .
cd ../vggt_vla
```

## 🚀 快速开始

### 推荐配置: facebook/vggt + Qwen3-0.6B-Base (单帧输入)

```bash
# 1. 测试模型
python scripts/test_vggt_qwen3.py

# 2. 开始训练 - 基础配置
bash scripts/quick_start.sh configs/train_vggt_qwen3.yaml

# 3. 或使用完整配置 (+ DINOv2)
bash scripts/quick_start.sh configs/train_vggt_qwen3_dinov2.yaml
```

> 📖 详细说明: [VGGT_QWEN3_GUIDE.md](./VGGT_QWEN3_GUIDE.md)

### 其他配置

```bash
# 简单配置 - 快速实验
bash scripts/quick_start.sh configs/train_simple.yaml

# 使用 DINOv2 vision tower
bash scripts/quick_start.sh configs/train_with_dinov2.yaml

# 完整配置 - 最佳性能
bash scripts/quick_start.sh configs/train_full.yaml
```

### 方式 2: 命令行参数

```bash
python scripts/train_vla.py \
    --dataset_repo lerobot/libero_spatial_image \
    --use_vision_tower \
    --vision_tower_name facebook/dinov2-base \
    --language_model Qwen/Qwen3-0.6B-Base \
    --freeze_language \
    --batch_size 24 \
    --num_epochs 100 \
    --lr 5e-5 \
    --log_dir ./logs \
    --exp_name my_experiment
```

### 监控训练

在配置中设置 `use_wandb: true`，训练时自动上报到 [Weights & Biases](https://wandb.ai)。也可查看 `log_dir` 下的 `train_*.log` 文本日志。

## 项目结构
```
vla_vggt_project/
├── configs/          # 配置文件
├── models/           # 模型实现
│   ├── components/   # VGGT 核心组件
│   ├── vision_encoder.py
│   ├── language_encoder.py
│   ├── vggt_backbone.py
│   ├── action_head.py
│   └── vla_model.py
├── data/             # 数据加载
├── training/         # 训练工具
└── scripts/          # 训练/评估脚本
```

## 关键实现细节

### 2D → 1D 但保留空间信息

虽然 vision tokens 从 2D grid 变成了 1D sequence，但空间信息通过以下方式保留:

1. **2D Positional Encoding**: 为每个 patch 编码其 (row, col) 位置
2. **Grid Graph**: 显式连接空间邻居 (4-connectivity 或 8-connectivity)
3. **Spatial Info Dict**: 记录 patch_positions [196, 2] 用于 graph 构建

### VGGT Layer 处理

每个 VGGT layer 包含:
1. **Graph Convolution**: 基于 graph edges 的局部信息聚合
2. **Self-Attention**: 全局的 token-to-token interaction
3. **FFN**: 特征变换

这样设计使得:
- Graph Conv 处理 intra-modal 结构 (language chain, vision grid)
- Attention 处理 cross-modal 交互

## 配置说明

编辑 `configs/model_config.py` 来自定义:
- 模型维度
- VGGT 层数
- Graph 结构类型
- Action head 参数

## 📊 配置选项

### 预定义配置

| 配置 | Vision | VGGT | 参数量 | 速度 | 适用场景 |
|------|--------|------|--------|------|----------|
| `train_simple.yaml` | Patch Embed | Simple | ~50M | 快 | 快速实验 |
| `train_with_dinov2.yaml` | DINOv2 | Simple | ~200M | 中 | 平衡性能 |
| `train_full.yaml` | DINOv2 | facebook/vggt | ~500M | 慢 | 最佳性能 |

### 自定义配置

复制并修改配置文件:
```bash
cp configs/train_simple.yaml configs/my_config.yaml
# 编辑 my_config.yaml
bash scripts/quick_start.sh configs/my_config.yaml
```

## 🔧 架构详解

详细的架构分析和多模态处理说明，请查看 [ARCHITECTURE_ANALYSIS.md](./ARCHITECTURE_ANALYSIS.md)

关键改进:
- ✅ 支持 HuggingFace 的 facebook/vggt
- ✅ 适配 Qwen3-0.6B-Base 语言模型
- ✅ 灵活的 vision tower 选项
- ✅ 改进的多模态融合策略
- ✅ HuggingFace datasets 集成

## 📝 常见问题

**Q: 内存不足?**
```bash
# 减小 batch size 或使用简化配置
python scripts/train_vla.py --config configs/train_simple.yaml --batch_size 16
```

**Q: 无法加载 Qwen3-0.6B?**
```bash
# 自动 fallback 到 Qwen2-0.5B
# 或手动指定: --language_model Qwen/Qwen2-0.5B
```

**Q: 如何使用本地数据集?**
```python
# 修改 data/libero_dataset.py 使用本地 HDF5
from data.libero_dataset import get_libero_dataloaders
train_loader, val_loader = get_libero_dataloaders(
    data_path="/path/to/local/data.hdf5",
    ...
)
```

更多问题? 查看 [ARCHITECTURE_ANALYSIS.md](./ARCHITECTURE_ANALYSIS.md#常见问题)

## 📚 相关资源

- [VGGT Paper](https://arxiv.org/abs/2403.08493)
- [LIBERO Dataset](https://huggingface.co/datasets/lerobot/libero_spatial_image)
- [Qwen3 Model](https://huggingface.co/Qwen/Qwen3-0.6B-Base)
- [DINOv2](https://huggingface.co/facebook/dinov2-base)

## 📄 Citation

```bibtex
@article{vla_vggt,
  title={Vision-Language-Action Model with VGGT Backbone},
  author={Your Name},
  year={2024}
}

@article{vggt,
  title={VGGT: Visual Geometry Grounded Transformer},
  author={Meta AI},
  year={2024}
}
```
