# VLA-VGGT: Vision-Language-Action Model with VGGT Backbone

Vision-Language-Action model for robotic manipulation using VGGT (Vision GNN Transformer) as the backbone.

## 架构特点

- **Vision Encoder**: 直接 patch embedding (不使用预训练 vision tower)
- **Language Encoder**: Qwen2-0.5B
- **Backbone**: VGGT (Graph + Transformer 混合)
- **Action Head**: MLP with action chunking

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

## 安装
```bash
pip install -r requirements.txt
```

## 快速开始

### 训练
```bash
python scripts/train.py \
    --data_path /path/to/libero_data.hdf5 \
    --task_names pick_and_place \
    --batch_size 32 \
    --num_epochs 100 \
    --lr 1e-4 \
    --log_dir ./logs
```

### 评估
```bash
python scripts/eval.py \
    --checkpoint ./logs/best_model.pth \
    --device cuda
```

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

## 下一步

1. 准备 LIBERO 数据集
2. 调整超参数
3. 训练并监控 tensorboard
4. 在 LIBERO simulator 中评估

## Citation
```bibtex
@article{vla_vggt,
  title={Vision-Language-Action Model with VGGT Backbone},
  author={Your Name},
  year={2024}
}
```
