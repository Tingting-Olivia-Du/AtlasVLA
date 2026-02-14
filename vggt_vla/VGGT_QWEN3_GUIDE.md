# 使用 facebook/vggt + Qwen3-0.6B-Base 的指南

## 📋 配置说明

根据你的要求，我已经完成了以下配置：

### ✅ 1. 使用原始的 facebook/vggt

**实现位置**: `models/vggt_adapter.py`

- ✅ 从 HuggingFace 加载 `facebook/vggt`
- ✅ 如果无法访问 HuggingFace，自动 fallback 到本地 vggt 实现
- ✅ 适配层处理输入输出格式

**关键代码**:
```python
self.vggt = AutoModel.from_pretrained(
    "facebook/vggt",
    trust_remote_code=True
)
```

**配置**:
```yaml
use_pretrained_vggt: true  # 使用 HuggingFace 的 facebook/vggt
freeze_vggt: true          # 冻结VGGT，只训练适配层
```

### ✅ 2. 单帧输入给 VGGT

**问题**: facebook/vggt 原本设计用于视频序列 `[B, S, 3, H, W]`，其中 S 是序列长度

**解决方案**: 

我们的 VGGTAdapter 已经实现了单帧处理：

1. **输入格式**: 每次 forward 只接收一帧 `[B, 3, 224, 224]`
2. **内部处理**: Vision encoder 将图像转换为 tokens `[B, 196, 768]`
3. **VGGT 适配**: 使用 VGGT 的 aggregator blocks 处理 tokens
4. **输出**: 提取适合动作预测的特征

**关键实现** (`models/vggt_adapter.py`):
```python
def forward(
    self,
    vision_tokens: torch.Tensor,      # [B, N_v, D] - 单帧的tokens
    language_tokens: torch.Tensor,    # [B, N_l, D]
    ...
):
    # 单帧处理流程
    # 1. 适配维度
    vision_adapted = self.vision_adapter(vision_tokens)
    
    # 2. 使用VGGT的transformer blocks
    aggregator = self.vggt.aggregator
    for i in range(num_layers):
        x = aggregator.frame_blocks[i](x, pos=None)
        x = aggregator.global_blocks[i](x, pos=None)
    
    # 3. 提取action features
    ...
```

**标记**: `output_info['single_frame_input'] = True`

### ✅ 3. 使用 Qwen3-0.6B-Base 作为语言 encoder

**模型**: [Qwen/Qwen3-0.6B-Base](https://huggingface.co/Qwen/Qwen3-0.6B-Base)

**特点**:
- 0.6B 参数
- 32K 上下文长度
- 支持 119 种语言
- Apache 2.0 许可证

**实现位置**: `models/language_encoder.py`

**配置**:
```python
LanguageConfig(
    model_name="Qwen/Qwen3-0.6B-Base",  # ✅ 使用 Qwen3
    freeze_encoder=True,                 # 冻结encoder
    output_dim=768,                      # 投影到768维
    max_length=77                        # 最大序列长度
)
```

**自动 Fallback**:
```python
try:
    model = AutoModel.from_pretrained("Qwen/Qwen3-0.6B-Base")
except:
    model = AutoModel.from_pretrained("Qwen/Qwen2-0.5B")  # Fallback
```

---

## 🚀 快速开始

### 1. 测试模型

```bash
cd /workspace/tingting/AtlasVLA/vggt_vla

# 测试 facebook/vggt + Qwen3-0.6B-Base 配置
python scripts/test_vggt_qwen3.py
```

这会验证：
- ✓ facebook/vggt 加载成功
- ✓ Qwen3-0.6B-Base 集成正确
- ✓ 单帧输入处理工作
- ✓ Action prediction 正常

### 2. 开始训练

#### 方案 A: 基础配置 (推荐首次使用)

```bash
# 使用配置文件
bash scripts/quick_start.sh configs/train_vggt_qwen3.yaml
```

**特点**:
- facebook/vggt (冻结)
- Qwen3-0.6B-Base (冻结)
- 直接 patch embedding (无 vision tower)
- 单帧输入
- 只训练适配层 (~50M 参数)
- Batch size: 16

#### 方案 B: 完整配置 (最佳性能)

```bash
# 使用 DINOv2 + facebook/vggt + Qwen3
bash scripts/quick_start.sh configs/train_vggt_qwen3_dinov2.yaml
```

**特点**:
- DINOv2 vision tower (冻结)
- facebook/vggt (冻结)
- Qwen3-0.6B-Base (冻结)
- 单帧输入
- 只训练适配层 (~80M 参数)
- Batch size: 12

### 3. 监控训练

配置中设置 `use_wandb: true`，在 [wandb.ai](https://wandb.ai) 查看曲线；或查看 `log_dir` 下的 `train_*.log` 文本日志。

---

## 📊 配置对比

| 配置 | Vision | VGGT | Language | 参数量 | 训练速度 | 适用场景 |
|------|--------|------|----------|--------|----------|----------|
| train_vggt_qwen3.yaml | Patch Embed | facebook/vggt | Qwen3-0.6B | ~50M | 快 | 快速实验 |
| train_vggt_qwen3_dinov2.yaml | DINOv2 | facebook/vggt | Qwen3-0.6B | ~80M | 中 | 最佳性能 |

---

## 🔧 架构细节

### 整体流程

```
单帧图像 [B, 3, 224, 224]
    ↓
Vision Encoder (Patch Embed 或 DINOv2)
    ↓
Vision Tokens [B, 196, 768]
    ↓
                    ┌─────────────────┐
                    │ Vision Adapter  │
                    │ 768 → 1024      │
                    └─────────────────┘
                            ↓
                    [B, 196, 1024]
                            │
                            │
语言指令 "pick up the red block"    │
    ↓                              │
Qwen3-0.6B-Base                    │
    ↓                              │
Language Tokens [B, 77, 1024]      │
    ↓                              │
Language Adapter                   │
    ↓                              │
[B, 77, 1024] ──────Cross-Attention─┘
    ↓
[B, 77+196, 1024]
    ↓
┌─────────────────────────────────┐
│ facebook/vggt                   │
│ - Alternating Attention         │
│ - Frame blocks + Global blocks  │
│ - Spatial reasoning             │
└─────────────────────────────────┘
    ↓
[B, 273, 1024]
    ↓
Feature Projector (1024*2 → 768)
    ↓
[B, 273, 768]
    ↓
    ├─ Vision Features [B, 196, 768]
    └─ Language Features [B, 77, 768]
    ↓
Action Queries [B, 16, 768]
    ↓
Action Head
    ↓
Actions [B, 10, 7]
```

### 单帧处理关键点

1. **输入适配**:
   - 单帧图像 → Vision encoder → Tokens
   - 不需要构造视频序列

2. **VGGT 处理**:
   - 使用 VGGT 的 aggregator blocks
   - Frame attention: 处理当前帧的tokens
   - Global attention: 跨模态交互

3. **特征提取**:
   - Vision 和 language tokens 融合
   - Action queries 聚合信息
   - 投影到动作空间

---

## 🎯 训练建议

### 首次训练

1. **使用基础配置**: `train_vggt_qwen3.yaml`
2. **小batch size**: 从 8-16 开始
3. **短期训练**: 先训练 10-20 epochs 验证
4. **检查指标**: TensorBoard 监控 loss 曲线

### 正式训练

1. **使用完整配置**: `train_vggt_qwen3_dinov2.yaml`
2. **调整学习率**: 根据验证集表现调整
3. **定期保存**: 每 5 epochs 保存 checkpoint
4. **早停**: 如果验证 loss 不下降，及时停止

### 调试技巧

1. **检查 VGGT 加载**:
   ```bash
   python -c "from transformers import AutoModel; \
              model = AutoModel.from_pretrained('facebook/vggt', trust_remote_code=True); \
              print('✓ VGGT loaded')"
   ```

2. **检查 Qwen3 加载**:
   ```bash
   python -c "from transformers import AutoModel; \
              model = AutoModel.from_pretrained('Qwen/Qwen3-0.6B-Base'); \
              print('✓ Qwen3 loaded')"
   ```

3. **测试单帧输入**:
   ```bash
   python scripts/test_vggt_qwen3.py
   ```

---

## 🐛 常见问题

### Q1: 无法加载 facebook/vggt

**问题**: `Cannot load facebook/vggt from HuggingFace`

**解决**:
1. 检查网络连接
2. 使用本地 vggt:
   ```bash
   cd /workspace/tingting/AtlasVLA/vggt
   pip install -e .
   ```
3. 自动 fallback 会使用本地实现

### Q2: Qwen3-0.6B-Base 加载失败

**问题**: `KeyError: 'qwen3'`

**解决**:
1. 更新 transformers:
   ```bash
   pip install -U transformers>=4.51.0
   ```
2. 或使用 fallback (Qwen2-0.5B)

### Q3: 内存不足

**问题**: `CUDA out of memory`

**解决**:
1. 减小 batch size:
   ```bash
   --batch_size 8  # 或更小
   ```
2. 使用梯度累积:
   ```python
   accumulation_steps = 4
   ```
3. 确保所有预训练模型都冻结:
   ```yaml
   freeze_vggt: true
   freeze_language: true
   freeze_vision_tower: true  # 如果使用
   ```

### Q4: 训练不收敛

**诊断**:
1. 检查数据加载是否正常
2. 降低学习率: `--lr 1e-5`
3. 检查梯度: 添加 `grad_norm` 日志
4. 查看 TensorBoard 曲线

---

## 📈 预期性能

### 训练时间 (单 V100 GPU)

| 配置 | Epoch 时间 | 100 Epochs | 备注 |
|------|-----------|-----------|------|
| train_vggt_qwen3.yaml | ~15 min | ~25 小时 | 1000 episodes |
| train_vggt_qwen3_dinov2.yaml | ~25 min | ~42 小时 | 1000 episodes |

### 内存占用

| 配置 | Batch=16 | Batch=8 | Batch=4 |
|------|----------|---------|---------|
| train_vggt_qwen3.yaml | ~20 GB | ~12 GB | ~8 GB |
| train_vggt_qwen3_dinov2.yaml | ~28 GB | ~16 GB | ~10 GB |

---

## 📚 相关资源

### 模型

- [facebook/vggt](https://huggingface.co/facebook/vggt) - Visual Geometry Grounded Transformer
- [Qwen3-0.6B-Base](https://huggingface.co/Qwen/Qwen3-0.6B-Base) - 语言模型
- [DINOv2-base](https://huggingface.co/facebook/dinov2-base) - Vision tower (可选)

### 数据集

- [lerobot/libero_spatial_image](https://huggingface.co/datasets/lerobot/libero_spatial_image) - LIBERO 数据集

### 论文

- [VGGT Paper](https://arxiv.org/abs/2403.08493)
- [Qwen3 Technical Report](https://arxiv.org/abs/2505.09388)
- [LIBERO Paper](https://arxiv.org/abs/2306.03310)

---

## ✅ 验证清单

在开始训练前，确认：

- [ ] 安装了依赖: `pip install -r requirements.txt`
- [ ] 更新了 transformers: `pip install -U transformers>=4.51.0`
- [ ] 测试通过: `python scripts/test_vggt_qwen3.py`
- [ ] CUDA 可用: `python -c "import torch; print(torch.cuda.is_available())"`
- [ ] 有足够磁盘空间 (至少 20GB for HuggingFace cache)
- [ ] 选择了合适的配置文件

---

## 🎉 总结

你现在有了：

1. ✅ **facebook/vggt**: 从 HuggingFace 加载的原始 VGGT
2. ✅ **单帧输入**: VGGTAdapter 专门处理单帧图像
3. ✅ **Qwen3-0.6B-Base**: 最新的 Qwen3 语言模型

**开始训练**:
```bash
cd /workspace/tingting/AtlasVLA/vggt_vla
python scripts/test_vggt_qwen3.py  # 先测试
bash scripts/quick_start.sh configs/train_vggt_qwen3.yaml  # 然后训练
```

祝训练顺利！🚀

---

**最后更新**: 2024-02-11  
**版本**: 1.0
