# VLA-VGGT 评估模块

VLA-VGGT 模型在 LIBERO 环境中的完整评估系统。

## 文件说明

| 文件 | 说明 |
|------|------|
| `eval_vla.py` | 主评估脚本，包含 `VLAEvaluator` 类 |
| `test_eval.py` | 评估脚本的单元测试和验证 |
| `run_eval.sh` | 便捷的 Shell 运行脚本（推荐） |
| `README.md` | 本文件 |

## 快速开始

### 方式 1: 直接 Python 命令

```bash
cd ../  # 进入 vggt_vla 目录

# 快速测试（单任务，2个回合）
python eval/eval_vla.py \
    --checkpoint logs/vla_libero_spatial/best_model_libero_spatial_image_20260214_045544_epoch297_step26690_loss0.0017.pt \
    --benchmark libero_spatial \
    --task_ids 0 \
    --num_episodes 2 \
    --num_envs 1

# 标准评估（所有任务，10个回合）
python eval/eval_vla.py \
    --checkpoint logs/vla_libero_spatial/best_model_libero_spatial_image_20260214_045544_epoch297_step26690_loss0.0017.pt \
    --benchmark libero_spatial
```

### 方式 2: Shell 脚本（推荐）

```bash
# 添加执行权限
chmod +x eval/run_eval.sh

# 快速测试
./eval/run_eval.sh \
    -c logs/vla_libero_spatial/best_model_libero_spatial_image_20260214_045544_epoch297_step26690_loss0.0017.pt \
    -b libero_spatial \
    -t "0" \
    -n 2 \
    -e 1

# 完整评估
./eval/run_eval.sh \
    -c logs/vla_libero_spatial/best_model_libero_spatial_image_20260214_045544_epoch297_step26690_loss0.0017.pt \
    -b libero_spatial \
    -n 20 \
    -v
```

## 命令行参数

### 必需参数

```
--checkpoint PATH          模型检查点文件路径
--benchmark {libero_*}     LIBERO 基准名称
```

### 可选参数

```
--task_ids [ID ...]        要评估的任务 ID（默认: 全部）
--num_episodes N           每个任务的评估回合数（默认: 10）
--max_steps N              每个回合的最大步数（默认: 500）
--num_envs N               并行环境数（默认: 20）
--save_videos              保存评估视频
--output_dir PATH          输出目录（默认: ./eval_results）
--device {cuda|cpu}        计算设备（默认: cuda）
```

## 支持的基准

- `libero_spatial`: 空间推理（10 个任务）
- `libero_object`: 物体识别（10 个任务）
- `libero_goal`: 目标推理（10 个任务）
- `libero_10`: 混合任务（10 个任务）

## 评估工作流

1. **验证环境** (可选)
   ```bash
   python eval/test_eval.py
   ```

2. **运行评估**
   ```bash
   python eval/eval_vla.py --checkpoint <path> --benchmark <name>
   ```

3. **查看结果**
   ```bash
   cat eval_results/eval_results.json
   ```

## 输出结构

```
eval_results/
├── eval_results.json         # 结果汇总（JSON）
└── videos_task_X/            # 视频（如果指定 --save_videos）
    ├── episode_0.mp4
    └── ...
```

## 使用场景

### 快速验证（~5 分钟）
```bash
python eval/eval_vla.py \
    --checkpoint logs/.../best_model.pt \
    --benchmark libero_spatial \
    --task_ids 0 \
    --num_episodes 2 \
    --num_envs 1
```

### 标准评估（~30-45 分钟）
```bash
python eval/eval_vla.py \
    --checkpoint logs/.../best_model.pt \
    --benchmark libero_spatial
```

### 完整评估（~1.5-2.5 小时）
```bash
python eval/eval_vla.py \
    --checkpoint logs/.../best_model.pt \
    --benchmark libero_spatial \
    --num_episodes 20 \
    --save_videos
```

## 常见问题

**Q: 如何保存视频？**
A: 使用 `--save_videos` 标志。

**Q: 内存不足？**
A: 减少 `--num_envs`，例如 `--num_envs 1` 或 `--num_envs 5`。

**Q: 找不到检查点？**
A: 检查文件路径是否正确且文件存在。

## 详细文档

更多信息请参考：
- `../../EVAL_README.md`: 完整的评估系统说明
- `../../vggt_vla/EVAL_GUIDE.md`: 详细的使用指南

## 参考资源

- 原始 LIBERO evaluate.py: `../../dataset/LIBERO/libero/lifelong/evaluate.py`
- VLA 模型: `../models/vla_model.py`
- 模型配置: `../configs/model_config.py`
