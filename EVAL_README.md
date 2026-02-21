# VLA-VGGT 评估脚本 - 完整说明

## 概述

全新的 VLA-VGGT 评估脚本系统，用于在 LIBERO 环境中评估 VLA 模型。脚本从头重写，基于 LIBERO 的官方 `evaluate.py` 进行设计。

## 新增文件

### 核心评估脚本（eval 文件夹）

| 文件 | 说明 |
|------|------|
| `vggt_vla/eval/eval_vla.py` | 主评估脚本，包含 `VLAEvaluator` 类 |
| `vggt_vla/eval/test_eval.py` | 评估脚本的单元测试和验证 |
| `vggt_vla/eval/run_eval.sh` | 完整的评估运行脚本（推荐使用） |
| `vggt_vla/eval/README.md` | eval 模块的快速参考 |

### 文档（项目根目录）

| 文件 | 说明 |
|------|------|
| `EVAL_README.md` | 完整的评估系统说明书 |
| `EVAL_IMPLEMENTATION_SUMMARY.md` | 实现细节和技术总结 |
| `vggt_vla/EVAL_GUIDE.md` | 详细的使用指南和示例（可选） |

## 快速开始

### 方式 1: 直接 Python 命令（推荐用于一次性运行）

```bash
cd vggt_vla

# 快速测试（单个任务，2个回合，1个环境）
python eval/eval_vla.py \
    --checkpoint logs/vla_libero_spatial/best_model_libero_spatial_image_20260214_045544_epoch297_step26690_loss0.0017.pt \
    --benchmark libero_spatial \
    --task_ids 0 \
    --num_episodes 2 \
    --num_envs 1

# 标准评估（所有任务，10个回合）
python eval/eval_vla.py \
    --checkpoint logs/vla_libero_spatial/best_model_libero_spatial_image_20260214_045544_epoch297_step26690_loss0.0017.pt \
    --benchmark libero_spatial \
    --num_episodes 10

# 完整评估（所有任务，20个回合，保存视频）
python eval/eval_vla.py \
    --checkpoint logs/vla_libero_spatial/best_model_libero_spatial_image_20260214_045544_epoch297_step26690_loss0.0017.pt \
    --benchmark libero_spatial \
    --num_episodes 20 \
    --save_videos
```

### 方式 2: 使用 Shell 脚本（推荐用于复杂参数）

```bash
cd vggt_vla

# 给脚本添加执行权限
chmod +x eval/run_eval.sh

# 使用 run_eval.sh（功能完整）
./eval/run_eval.sh \
    --checkpoint logs/vla_libero_spatial/best_model_libero_spatial_image_20260214_045544_epoch297_step26690_loss0.0017.pt \
    --benchmark libero_spatial \
    --task_ids "0 1 2 3 4" \
    --num_episodes 10

# 快速测试版本
./eval/run_eval.sh \
    --checkpoint logs/vla_libero_spatial/best_model_libero_spatial_image_20260214_045544_epoch297_step26690_loss0.0017.pt \
    --benchmark libero_spatial \
    --task_ids "0" \
    --num_episodes 2 \
    --num_envs 1
```

## 命令行参数详解

### 必需参数

```
--checkpoint PATH
  模型检查点文件路径
  例: logs/vla_libero_spatial/best_model_libero_spatial_image_20260214_045544_epoch297_step26690_loss0.0017.pt
```

### 基准选择

```
--benchmark {libero_spatial | libero_object | libero_goal | libero_10}
  选择要评估的 LIBERO 基准
  - libero_spatial: 空间推理（10个任务）
  - libero_object: 物体识别（10个任务）
  - libero_goal: 目标推理（10个任务）
  - libero_10: 混合任务（10个任务）
```

### 任务选择

```
--task_ids [TASK_ID ...]
  要评估的任务 ID 列表
  例: --task_ids 0 1 2  (评估 Task 0, 1, 2)
  默认: None (评估所有任务)
```

### 评估设置

```
--num_episodes N
  每个任务的评估回合数（默认: 10）
  影响成功率计算的可靠性
  - 用于快速测试: 2-5
  - 用于验证: 10
  - 用于最终报告: 20+

--max_steps N
  每个回合的最大步数（默认: 500）
  LIBERO 环境标准值

--num_envs N
  并行环境数（默认: 20）
  增加加快评估速度（需要更多 GPU 内存）
  减少节省内存
  - 快速测试: 1-5
  - 标准评估: 10-20
  - 高速评估: 20-50
```

### 输出和可视化

```
--save_videos
  保存评估过程中的视频
  视频保存在 output_dir/videos_task_X/ 文件夹

--output_dir PATH
  输出目录（默认: ./eval_results）
  结果保存为 eval_results.json
```

### 其他参数

```
--device {cuda | cpu}
  计算设备（默认: cuda）
```

## 评估工作流

### 步骤 1: 验证环境

```bash
cd vggt_vla

# 运行测试脚本（可选）
python eval/test_eval.py
```

### 步骤 2: 运行评估

```bash
# 方式A: 快速测试（验证脚本工作）
python eval/eval_vla.py \
    --checkpoint logs/vla_libero_spatial/best_model_libero_spatial_image_20260214_045544_epoch297_step26690_loss0.0017.pt \
    --benchmark libero_spatial \
    --task_ids 0 \
    --num_episodes 2 \
    --num_envs 1

# 方式B: 标准评估
python eval/eval_vla.py \
    --checkpoint logs/vla_libero_spatial/best_model_libero_spatial_image_20260214_045544_epoch297_step26690_loss0.0017.pt \
    --benchmark libero_spatial \
    --num_episodes 10
```

### 步骤 3: 查看结果

```bash
# 查看结果文件
cat eval_results/eval_results.json | python -m json.tool

# 查看特定任务的结果
python -c "
import json
with open('eval_results/eval_results.json') as f:
    results = json.load(f)
    print(f\"总成功率: {results['overall_success_rate']:.2%}\")
    for task_name, task_result in results['results'].items():
        print(f\"{task_name}: {task_result['success_rate']:.2%} ({task_result['num_success']}/{task_result['num_episodes']})\")
"
```

## 评估时间估计

基于硬件配置（A100 GPU）和参数设置：

| 配置 | 时间 | 用途 |
|------|------|------|
| 1 任务 × 2 回合 × 1 环境 | ~2-5 分钟 | 快速验证 |
| 3 任务 × 5 回合 × 5 环境 | ~10-15 分钟 | 中等测试 |
| 10 任务 × 10 回合 × 10 环境 | ~30-45 分钟 | 标准评估 |
| 10 任务 × 20 回合 × 20 环境 | ~1.5-2.5 小时 | 完整评估 |

## 输出结果格式

### 目录结构

```
eval_results/
├── eval_results.json           # 结果汇总（JSON）
└── videos_task_0/              # Task 0 的视频（如果指定 --save_videos）
    ├── episode_0.mp4
    ├── episode_1.mp4
    └── ...
```

### JSON 结果格式

```json
{
  "benchmark": "libero_spatial",
  "checkpoint": "logs/vla_libero_spatial/best_model.pt",
  "num_tasks": 10,
  "num_episodes_per_task": 10,
  "overall_success_rate": 0.75,          // 全局成功率
  "total_success": 75,                   // 总成功回合数
  "total_episodes": 100,                 // 总回合数
  "timestamp": "2026-02-20T12:34:56",
  "results": {
    "task_0": {
      "task_id": 0,
      "task_name": "Task description",
      "num_success": 8,
      "num_episodes": 10,
      "success_rate": 0.8,
      "elapsed_time": 234.5,
      "episode_results": [
        {"episode": 0, "success": true, "steps": 234},
        {"episode": 1, "success": false, "steps": 500},
        ...
      ]
    },
    ...
  }
}
```

## 常见问题和解决方案

### Q1: 脚本找不到 LIBERO 模块？

**症状**: `ModuleNotFoundError: No module named 'libero'`

**解决**:
1. 确保 `dataset/LIBERO/libero` 目录存在
2. 从项目根目录运行脚本（即 `cd vggt_vla` 后运行 `python eval_vla.py`）
3. 检查 LIBERO 安装: `ls dataset/LIBERO/libero/`

### Q2: 内存不足（OOM）？

**症状**: `RuntimeError: CUDA out of memory`

**解决**:
1. 减少 `--num_envs`: `--num_envs 5` 或 `--num_envs 1`
2. 减少 `--num_episodes`: `--num_episodes 5`
3. 使用 CPU: `--device cpu`（较慢）

### Q3: 评估很慢？

**症状**: 单个任务需要很长时间

**原因**: 这是正常的。LIBERO 环境模拟机器人动作很耗时。

**优化**:
1. 增加 `--num_envs` (需要更多 GPU 内存)
2. 减少 `--num_episodes` (牺牲准确性)
3. 使用高性能 GPU

### Q4: 检查点加载失败？

**症状**: `FileNotFoundError` 或 `RuntimeError: Unexpected key in state_dict`

**解决**:
1. 检查文件路径是否正确: `ls -la logs/vla_libero_spatial/`
2. 确保检查点文件完整（大小 > 100MB）
3. 检查 GPU 内存是否充足

### Q5: 如何保存评估视频？

**解决**: 使用 `--save_videos` 标志

```bash
python eval_vla.py \
    --checkpoint ... \
    --benchmark libero_spatial \
    --save_videos \
    --output_dir ./eval_results_with_videos
```

## 模型检查点

### 当前可用检查点

在 `vggt_vla/logs/vla_libero_spatial/` 目录下查看：

```bash
ls -lh vggt_vla/logs/vla_libero_spatial/*.pt
```

### 检查点文件命名规则

`best_model_libero_spatial_image_YYYYMMDD_HHMMSS_epochN_stepN_lossX.pt`

| 部分 | 含义 |
|------|------|
| `best_model` | 最佳模型标记 |
| `libero_spatial_image` | 数据集和模式 |
| `YYYYMMDD_HHMMSS` | 保存时间戳 |
| `epochN` | 训练轮数 |
| `stepN` | 训练步数 |
| `lossX` | 验证损失值 |

## 参考资源

### 原始实现
- LIBERO evaluate.py: `dataset/LIBERO/libero/lifelong/evaluate.py`
- VLA 模型: `vggt_vla/models/vla_model.py`

### 相关文档
- VLA 训练: `vggt_vla/scripts/train_vla.py`
- 模型配置: `vggt_vla/configs/model_config.py`
- 数据集: `vggt_vla/data/libero_hf_dataset.py`

## 扩展和定制

### 修改评估逻辑

编辑 `vggt_vla/eval_vla.py` 中的 `VLAEvaluator` 类：

```python
def evaluate_task(self, task_id, ...):
    # 修改此方法以自定义评估逻辑
    pass
```

### 添加新的基准

在 `VLAEvaluator._load_benchmark()` 中添加新的基准映射：

```python
benchmark_map = {
    ...
    "my_custom_benchmark": "MY_CUSTOM_BENCHMARK",
}
```

## 支持和反馈

如有问题或建议，请参考主项目 README。
