# VLA-VGGT 评估指南

## 快速开始

### 1. 评估单个任务（快速测试）

```bash
cd vggt_vla

# 评估 Task 0，仅 2 个回合，1 个环境（快速调试）
python eval_vla.py \
    --checkpoint logs/vla_libero_spatial/best_model_libero_spatial_image_20260214_045544_epoch297_step26690_loss0.0017.pt \
    --benchmark libero_spatial \
    --task_ids 0 \
    --num_episodes 2 \
    --num_envs 1
```

### 2. 评估多个任务

```bash
# 评估 Task 0-4，每个任务 10 个回合
python eval_vla.py \
    --checkpoint logs/vla_libero_spatial/best_model_libero_spatial_image_20260214_045544_epoch297_step26690_loss0.0017.pt \
    --benchmark libero_spatial \
    --task_ids 0 1 2 3 4 \
    --num_episodes 10
```

### 3. 完整评估（所有任务）

```bash
# 评估所有 10 个任务，每个任务 20 个回合，保存视频
python eval_vla.py \
    --checkpoint logs/vla_libero_spatial/best_model_libero_spatial_image_20260214_045544_epoch297_step26690_loss0.0017.pt \
    --benchmark libero_spatial \
    --num_episodes 20 \
    --save_videos \
    --output_dir ./eval_results_full
```

### 4. 使用 Shell 脚本运行（推荐）

```bash
# 给脚本添加执行权限
chmod +x scripts/run_eval.sh

# 评估单个任务，保存视频
./scripts/run_eval.sh \
    --checkpoint logs/vla_libero_spatial/best_model_libero_spatial_image_20260214_045544_epoch297_step26690_loss0.0017.pt \
    --benchmark libero_spatial \
    --task_ids "0" \
    --save_videos

# 快速测试
./scripts/run_eval.sh \
    --checkpoint logs/vla_libero_spatial/best_model_libero_spatial_image_20260214_045544_epoch297_step26690_loss0.0017.pt \
    --benchmark libero_spatial \
    --task_ids "0 1 2" \
    --num_episodes 2 \
    --num_envs 1

# 完整评估
./scripts/run_eval.sh \
    --checkpoint logs/vla_libero_spatial/best_model_libero_spatial_image_20260214_045544_epoch297_step26690_loss0.0017.pt \
    --benchmark libero_spatial \
    --num_episodes 20 \
    --save_videos \
    --output_dir ./eval_results
```

## 命令行参数说明

### 必需参数

- `--checkpoint PATH`: 模型检查点路径
  - 例: `logs/vla_libero_spatial/best_model_libero_spatial_image_20260214_045544_epoch297_step26690_loss0.0017.pt`

### 基准选择

- `--benchmark {libero_spatial | libero_object | libero_goal | libero_10}`
  - `libero_spatial`: LIBERO-SPATIAL（10 个空间推理任务）
  - `libero_object`: LIBERO-OBJECT（10 个物体识别任务）
  - `libero_goal`: LIBERO-GOAL（10 个目标推理任务）
  - `libero_10`: LIBERO-10（10 个混合任务）

### 任务和评估设置

- `--task_ids [TASK_ID ...]`: 要评估的任务 ID 列表
  - 默认: `None`（评估所有任务）
  - 例: `0 1 2 3 4` 或 `0 2 4`

- `--num_episodes N`: 每个任务的评估回合数
  - 默认: `10`
  - 用于评估成功率的可靠性
  - 增加此值以获得更可靠的结果

- `--max_steps N`: 每个回合的最大步数
  - 默认: `500`
  - LIBERO 环境的实际最大步数

- `--num_envs N`: 并行环境数
  - 默认: `20`
  - 增加以加快评估速度（如果 GPU 内存允许）
  - 减少以节省内存

### 其他选项

- `--save_videos`: 保存评估视频
  - 视频保存在 `output_dir/videos_task_X/` 文件夹下

- `--output_dir PATH`: 输出目录
  - 默认: `./eval_results`
  - 结果保存为 `eval_results.json`

- `--device {cuda | cpu}`: 计算设备
  - 默认: `cuda`

## 输出结果

评估完成后，结果保存在 `--output_dir` 指定的目录中：

```
eval_results/
├── eval_results.json          # 主结果文件（JSON 格式）
└── videos_task_0/             # Task 0 的视频
    ├── episode_0.mp4
    ├── episode_1.mp4
    └── ...
```

### 结果格式

`eval_results.json` 包含以下信息：

```json
{
  "benchmark": "libero_spatial",
  "checkpoint": "logs/vla_libero_spatial/best_model.pt",
  "num_tasks": 10,
  "num_episodes_per_task": 10,
  "overall_success_rate": 0.75,
  "total_success": 75,
  "total_episodes": 100,
  "timestamp": "2026-02-20T12:34:56.789123",
  "results": {
    "task_0": {
      "task_id": 0,
      "task_name": "task description",
      "num_success": 8,
      "num_episodes": 10,
      "success_rate": 0.8,
      "elapsed_time": 123.45,
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

## 建议的评估策略

### 快速验证（Debug）
```bash
# ~5 分钟
python eval_vla.py \
    --checkpoint logs/vla_libero_spatial/best_model.pt \
    --benchmark libero_spatial \
    --task_ids 0 1 2 \
    --num_episodes 2 \
    --num_envs 1
```

### 中等评估（Validation）
```bash
# ~30 分钟
python eval_vla.py \
    --checkpoint logs/vla_libero_spatial/best_model.pt \
    --benchmark libero_spatial \
    --num_episodes 10 \
    --num_envs 10
```

### 完整评估（Final）
```bash
# ~2-3 小时
python eval_vla.py \
    --checkpoint logs/vla_libero_spatial/best_model.pt \
    --benchmark libero_spatial \
    --num_episodes 20 \
    --num_envs 20 \
    --save_videos
```

## 常见问题

### Q: 评估速度太慢？
A: 增加 `--num_envs` 参数，但要确保 GPU 内存足够。建议开始时使用 `--num_envs 10`。

### Q: 内存不足？
A: 减少 `--num_envs`。建议使用 `--num_envs 5` 或 `--num_envs 1` 进行调试。

### Q: 检查点加载失败？
A: 检查以下几点：
1. 路径是否正确
2. 文件是否存在且完整
3. 检查点格式是否为 `.pt` 或 `.pth`

### Q: 评估很慢？
A: 这是正常的。LIBERO 环境评估每个任务需要数分钟。
- 10 个任务 × 10 回合 × 20 并行环境 ≈ 30-60 分钟

## 参考

- 原始 LIBERO evaluate.py: `dataset/LIBERO/libero/lifelong/evaluate.py`
- VLA 模型代码: `vggt_vla/models/vla_model.py`
- 训练脚本: `vggt_vla/scripts/train_vla.py`
