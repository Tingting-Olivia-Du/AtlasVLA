# 评估脚本快速开始

## 文件位置

所有评估脚本都在 **`vggt_vla/eval/`** 文件夹中：

```
vggt_vla/eval/
├── eval_vla.py          # 主评估脚本
├── test_eval.py         # 测试脚本
├── run_eval.sh          # Shell 运行脚本
└── README.md            # eval 模块文档
```

## 最快的使用方式（3 步）

### 1. 进入目录
```bash
cd vggt_vla
```

### 2. 快速验证（~5 分钟）
```bash
# 评估单个任务，验证脚本工作
python eval/eval_vla.py \
    --checkpoint logs/vla_libero_spatial/best_model_libero_spatial_image_20260214_045544_epoch297_step26690_loss0.0017.pt \
    --benchmark libero_spatial \
    --task_ids 0 \
    --num_episodes 2 \
    --num_envs 1
```

### 3. 完整评估（~45 分钟）
```bash
# 评估所有任务
python eval/eval_vla.py \
    --checkpoint logs/vla_libero_spatial/best_model_libero_spatial_image_20260214_045544_epoch297_step26690_loss0.0017.pt \
    --benchmark libero_spatial \
    --num_episodes 10
```

## 使用 Shell 脚本（可选）

```bash
# 添加执行权限（只需一次）
chmod +x vggt_vla/eval/run_eval.sh

# 快速测试
./vggt_vla/eval/run_eval.sh \
    -c logs/vla_libero_spatial/best_model_libero_spatial_image_20260214_045544_epoch297_step26690_loss0.0017.pt \
    -b libero_spatial \
    -t "0" \
    -n 2 \
    -e 1

# 完整评估
./vggt_vla/eval/run_eval.sh \
    -c logs/vla_libero_spatial/best_model_libero_spatial_image_20260214_045544_epoch297_step26690_loss0.0017.pt \
    -b libero_spatial \
    -n 20
```

## 常用命令

### 评估单个任务
```bash
python vggt_vla/eval/eval_vla.py \
    --checkpoint <checkpoint_path> \
    --benchmark libero_spatial \
    --task_ids 0
```

### 评估多个任务
```bash
python vggt_vla/eval/eval_vla.py \
    --checkpoint <checkpoint_path> \
    --benchmark libero_spatial \
    --task_ids 0 1 2 3 4
```

### 保存视频
```bash
python vggt_vla/eval/eval_vla.py \
    --checkpoint <checkpoint_path> \
    --benchmark libero_spatial \
    --save_videos
```

### 快速内存测试
```bash
python vggt_vla/eval/eval_vla.py \
    --checkpoint <checkpoint_path> \
    --benchmark libero_spatial \
    --task_ids 0 \
    --num_episodes 1 \
    --num_envs 1
```

## 参数速查

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--checkpoint PATH` | 检查点文件路径 | 必需 |
| `--benchmark NAME` | 基准名称 | 必需 |
| `--task_ids IDS` | 任务 ID（空格分隔）| 全部 |
| `--num_episodes N` | 每个任务的回合数 | 10 |
| `--num_envs N` | 并行环境数 | 20 |
| `--save_videos` | 保存视频标志 | 无 |
| `--output_dir PATH` | 输出目录 | ./eval_results |

## 基准选择

```
--benchmark libero_spatial   # 空间推理（10个任务）
--benchmark libero_object    # 物体识别（10个任务）
--benchmark libero_goal      # 目标推理（10个任务）
--benchmark libero_10        # 混合任务（10个任务）
```

## 结果查看

评估完成后，结果保存在 `eval_results/eval_results.json`：

```bash
# 查看完整结果
cat eval_results/eval_results.json | python -m json.tool

# 快速查看成功率
python -c "
import json
with open('eval_results/eval_results.json') as f:
    r = json.load(f)
    print(f'总成功率: {r[\"overall_success_rate\"]:.2%}')
    for t, v in r['results'].items():
        print(f'{t}: {v[\"success_rate\"]:.2%}')
"
```

## 问题排查

### 问题：找不到 eval_vla.py
**解决**: 确保在 `vggt_vla` 目录中运行：`cd vggt_vla`

### 问题：检查点加载失败
**解决**: 检查路径是否正确且文件存在

### 问题：内存不足
**解决**: 减少 `--num_envs`，例如 `--num_envs 1`

### 问题：LIBERO 模块错误
**解决**: 检查 `dataset/LIBERO/libero` 是否存在

## 文档

- **快速参考**: `vggt_vla/eval/README.md`
- **详细指南**: `EVAL_README.md`
- **实现细节**: `EVAL_IMPLEMENTATION_SUMMARY.md`
- **使用指南**: `vggt_vla/EVAL_GUIDE.md`

## 下一步

1. ✅ 运行快速验证命令
2. ✅ 检查 `eval_results/eval_results.json` 查看结果
3. ✅ 根据需要调整参数运行完整评估

---

**需要帮助？** 查看 `vggt_vla/eval/README.md` 或 `EVAL_README.md` 获取更多信息。
