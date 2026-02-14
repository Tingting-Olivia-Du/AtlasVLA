# VLA-LIBERO 评估

在 LIBERO MuJoCo 仿真环境中评估 VLA 模型的 success rate。

## 依赖

1. 安装 LIBERO: `cd dataset/LIBERO && pip install -e .`
2. 下载 LIBERO 数据集和资源 (bddl_files, init_states 等)
3. 配置 `~/.libero/config.yaml` 中的路径

## 用法

### 命令行

```bash
# 在 vggt_vla 目录下运行
cd vggt_vla

# 评估完整 checkpoint
python eval/eval_vla_libero.py \
    --checkpoint logs/vla_libero_spatial/best_model_libero_spatial_image_20260213_212324_epoch15_loss0.0356.pth \
    --config configs/train_whole.yaml \
    --benchmark libero_spatial \
    --n_eval 20

# 只评估指定任务 (task 0, 1, 2)
python eval/eval_vla_libero.py \
    --checkpoint logs/vla_libero_spatial/best_model_xxx.pth \
    --config configs/train_whole.yaml \
    --task_ids 0 1 2
```

### Shell 脚本

```bash
./eval/run_eval.sh                                    # 用默认 checkpoint 评估全部
./eval/run_eval.sh logs/xxx/best_model_xxx.pth        # 指定 checkpoint
./eval/run_eval.sh logs/xxx/best_model_xxx.pth 0 1 2  # 只评估 task 0,1,2
GPUS=1 ./eval/run_eval.sh                             # 使用 GPU 1
```

默认 checkpoint 已写在 `run_eval.sh` 中，可修改 `DEFAULT_CHECKPOINT` 变量。

### 参数

| 参数 | 说明 | 默认 |
|------|------|------|
| `--checkpoint` | 模型 checkpoint 路径 (.pth 或 .pt) | 必填 |
| `--config` | 训练配置 yaml (**checkpoint 内含 config 时不需要**) | None |
| `--benchmark` | LIBERO 任务集 | libero_spatial |
| `--task_ids` | 要评估的任务 ID | 全部 (0-9) |
| `--n_eval` | 每任务评估 episode 数（每个任务跑多少次仿真） | 20 |
| `--max_steps` | 每个 episode 最多步数（超过则视为失败） | 600 |
| `--gpus` | 多卡并行: 如 `0,1,2,3,4,5,6,7` 将任务分配到多 GPU 并行 | None |
| `--device` | 设备 | cuda:0 |
| `--output_dir` | 结果保存目录 | checkpoint 同目录 |

## 输出

- JSON 结果文件: `eval_<checkpoint_stem>_<timestamp>.json`
- 包含各任务 success rate 及平均值
