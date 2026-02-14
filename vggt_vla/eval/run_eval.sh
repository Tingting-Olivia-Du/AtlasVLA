#!/bin/bash
# VLA-LIBERO 评估脚本
# 用法: ./run_eval.sh [checkpoint_path] [task_ids...]
#   - 不传 checkpoint 时使用默认路径
#   - task_ids 可选，如 0 1 2 只评估指定任务
#
# 环境变量:
#   GPUS: 指定 GPU，如 "0" 或 "0,1,2,3"
#
# 示例:
#   ./run_eval.sh                                    # 用默认 checkpoint 评估全部
#   ./run_eval.sh logs/xxx/best_model_xxx.pth 0 1 2  # 指定 checkpoint 和任务
#   GPUS=1 ./run_eval.sh                              # 使用 GPU 1

set -e

EVAL_DIR="$(cd "$(dirname "$0")" && pwd)"
VGGT_ROOT="$(cd "${EVAL_DIR}/.." && pwd)"

# 默认 checkpoint（可修改）
DEFAULT_CHECKPOINT="${VGGT_ROOT}/logs/vla_libero_spatial/best_model_libero_spatial_image_20260213_212324_epoch15_loss0.0356.pth"

if [ -n "$1" ] && { [[ "$1" == *"/"* ]] || [[ "$1" == *.pth ]] || [[ "$1" == *.pt ]]; }; then
    CHECKPOINT="$1"
    shift
fi
CHECKPOINT=${CHECKPOINT:-"$DEFAULT_CHECKPOINT"}
TASK_IDS="$@"

cd "${VGGT_ROOT}"

# 使用 OSMesa 替代 EGL，避免 EGLGLContext/MjRenderContext 清理错误（参考 openvla-SF）
export MUJOCO_GL="osmesa"
export PYOPENGL_PLATFORM="osmesa"
export NVIDIA_DRIVER_CAPABILITIES="all"

# GPU
GPUS=${GPUS:-${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}}
export CUDA_VISIBLE_DEVICES="$GPUS"
export MUJOCO_EGL_DEVICE_ID=$(echo "$GPUS" | cut -d',' -f1)

# num_procs: 并行 env 数（batch 推理），8x RTX6000 推荐 8
NUM_PROCS=${NUM_PROCS:-8}

export PYTHONPATH="${VGGT_ROOT}:${VGGT_ROOT}/../dataset/LIBERO:${PYTHONPATH}"
[ -z "$LIBERO_CONFIG_PATH" ] && export LIBERO_CONFIG_PATH="${VGGT_ROOT}/../dataset/LIBERO/libero/libero"

# 创建 datasets 目录，消除 LIBERO warning（eval 不用，但 config 会检查）
mkdir -p "${VGGT_ROOT}/../dataset/LIBERO/libero/datasets"

echo "=========================================="
echo "VLA-LIBERO Evaluation"
echo "=========================================="
echo "Checkpoint: $CHECKPOINT"
echo "GPUs: $GPUS"
echo "Task IDs: ${TASK_IDS:-all (0-9)}"
echo "  --n_eval 20: 每任务 20 个 episode"
echo "  --max_steps 600: 每 episode 最多 600 步"
echo "  num_procs: ${NUM_PROCS} (并行 env，batch 推理)"
echo "=========================================="

EXTRA=""
[ -n "$TASK_IDS" ] && EXTRA="--task_ids $TASK_IDS"
[ -n "$GPUS" ] && EXTRA="$EXTRA --gpus $GPUS"
[ -n "$NUM_PROCS" ] && EXTRA="$EXTRA --num_procs $NUM_PROCS"

# config 可选：checkpoint 内含 config 时不需要
CONFIG_ARG=""
[ -f "${VGGT_ROOT}/configs/train_whole.yaml" ] && CONFIG_ARG="--config ${VGGT_ROOT}/configs/train_whole.yaml"

python "${EVAL_DIR}/eval_vla_libero.py" \
    --checkpoint "$CHECKPOINT" \
    $CONFIG_ARG \
    --benchmark libero_spatial \
    --n_eval 20 \
    --max_steps 600 \
    --device cuda:0 \
    $EXTRA

echo "=========================================="
echo "Evaluation complete"
echo "=========================================="
