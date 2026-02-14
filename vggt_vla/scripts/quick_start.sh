#!/bin/bash
# 快速启动训练脚本
# Usage:
#   ./vggt_vla/scripts/quick_start.sh ./vggt_vla/configs/train_whole.yaml   # 双卡 GPU 0,1
#   ./vggt_vla/scripts/quick_start.sh ./vggt_vla/configs/train_whole.yaml cuda "0,1" \
#       --resume logs/vla_libero_spatial/best_model_xxx.pth --wandb_resume   # 从 checkpoint 继续，续写 wandb

set -e

echo "=================================="
echo "VLA-VGGT Quick Start Training"
echo "=================================="

# 脚本所在目录的上级 = vggt_vla
VGGT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
CONFIG=${1:-"${VGGT_ROOT}/configs/train_simple.yaml"}
DEVICE=${2:-"cuda"}
GPUS=${3:-""}

echo "Config: $CONFIG"
echo "Device: $DEVICE"
[ -n "$GPUS" ] && echo "GPUs: $GPUS (multi-GPU DDP)"
echo ""

# 检查 CUDA 可用性
if [ "$DEVICE" = "cuda" ] || [ "$DEVICE" = "cuda:0" ]; then
    python -c "import torch; assert torch.cuda.is_available(), 'CUDA not available!'"
    echo "✓ CUDA is available"
    python -c "import torch; print(f'  GPUs: {torch.cuda.device_count()}')"
    echo ""
fi

# 运行训练
echo "Starting training..."
echo ""

EXTRA_ARGS=""
[ -n "$GPUS" ] && EXTRA_ARGS="--gpus $GPUS"
# 第4个及之后的参数传给 train_vla.py（如 --resume path --wandb_resume）
python "${VGGT_ROOT}/scripts/train_vla.py" --config "$CONFIG" --device "$DEVICE" $EXTRA_ARGS "${@:4}"

echo ""
echo "=================================="
echo "Training Complete!"
echo "=================================="
