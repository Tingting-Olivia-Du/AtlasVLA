#!/bin/bash
# 快速启动训练脚本

set -e

echo "=================================="
echo "VLA-VGGT Quick Start Training"
echo "=================================="

# 设置默认值
CONFIG=${1:-"configs/train_simple.yaml"}
DEVICE=${2:-"cuda:0"}

echo "Config: $CONFIG"
echo "Device: $DEVICE"
echo ""

# 检查CUDA可用性
if [ "$DEVICE" = "cuda" ]; then
    python -c "import torch; assert torch.cuda.is_available(), 'CUDA not available!'"
    echo "✓ CUDA is available"
    python -c "import torch; print(f'  GPU: {torch.cuda.get_device_name(0)}')"
    python -c "import torch; print(f'  Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')"
    echo ""
fi

# 运行训练
echo "Starting training..."
echo ""

python scripts/train_vla.py --config $CONFIG --device $DEVICE

echo ""
echo "=================================="
echo "Training Complete!"
echo "=================================="
