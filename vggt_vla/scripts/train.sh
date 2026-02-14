#!/bin/bash
# 通用训练脚本：用 YAML 配置启动 VLA-VGGT 训练
#
# Usage:
#   ./vggt_vla/scripts/train.sh <config>              # 使用 configs/<config>.yaml 或 configs/<config>
#   ./vggt_vla/scripts/train.sh train_whole          # 即 configs/train_whole.yaml
#   ./vggt_vla/scripts/train.sh train_whole.yaml     # 同上
#   ./vggt_vla/scripts/train.sh ./configs/train_whole.yaml   # 显式路径
#   ./vggt_vla/scripts/train.sh train_whole "0,1"    # 指定 GPU
#   ./vggt_vla/scripts/train.sh train_whole "" --batch_size 8   # 覆盖参数
#
# 配置放在 vggt_vla/configs/ 下，例如: train_whole.yaml, train_simple.yaml

set -e

# 脚本所在目录的上级 = vggt_vla
VGGT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
CONFIGS_DIR="${VGGT_ROOT}/configs"

# 第一个参数：配置名或路径
CONFIG_INPUT="${1:-train_simple}"
shift || true

# 解析配置路径
if [[ -f "$CONFIG_INPUT" ]]; then
    CONFIG_PATH="$CONFIG_INPUT"
elif [[ -f "${CONFIGS_DIR}/${CONFIG_INPUT}" ]]; then
    CONFIG_PATH="${CONFIGS_DIR}/${CONFIG_INPUT}"
elif [[ -f "${CONFIGS_DIR}/${CONFIG_INPUT}.yaml" ]]; then
    CONFIG_PATH="${CONFIGS_DIR}/${CONFIG_INPUT}.yaml"
else
    echo "Error: Config not found: $CONFIG_INPUT"
    echo "  Looked at: $CONFIG_INPUT, ${CONFIGS_DIR}/${CONFIG_INPUT}, ${CONFIGS_DIR}/${CONFIG_INPUT}.yaml"
    exit 1
fi

# 可选第二个参数：GPU 列表（多卡 DDP）
GPUS="${1:-}"
if [[ "$GPUS" == --* ]] || [[ -z "$GPUS" ]]; then
    # 第二个参数是 --xxx 或空，则不是 GPUS，插回参数列表
    [[ -n "$GPUS" ]] && set -- "$GPUS" "$@"
    GPUS=""
else
    shift || true
fi

echo "=============================================="
echo "VLA-VGGT Training (generic)"
echo "=============================================="
echo "Config:  $CONFIG_PATH"
echo "Configs: $CONFIGS_DIR"
echo "Root:    $VGGT_ROOT"
[[ -n "$GPUS" ]] && echo "GPUs:    $GPUS (DDP)"
echo "=============================================="
echo ""

# 检查 CUDA（仅当使用 cuda 时）
python -c "
import torch
if not torch.cuda.is_available():
    print('Warning: CUDA not available, training will use CPU.')
else:
    print('CUDA available, devices:', torch.cuda.device_count())
"
echo ""

EXTRA_ARGS=()
[[ -n "$GPUS" ]] && EXTRA_ARGS+=(--gpus "$GPUS")

cd "$VGGT_ROOT"
python "${VGGT_ROOT}/scripts/train_vla.py" --config "$CONFIG_PATH" "${EXTRA_ARGS[@]}" "$@"

echo ""
echo "=============================================="
echo "Done."
echo "=============================================="
