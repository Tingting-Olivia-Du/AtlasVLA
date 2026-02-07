#!/bin/bash
# Atlas VLA 统一训练脚本
# 支持HuggingFace数据和本地数据，单GPU和多GPU训练
# conda activate atlas
# Usage: bash atlas/scripts/train.sh

set -e  # 遇到错误立即退出

# ==================== 配置区域 ====================

# GPU设置（使用哪些GPU，例如 "0,1,2,3"）
# 如果为空，使用所有可用GPU
CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"

# WandB设置（可选，如果不设置会使用全局登录）
# 去 https://wandb.ai/authorize 获取你的 API key
WANDB_API_KEY="${WANDB_API_KEY:-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}"  # 设置你的 wandb API key
WANDB_ENTITY="${WANDB_ENTITY:-tingtingdu06-uw-madison}"  # 你的 wandb 用户名

# 训练配置文件路径（相对于项目根目录）
CONFIG_PATH="${CONFIG_PATH:-atlas/configs/train_config.yaml}"

# 是否恢复训练
RESUME_CHECKPOINT="${RESUME_CHECKPOINT:-}"

# 日志文件路径（可选）
LOG_FILE="${LOG_FILE:-}"

# 是否在后台运行
BACKGROUND="${BACKGROUND:-false}"

# ==================== 脚本开始 ====================

# 获取脚本所在目录
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/../.." && pwd )"

echo "=========================================="
echo "Atlas VLA 训练脚本"
echo "=========================================="
echo "项目根目录: $PROJECT_ROOT"
echo "配置文件: $CONFIG_PATH"
echo ""

# 进入项目根目录
cd "$PROJECT_ROOT"

# 检查配置文件是否存在
if [ ! -f "$CONFIG_PATH" ]; then
    echo "错误: 配置文件不存在: $CONFIG_PATH"
    exit 1
fi

# 检查Python环境
if ! command -v python3 &> /dev/null; then
    echo "错误: 未找到python3"
    exit 1
fi

# 检查PyTorch和CUDA
python3 << EOF
import torch
if not torch.cuda.is_available():
    print("警告: CUDA不可用，将使用CPU训练（速度很慢）")
else:
    print(f"✓ CUDA可用，GPU数量: {torch.cuda.device_count()}")
EOF

# 设置GPU
if [ -n "$CUDA_VISIBLE_DEVICES" ]; then
    export CUDA_VISIBLE_DEVICES
    echo "使用GPU: $CUDA_VISIBLE_DEVICES"
    NUM_GPUS=$(echo $CUDA_VISIBLE_DEVICES | tr ',' '\n' | wc -l)
else
    # 自动检测GPU数量
    NUM_GPUS=$(python3 -c "import torch; print(torch.cuda.device_count())" 2>/dev/null || echo "1")
    echo "自动检测到 $NUM_GPUS 个GPU"
fi

# 设置 WandB 环境变量（如果提供）
if [ -n "$WANDB_API_KEY" ]; then
    export WANDB_API_KEY
    echo "✓ WandB API key 已设置"
fi
if [ -n "$WANDB_ENTITY" ]; then
    export WANDB_ENTITY
    echo "✓ WandB entity: $WANDB_ENTITY"
fi

# 准备训练命令参数
TRAIN_ARGS="--config $CONFIG_PATH"

# 添加恢复检查点参数
if [ -n "$RESUME_CHECKPOINT" ]; then
    TRAIN_ARGS="$TRAIN_ARGS --resume $RESUME_CHECKPOINT"
    echo "恢复训练从: $RESUME_CHECKPOINT"
fi

# 多GPU训练
if [ "$NUM_GPUS" -gt 1 ]; then
    echo ""
    echo "使用 $NUM_GPUS 个GPU进行分布式训练"
    # torchrun会自动使用python，不需要python3
    TRAIN_CMD="torchrun --nproc_per_node=$NUM_GPUS atlas/train.py $TRAIN_ARGS"
else
    echo ""
    echo "使用单GPU训练"
    TRAIN_CMD="python3 atlas/train.py $TRAIN_ARGS"
fi

# 日志输出
if [ -n "$LOG_FILE" ]; then
    # 创建日志目录
    LOG_DIR=$(dirname "$LOG_FILE")
    if [ "$LOG_DIR" != "." ] && [ ! -d "$LOG_DIR" ]; then
        mkdir -p "$LOG_DIR"
    fi
    echo "日志文件: $LOG_FILE"
    TRAIN_CMD="$TRAIN_CMD 2>&1 | tee $LOG_FILE"
fi

# 后台运行
if [ "$BACKGROUND" = "true" ]; then
    echo "在后台运行训练..."
    nohup bash -c "$TRAIN_CMD" > /dev/null 2>&1 &
    PID=$!
    echo "训练进程PID: $PID"
    echo "查看日志: tail -f $LOG_FILE"
else
    echo ""
    echo "=========================================="
    echo "开始训练..."
    echo "=========================================="
    echo ""
    
    # 执行训练命令
    eval $TRAIN_CMD
fi

echo ""
echo "=========================================="
echo "训练完成！"
echo "Checkpoints保存在: ./checkpoints/"
echo "=========================================="
