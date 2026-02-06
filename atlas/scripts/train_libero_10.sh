#!/bin/bash
# 快速启动脚本：使用LIBERO_10数据集训练Atlas模型

set -e  # 遇到错误立即退出

# ==================== 配置区域 ====================
# 请根据你的环境修改以下路径

# GPU设置（使用哪些GPU，例如 "0,1,2,3"）
export CUDA_VISIBLE_DEVICES=0

# LIBERO数据目录（LIBERO下载后的原始数据路径）
# 如果使用默认路径，可以设为空，脚本会自动检测
LIBERO_DATA_DIR=""

# Atlas格式数据输出目录（转换后的数据保存位置）
ATLAS_DATA_DIR="./dataset/libero_10_atlas_format"

# 训练配置文件路径
CONFIG_PATH="atlas/configs/train_config.yaml"

# 是否跳过数据转换（如果数据已转换，设为true）
SKIP_CONVERSION=false

# ==================== 脚本开始 ====================

# 获取脚本所在目录
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/../.." && pwd )"

echo "=========================================="
echo "Atlas LIBERO_10 训练脚本"
echo "=========================================="
echo "项目根目录: $PROJECT_ROOT"
echo "GPU: $CUDA_VISIBLE_DEVICES"
echo ""

# 进入项目根目录
cd "$PROJECT_ROOT"

# 步骤1: 检查LIBERO数据
if [ "$SKIP_CONVERSION" = false ]; then
    echo "步骤1: 检查并转换LIBERO_10数据..."
    
    # 如果没有指定LIBERO数据目录，尝试使用默认路径
    if [ -z "$LIBERO_DATA_DIR" ]; then
        echo "未指定LIBERO数据目录，尝试使用默认路径..."
        # 尝试通过Python获取默认路径
        LIBERO_DATA_DIR=$(python3 -c "
import sys
sys.path.insert(0, 'dataset/LIBERO')
try:
    from libero.libero import get_libero_path
    print(get_libero_path('datasets'))
except:
    print('')
" 2>/dev/null || echo "")
        
        if [ -z "$LIBERO_DATA_DIR" ]; then
            echo "错误: 无法自动检测LIBERO数据目录"
            echo "请手动设置 LIBERO_DATA_DIR 变量"
            exit 1
        fi
    fi
    
    echo "LIBERO数据目录: $LIBERO_DATA_DIR"
    
    # 检查数据是否存在
    if [ ! -d "$LIBERO_DATA_DIR" ]; then
        echo "错误: LIBERO数据目录不存在: $LIBERO_DATA_DIR"
        echo ""
        echo "请先下载LIBERO_100数据集:"
        echo "  cd dataset/LIBERO"
        echo "  python benchmark_scripts/download_libero_datasets.py --datasets libero_100 --use-huggingface"
        exit 1
    fi
    
    # 运行转换脚本
    echo "开始转换数据格式..."
    python3 "$SCRIPT_DIR/convert_libero_to_atlas_format.py" \
        --libero-data-dir "$LIBERO_DATA_DIR" \
        --output-dir "$ATLAS_DATA_DIR" \
        --benchmark libero_10
    
    if [ $? -ne 0 ]; then
        echo "错误: 数据转换失败"
        exit 1
    fi
    
    echo "✓ 数据转换完成"
    echo ""
else
    echo "跳过数据转换（SKIP_CONVERSION=true）"
    echo ""
fi

# 步骤2: 检查转换后的数据
if [ ! -d "$ATLAS_DATA_DIR/train" ]; then
    echo "错误: Atlas格式数据不存在: $ATLAS_DATA_DIR/train"
    echo "请先运行数据转换"
    exit 1
fi

# 统计episode数量
EPISODE_COUNT=$(find "$ATLAS_DATA_DIR/train" -maxdepth 1 -type d -name "episode_*" | wc -l)
echo "找到 $EPISODE_COUNT 个训练episodes"
echo ""

# 步骤3: 更新配置文件中的数据路径
echo "步骤2: 更新配置文件..."
# 使用绝对路径
ABS_ATLAS_DATA_DIR=$(cd "$ATLAS_DATA_DIR" && pwd)

# 备份原配置文件
if [ ! -f "${CONFIG_PATH}.backup" ]; then
    cp "$CONFIG_PATH" "${CONFIG_PATH}.backup"
    echo "已备份原配置文件到 ${CONFIG_PATH}.backup"
fi

# 更新数据路径（使用Python更可靠）
python3 << EOF
import yaml
import os

config_path = "$CONFIG_PATH"
data_dir = "$ABS_ATLAS_DATA_DIR"

# 读取配置
with open(config_path, 'r') as f:
    config = yaml.safe_load(f)

# 更新数据路径
if 'data' not in config:
    config['data'] = {}
config['data']['data_dir'] = data_dir

# 如果没有val_split或val数据不存在，设为null
val_path = os.path.join(data_dir, 'val')
if not os.path.exists(val_path):
    config['data']['val_split'] = None

# 保存配置
with open(config_path, 'w') as f:
    yaml.dump(config, f, default_flow_style=False, allow_unicode=True)

print(f"✓ 已更新数据路径为: {data_dir}")
EOF

echo ""

# 步骤4: 开始训练
echo "步骤3: 开始训练..."
echo "=========================================="
echo ""

# 检查GPU数量
NUM_GPUS=$(echo $CUDA_VISIBLE_DEVICES | tr ',' '\n' | wc -l)

if [ $NUM_GPUS -gt 1 ]; then
    echo "使用 $NUM_GPUS 个GPU进行分布式训练"
    torchrun --nproc_per_node=$NUM_GPUS \
        atlas/train.py \
        --config "$CONFIG_PATH"
else
    echo "使用单GPU训练"
    python3 atlas/train.py \
        --config "$CONFIG_PATH"
fi

echo ""
echo "=========================================="
echo "训练完成！"
echo "Checkpoints保存在: ./checkpoints/"
echo "=========================================="
