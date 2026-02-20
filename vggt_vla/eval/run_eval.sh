#!/bin/bash
# VLA-VGGT 评估脚本

set -e

# 默认参数
CHECKPOINT=""
BENCHMARK="libero_spatial"
TASK_IDS=""
NUM_EPISODES=10
MAX_STEPS=500
NUM_ENVS=20
SAVE_VIDEOS=false
OUTPUT_DIR="./eval_results"
DEVICE="cuda"

# 帮助信息
usage() {
    cat <<EOF
用法: $0 [选项]

必需参数:
    -c, --checkpoint PATH       模型检查点路径 (必需)

可选参数:
    -b, --benchmark BENCHMARK   LIBERO 基准 (默认: libero_spatial)
                                选择: libero_spatial, libero_object, libero_goal, libero_10
    -t, --task_ids IDS          要评估的任务 ID，用空格分隔 (默认: 全部)
                                例: -t "0 1 2" 或 -t "0 2 4"
    -n, --num_episodes N        每个任务的评估回合数 (默认: 10)
    -m, --max_steps N           每个回合的最大步数 (默认: 500)
    -e, --num_envs N            并行环境数 (默认: 20)
    -v, --save_videos           保存评估视频
    -o, --output_dir DIR        输出目录 (默认: ./eval_results)
    -d, --device DEVICE         计算设备 (默认: cuda)
    -h, --help                  显示本帮助信息

示例:
    # 评估单个任务，保存视频
    $0 -c logs/vla_libero_spatial/best_model.pt -b libero_spatial -t "0" -v

    # 评估所有任务
    $0 -c logs/vla_libero_spatial/best_model.pt -b libero_spatial

    # 快速调试 (少量回合和环境)
    $0 -c logs/vla_libero_spatial/best_model.pt -b libero_spatial -n 2 -e 1

    # 完整评估，保存视频，多任务
    $0 -c logs/vla_libero_spatial/best_model.pt -b libero_spatial -t "0 1 2 3 4" -n 20 -v -o ./full_eval
EOF
    exit 1
}

# 解析参数
while [[ $# -gt 0 ]]; do
    case $1 in
        -c|--checkpoint)
            CHECKPOINT="$2"
            shift 2
            ;;
        -b|--benchmark)
            BENCHMARK="$2"
            shift 2
            ;;
        -t|--task_ids)
            TASK_IDS="$2"
            shift 2
            ;;
        -n|--num_episodes)
            NUM_EPISODES="$2"
            shift 2
            ;;
        -m|--max_steps)
            MAX_STEPS="$2"
            shift 2
            ;;
        -e|--num_envs)
            NUM_ENVS="$2"
            shift 2
            ;;
        -v|--save_videos)
            SAVE_VIDEOS=true
            shift
            ;;
        -o|--output_dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        -d|--device)
            DEVICE="$2"
            shift 2
            ;;
        -h|--help)
            usage
            ;;
        *)
            echo "未知选项: $1"
            usage
            ;;
    esac
done

# 检查必需参数
if [ -z "$CHECKPOINT" ]; then
    echo "错误: 必需指定 --checkpoint"
    usage
fi

# 检查检查点文件是否存在
if [ ! -f "$CHECKPOINT" ]; then
    echo "错误: 检查点文件不存在: $CHECKPOINT"
    exit 1
fi

# 获取脚本所在目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

# 构建命令
CMD="python $SCRIPT_DIR/eval_vla.py"
CMD="$CMD --checkpoint $CHECKPOINT"
CMD="$CMD --benchmark $BENCHMARK"
CMD="$CMD --num_episodes $NUM_EPISODES"
CMD="$CMD --max_steps $MAX_STEPS"
CMD="$CMD --num_envs $NUM_ENVS"
CMD="$CMD --output_dir $OUTPUT_DIR"
CMD="$CMD --device $DEVICE"

if [ ! -z "$TASK_IDS" ]; then
    CMD="$CMD --task_ids $TASK_IDS"
fi

if [ "$SAVE_VIDEOS" = true ]; then
    CMD="$CMD --save_videos"
fi

# 输出命令和参数
echo "=========================================="
echo "VLA-VGGT 评估"
echo "=========================================="
echo "检查点: $CHECKPOINT"
echo "基准: $BENCHMARK"
if [ ! -z "$TASK_IDS" ]; then
    echo "任务: $TASK_IDS"
else
    echo "任务: 全部"
fi
echo "回合数: $NUM_EPISODES"
echo "最大步数: $MAX_STEPS"
echo "并行环境: $NUM_ENVS"
echo "保存视频: $SAVE_VIDEOS"
echo "输出目录: $OUTPUT_DIR"
echo "设备: $DEVICE"
echo "=========================================="
echo ""

# 执行命令
echo "执行命令: $CMD"
echo ""
$CMD
