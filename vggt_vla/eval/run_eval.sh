#!/bin/bash
# VLA-VGGT 评估脚本
#
# 默认多卡并行（GPUS=0,1,2,3）。直接运行: cd vggt_vla && ./eval/run_eval.sh
# 单卡时请指定: -g "" 或修改下方 GPUS 为空并设 DEVICE。
# 配置以本脚本为准，无 yaml。
set -e

# 默认参数（与 eval_vla.py 内置默认一致，改此处即可）
# 使用相对路径，适配 atlas / 任意 workspace
CHECKPOINT="../logs/vla_libero_spatial/best_model_libero_spatial_image_20260214_045544_epoch297_step26690_loss0.0017.pt"
BENCHMARK="libero_spatial"
TASK_IDS=""
NUM_EPISODES=5
MAX_STEPS=220
# 并行环境数：模型一次前向 batch=num_envs，显存大(如 140G)可设 20–32；受 CPU/子进程数限制不宜过大
NUM_ENVS=20
SAVE_VIDEOS=true
OUTPUT_DIR="eval_results"   # 实际输出为 eval_results/<BENCHMARK>/（见下方）
DEVICE="cuda"
# 默认多卡；设为空则单卡（用 DEVICE）
GPUS="3"

# 帮助信息
usage() {
    cat <<EOF
用法: $0 [选项]

可选参数:
    -c, --checkpoint PATH       模型检查点路径 (默认见脚本顶部)
    -b, --benchmark BENCHMARK   LIBERO 基准 (默认: libero_spatial)
                                选择: libero_spatial, libero_object, libero_goal, libero_10
    -t, --task_ids IDS          要评估的任务 ID，用空格分隔 (默认: 全部)
                                例: -t "0 1 2" 或 -t "0 1 2 3 4 5 6 7 8 9"
    -n, --num_episodes N        每个任务的评估回合数 (默认: 10)
    -m, --max_steps N           每个回合的最大步数 (默认: 500)
    -e, --num_envs N            并行环境数 (默认: 20)
    -v, --save_videos           保存评估视频
    -o, --output_dir DIR        输出目录 (默认: ./eval_results)
    -d, --device DEVICE         计算设备，单卡时使用 (默认: cuda)，例: cuda:0
    -g, --gpus GPUS             多卡并行时使用的 GPU 编号，逗号分隔 (例: 0,1,2,3 或 2,5,7)
                                指定后会把 -t 的任务按 GPU 数分片并行跑；未指定 -t 时请先指定 -t
    -h, --help                  显示本帮助信息

示例:
    # 单卡，指定用第 2 张 GPU
    $0 -c logs/best_model.pt -d cuda:2

    # 用 GPU 0,1,2,3 并行评估任务 0~7（每卡 2 个任务）
    $0 -c logs/best_model.pt -t "0 1 2 3 4 5 6 7" -g 0,1,2,3

    # 用 8 张卡跑全部 10 个任务（任务 0~7 分到 8 卡，任务 8 9 在卡 0 1）
    $0 -c logs/best_model.pt -t "0 1 2 3 4 5 6 7 8 9" -g 0,1,2,3,4,5,6,7

    # 评估单个任务，保存视频
    $0 -c logs/best_model.pt -b libero_spatial -t "0" -v
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
        -g|--gpus)
            GPUS="$2"
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

# 获取脚本所在目录（vggt_vla/eval/run_eval.sh → PROJECT_DIR=vggt_vla）
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

# 检查点：相对路径视为相对于 PROJECT_DIR (vggt_vla)，便于 atlas 等任意 workspace
if [ -f "$PROJECT_DIR/$CHECKPOINT" ]; then
    CHECKPOINT="$PROJECT_DIR/$CHECKPOINT"
elif [ -f "$CHECKPOINT" ]; then
    :  # 绝对路径且存在，保持
else
    echo "错误: 检查点文件不存在: $CHECKPOINT (也未找到 $PROJECT_DIR/$CHECKPOINT)"
    echo "  请用 -c 指定或修改脚本顶部 CHECKPOINT 默认值"
    exit 1
fi
# 传给 Python 时使用绝对路径（已在上面的 if 中归一化）
case "$CHECKPOINT" in
    /*) ;;
    *) CHECKPOINT="$PROJECT_DIR/$CHECKPOINT"
esac

cd "$PROJECT_DIR"

# 不同 suite 存到不同子目录：eval_results/libero_spatial/、eval_results/libero_object/ 等
OUTPUT_DIR="${OUTPUT_DIR}/${BENCHMARK}"

# 构建单次运行的基础命令。参数: $1=可选 output_dir，默认 OUTPUT_DIR
base_cmd() {
    local out="${1:-$OUTPUT_DIR}"
    echo "python $SCRIPT_DIR/eval_vla.py --checkpoint $CHECKPOINT --benchmark $BENCHMARK --num_episodes $NUM_EPISODES --max_steps $MAX_STEPS --num_envs $NUM_ENVS --output_dir $out"
}

# 多卡并行：按 GPU 分片任务，每卡使用独立输出子目录避免写同一文件
if [ -n "$GPUS" ]; then
    # 解析 GPU 列表
    IFS=',' read -ra GPU_ARR <<< "$GPUS"
    NGPU=${#GPU_ARR[@]}
    if [ $NGPU -eq 0 ]; then
        echo "错误: -g/--gpus 需要至少一个 GPU 编号，例如 -g 0,1,2,3"
        exit 1
    fi
    # 解析任务列表：未指定 -t 时默认跑全部 10 个任务 (libero_spatial)，再按 GPU 分片
    if [ -z "$TASK_IDS" ]; then
        TASK_IDS="0 1 2 3 4 5 6 7 8 9"
        echo "未指定 -t，默认全部任务: $TASK_IDS"
    fi
    read -ra TASK_ARR <<< "$TASK_IDS"
    NTASKS=${#TASK_ARR[@]}
    # 把任务尽量均匀分到各 GPU；每卡输出到 OUTPUT_DIR/gpu_<id> 避免多进程写同一目录
    PIDS=()
    for ((i=0; i<NGPU; i++)); do
        gpu_id="${GPU_ARR[$i]}"
        # 任务索引: i, i+NGPU, i+2*NGPU, ...
        sub_tasks=()
        for ((j=i; j<NTASKS; j+=NGPU)); do
            sub_tasks+=("${TASK_ARR[$j]}")
        done
        [ ${#sub_tasks[@]} -eq 0 ] && continue
        sub_tids="${sub_tasks[*]}"
        out_dir="${OUTPUT_DIR}/gpu_${gpu_id}"
        export CUDA_VISIBLE_DEVICES="$gpu_id"
        CMD="$(base_cmd "$out_dir") --task_ids $sub_tids --device cuda:0"
        [ "$SAVE_VIDEOS" = true ] && CMD="$CMD --save_videos"
        echo "[GPU $gpu_id] 任务: $sub_tids 输出: $out_dir"
        $CMD &
        PIDS+=($!)
    done
    echo "=========================================="
    echo "等待 ${#PIDS[@]} 个进程结束..."
    echo "结果目录: ${OUTPUT_DIR}/gpu_0, ${OUTPUT_DIR}/gpu_1, ..."
    echo "=========================================="
    for p in "${PIDS[@]}"; do
        wait $p
    done
    echo "全部完成. 各卡结果见 ${OUTPUT_DIR}/gpu_<id>/eval_results_*.json"
    exit 0
fi

# 单卡：原有逻辑
CMD="$(base_cmd) --device $DEVICE"
if [ -n "$TASK_IDS" ]; then
    CMD="$CMD --task_ids $TASK_IDS"
fi
if [ "$SAVE_VIDEOS" = true ]; then
    CMD="$CMD --save_videos"
fi

echo "=========================================="
echo "VLA-VGGT 评估"
echo "=========================================="
echo "检查点: $CHECKPOINT"
echo "基准: $BENCHMARK"
if [ -n "$TASK_IDS" ]; then
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
echo "执行命令: $CMD"
echo ""
$CMD
