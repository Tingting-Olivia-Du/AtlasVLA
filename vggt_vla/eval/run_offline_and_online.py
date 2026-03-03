#!/usr/bin/env python3
"""
一键跑「训练集离线 MSE」+「在线仿真成功率」，方便对比：训练误差有多大、仿真里能不能成功。

流程:
  1. 运行 eval_train_set.py → 得到训练集上的首步 MSE（离线 error）
  2. 运行 eval_vla.py → 在 LIBERO 仿真里 rollout，得到各任务成功率（在线 success rate）
  3. 打印对比摘要

用法（在 vggt_vla/ 目录下）:

  # 快速小样本：500 样本离线 + 每任务 3 回合、任务 0 1 2 在线
  python eval/run_offline_and_online.py --checkpoint logs/xxx.pt --max_samples 500 --num_episodes 3 --task_ids 0 1 2

  # 只跑在线仿真（不看离线 MSE）
  python eval/run_offline_and_online.py --checkpoint logs/xxx.pt --no_offline --num_episodes 5 --task_ids 0 1 2 3 4

  # 只跑离线（不看在线）
  python eval/run_offline_and_online.py --checkpoint logs/xxx.pt --no_online --max_samples 1000

在线仿真依赖: dataset/LIBERO、robosuite、bddl 等（与 eval_vla.py / run_eval.sh 一致）。
多卡多任务可用: ./eval/run_eval.sh -c logs/xxx.pt -t "0 1 2 3 4 5 6 7 8 9" -n 10 -g 0,1,2,3
"""

import os
import sys
import json
import argparse
import subprocess
import glob

_vggt_vla_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(_vggt_vla_dir)
sys.path.insert(0, _vggt_vla_dir)


def run_offline(checkpoint: str, output_base: str, gpu: int = None, max_samples: int = 500, extra_args: list = None):
    cmd = [
        sys.executable,
        "eval/eval_train_set.py",
        "--checkpoint", checkpoint,
        "--output_dir", os.path.join(output_base, "offline"),
        "--log_dir", os.path.join(output_base, "offline"),
        "--max_samples", str(max_samples),
        "--no_multi_view",
    ]
    if gpu is not None:
        cmd += ["--gpu", str(gpu)]
    if extra_args:
        cmd += extra_args
    print("[离线] 运行:", " ".join(cmd))
    subprocess.run(cmd, check=True)
    p = os.path.join(output_base, "offline", "eval_train_set_results.json")
    if not os.path.isfile(p):
        raise FileNotFoundError(f"离线结果未生成: {p}")
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


def run_online(checkpoint: str, output_base: str, gpu: int = None,
               num_episodes: int = 5, task_ids: list = None, max_steps: int = 220,
               num_envs: int = 1, extra_args: list = None):
    out_dir = os.path.join(output_base, "online")
    cmd = [
        sys.executable,
        "eval/eval_vla.py",
        "--checkpoint", checkpoint,
        "--benchmark", "libero_spatial",
        "--output_dir", out_dir,
        "--num_episodes", str(num_episodes),
        "--max_steps", str(max_steps),
        "--num_envs", str(num_envs),
    ]
    if task_ids is not None and len(task_ids) > 0:
        cmd += ["--task_ids"] + [str(t) for t in task_ids]
    if gpu is not None:
        cmd += ["--device", f"cuda:{gpu}"]
    if extra_args:
        cmd += extra_args
    print("[在线] 运行:", " ".join(cmd))
    subprocess.run(cmd, check=True)
    # 结果文件: eval_results_<timestamp>.json
    pattern = os.path.join(out_dir, "eval_results_*.json")
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"在线结果未找到: {pattern}")
    with open(files[-1], "r", encoding="utf-8") as f:
        return json.load(f)


def main():
    parser = argparse.ArgumentParser(description="离线 MSE + 在线仿真成功率 一键对比")
    parser.add_argument("--checkpoint", type=str, required=True, help="模型 checkpoint 路径")
    parser.add_argument("--output_dir", type=str, default="./eval_results/compare",
                        help="结果目录，其下 offline/ 与 online/ 分别存两次运行结果")
    parser.add_argument("--gpu", type=int, default=None, help="使用的 GPU 编号（离线/在线共用）")
    parser.add_argument("--max_samples", type=int, default=500,
                        help="离线评估最多样本数（小样本可设 200~500）")
    parser.add_argument("--num_episodes", type=int, default=5, help="在线每任务回合数")
    parser.add_argument("--task_ids", type=int, nargs="*", default=[0, 1, 2],
                        help="在线评估的任务 ID 列表，默认 0 1 2")
    parser.add_argument("--max_steps", type=int, default=220, help="在线每回合最大步数")
    parser.add_argument("--num_envs", type=int, default=1, help="在线并行环境数")
    parser.add_argument("--no_offline", action="store_true", help="不跑离线，只跑在线仿真")
    parser.add_argument("--no_online", action="store_true", help="不跑在线，只跑离线 MSE")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    offline_data = None
    online_data = None

    if not args.no_offline:
        offline_data = run_offline(
            args.checkpoint,
            args.output_dir,
            gpu=args.gpu,
            max_samples=args.max_samples,
        )
    if not args.no_online:
        online_data = run_online(
            args.checkpoint,
            args.output_dir,
            gpu=args.gpu,
            num_episodes=args.num_episodes,
            task_ids=args.task_ids,
            max_steps=args.max_steps,
            num_envs=args.num_envs,
        )

    # 打印对比
    print("\n" + "=" * 60)
    print("对比摘要：训练误差 vs 仿真成功率")
    print("=" * 60)
    if offline_data:
        n = offline_data.get("n_samples", 0)
        mse_mean = offline_data.get("mse_first_step_mean")
        mse_median = offline_data.get("mse_first_step_median")
        print(f"  [离线] 训练集首步 MSE  (n={n})  mean={mse_mean:.6f}  median={mse_median:.6f}")
    if online_data:
        total = online_data.get("total_episodes", 0)
        success = online_data.get("total_success", 0)
        rate = online_data.get("overall_success_rate", 0.0)
        print(f"  [在线] 仿真成功率  {success}/{total} = {rate*100:.1f}%")
        if online_data.get("results"):
            print("  各任务:")
            for k, v in online_data["results"].items():
                if isinstance(v, dict) and "success_rate" in v:
                    print(f"    {k}: {v['num_success']}/{v['num_episodes']} = {v['success_rate']*100:.1f}%")
    print("=" * 60)


if __name__ == "__main__":
    main()
