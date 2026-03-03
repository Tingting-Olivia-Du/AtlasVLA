#!/usr/bin/env python3
"""
在「训练集」上做离线评估：用与训练相同的数据源（HuggingFace + split_seed）加载训练集，
对每个样本跑模型预测，计算与 GT action 的 MSE，用于查看训练集上的拟合效果。

不跑仿真，只做 action 预测误差。与训练脚本一致：数据来自 get_libero_train_dataset（split_seed=42）。

每个 episode 的样本数 = max(1, episode_length - action_horizon)。若要看「完整 episode 成功率」：
  离线：用 --episode_success_threshold T，将「该 episode 内首步 MSE 平均值 < T」的 episode 视为达标，输出达标率。
  在线：需用 eval_vla.py 跑仿真，按 task success 统计。

视频保存与 eval_vla.py 一致：都用 libero VideoWriter（append_image -> save）。区别仅在于：
  - eval_vla：每帧来自仿真 env.step 的 agentview_image（rollout 过程）
  - 本脚本：每帧是数据集中一张静态图 + 文字标注（样本序列），无仿真

用法（在 vggt_vla/ 目录下）:

  python eval/eval_train_set.py --checkpoint ./logs/xxx.pt --gpu 3 --log_dir ./eval_results/train_set --save_vis 50 --vis_dir ./eval_results/train_set/vis
  python eval/eval_train_set.py --checkpoint logs/xxx.pt --dataset_repo lerobot/libero_spatial_image --split_seed 42

python eval/eval_train_set.py --checkpoint /workspace/tingting/AtlasVLA/logs/vla_libero_spatial/best_model_libero_spatial_image_20260214_045544_epoch297_step26690_loss0.0017.pt \
  --gpu 0 \
  --no_multi_view \
  --log_dir ./eval_results/train_set \
  --output_dir ./eval_results/train_set \
  --save_vis 50 \
  --vis_dir ./eval_results/train_set/vis


python eval/eval_train_set.py \
  --checkpoint /workspace/tingting/AtlasVLA/logs/vla_libero_spatial/best_model_libero_spatial_image_20260214_045544_epoch297_step26690_loss0.0017.pt \
  --gpu 0 \
  --no_multi_view \
  --max_samples 500 \
  --save_vis 100 \
  --vis_dir ./eval_results/train_set/vis \
  --log_dir ./eval_results/train_set \
  --output_dir ./eval_results/train_set


python eval/run_offline_and_online.py \
  --checkpoint /workspace/tingting/AtlasVLA/logs/vla_libero_spatial/best_model_libero_spatial_image_20260214_045544_epoch297_step26690_loss0.0017.pt \
  --max_samples 500 \
  --num_episodes 3 \
  --task_ids 0 1 2 \
  --gpu 0



"""

import os
import sys
import json
import argparse
from datetime import datetime

_vggt_vla_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_project_root = os.path.dirname(_vggt_vla_dir)
sys.path.insert(0, _vggt_vla_dir)
sys.path.insert(0, os.path.join(_project_root, "dataset", "LIBERO"))

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from configs.model_config import ModelConfig
from models.vla_model import VLAModel
from data.libero_hf_dataset import get_libero_train_dataset

# 可视化/视频用
IMG_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMG_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def _safe_torch_load(path, map_location=None, **kwargs):
    try:
        return torch.load(path, map_location=map_location, weights_only=False, **kwargs)
    except TypeError:
        return torch.load(path, map_location=map_location, **kwargs)


class Tee:
    """同时写入 stdout 和文件"""
    def __init__(self, log_path):
        self.stdout = sys.stdout
        self.file = open(log_path, "w", encoding="utf-8")
    def write(self, data):
        self.stdout.write(data)
        self.file.write(data)
        self.file.flush()
    def flush(self):
        self.stdout.flush()
        self.file.flush()
    def close(self):
        self.file.close()
        sys.stdout = self.stdout


def _make_vis_frame(image_tensor: torch.Tensor, instruction: str, mse: float,
                    pred_action: np.ndarray, gt_action: np.ndarray, idx: int):
    """生成一帧可视化图像（RGB H,W,3），与 eval_vla 的 VideoWriter.append_image 一致用 RGB。"""
    try:
        import cv2
    except ImportError:
        return None
    if image_tensor.dim() == 4:
        image_tensor = image_tensor[0]
    x = image_tensor.cpu().numpy()
    if x.shape[0] == 3:
        x = np.transpose(x, (1, 2, 0))
    else:
        x = np.transpose(x[0], (1, 2, 0))
    x = (x * IMG_STD + IMG_MEAN).clip(0, 1)
    x = (x * 255).astype(np.uint8)
    bgr = cv2.cvtColor(x, cv2.COLOR_RGB2BGR)
    h, w = bgr.shape[:2]
    lines = [
        f"sample {idx}",
        f"MSE: {mse:.4f}",
        f"instruction: {instruction[:60]}..." if len(instruction) > 60 else f"instruction: {instruction}",
        f"pred[0]: {np.array2string(pred_action[:4], precision=2)}",
        f"gt[0]:   {np.array2string(gt_action[:4], precision=2)}",
    ]
    y0 = h
    for line in lines:
        y0 += 22
        cv2.putText(bgr, line, (5, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def load_model(checkpoint_path: str, device: str = "cuda"):
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint 不存在: {checkpoint_path}")
    checkpoint = _safe_torch_load(checkpoint_path, map_location=device)
    cfg = checkpoint.get("config", None)
    if not isinstance(cfg, ModelConfig):
        cfg = ModelConfig()
    state_dict = (
        checkpoint.get("model_state_dict")
        or checkpoint.get("state_dict")
        or checkpoint
    )
    model = VLAModel(cfg)
    model.load_state_dict(state_dict, strict=False)
    model.to(device).eval()
    return model, cfg


def main():
    parser = argparse.ArgumentParser(description="Eval VLA on training set (offline MSE)")
    parser.add_argument("--checkpoint", type=str, required=True, help="模型 checkpoint 路径")
    parser.add_argument("--dataset_repo", type=str, default="lerobot/libero_spatial_image",
                        help="与训练一致的 HuggingFace 数据集")
    parser.add_argument("--task_indices", type=int, nargs="+", default=None,
                        help="任务索引，与训练一致（默认全部）")
    parser.add_argument("--split_seed", type=int, default=42,
                        help="与训练一致的 episode 划分 seed（训练脚本默认 42）")
    parser.add_argument("--train_split_ratio", type=float, default=0.9,
                        help="训练集比例，与训练一致")
    parser.add_argument("--max_episodes", type=int, default=None, help="最多用多少 episode（调试用）")
    parser.add_argument("--max_samples", type=int, default=None, help="最多评估多少样本（调试用）")
    parser.add_argument("--action_horizon", type=int, default=10, help="与训练一致")
    parser.add_argument("--use_multi_view", action="store_true", default=True,
                        help="使用双视角（与训练一致）")
    parser.add_argument("--no_multi_view", action="store_true", help="单视角")
    parser.add_argument("--batch_size", type=int, default=32, help="评估时 batch size")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--gpu", type=int, default=None,
                        help="指定 GPU 编号，会设置 CUDA_VISIBLE_DEVICES 并使用 cuda:0")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="结果 JSON 保存目录（默认与 checkpoint 同目录）")
    parser.add_argument("--log_dir", type=str, default=None,
                        help="日志目录，将 stdout 同时写入该目录下带时间戳的 .log 文件")
    parser.add_argument("--log_file", type=str, default=None,
                        help="日志文件路径（与 log_dir 二选一，直接指定文件）")
    parser.add_argument("--save_vis", type=int, default=0,
                        help="保存前 N 个样本的可视化图像（图像+指令+MSE+action），用于做视频")
    parser.add_argument("--vis_dir", type=str, default=None,
                        help="可视化图像/视频保存目录（默认 output_dir/vis 或 log_dir/vis）")
    parser.add_argument("--video_fps", type=int, default=2,
                        help="由 save_vis 图像合成视频时的帧率")
    parser.add_argument("--episode_success_threshold", type=float, default=None,
                        help="若指定，按 episode 聚合首步 MSE，平均 MSE 低于该阈值的 episode 视为「达标」，输出 episode 达标率")
    parser.add_argument("--cache_dir", type=str, default=None, help="HuggingFace cache")
    args = parser.parse_args()

    use_multi_view = args.use_multi_view and not args.no_multi_view

    # 指定 GPU（必须在 import 模型前设置 CUDA_VISIBLE_DEVICES）
    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
        args.device = "cuda:0"
        print(f"  [GPU] CUDA_VISIBLE_DEVICES={args.gpu} -> device={args.device}")
    device = args.device  # 始终与 args.device 一致（指定 --gpu 后为 cuda:0）

    # 日志：同时输出到文件
    tee = None
    if args.log_file:
        tee = Tee(os.path.abspath(args.log_file))
        sys.stdout = tee
        print(f"  [Log] 同时写入: {args.log_file}")
    elif args.log_dir:
        os.makedirs(args.log_dir, exist_ok=True)
        log_name = datetime.now().strftime("eval_train_set_%Y%m%d_%H%M%S.log")
        log_path = os.path.join(args.log_dir, log_name)
        tee = Tee(log_path)
        sys.stdout = tee
        print(f"  [Log] 同时写入: {log_path}")

    print("=" * 60)
    print("训练集离线评估 (Train Set Eval)")
    print("=" * 60)
    print(f"  Checkpoint   : {args.checkpoint}")
    print(f"  Dataset      : {args.dataset_repo}")
    print(f"  split_seed    : {args.split_seed} (需与训练时一致)")
    print(f"  multi_view   : {use_multi_view}")
    print("=" * 60)

    if device.startswith("cuda") and not torch.cuda.is_available():
        device = "cpu"
        print("  [提示] CUDA 不可用，使用 CPU")

    model, config = load_model(args.checkpoint, device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  模型参数量: {n_params/1e6:.1f}M\n")

    print("加载训练集（与训练脚本相同数据源 + split_seed）...")
    train_dataset = get_libero_train_dataset(
        repo_id=args.dataset_repo,
        task_indices=args.task_indices,
        max_episodes=args.max_episodes,
        max_samples=args.max_samples,
        action_horizon=args.action_horizon,
        train_split_ratio=args.train_split_ratio,
        split_seed=args.split_seed,
        cache_dir=args.cache_dir,
        use_multi_view=use_multi_view,
        verbose=True,
    )

    loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=(device == "cuda"),
    )

    vis_dir = args.vis_dir
    if args.save_vis > 0 and vis_dir is None:
        vis_dir = os.path.join(args.output_dir or os.path.dirname(os.path.abspath(args.checkpoint)), "vis")
    # 与 eval_vla.py 一致：使用 libero VideoWriter（append_image -> save），只是这里每帧是静态样本图而非仿真 rollout
    video_writer = None
    if args.save_vis > 0 and vis_dir:
        try:
            from libero.libero.utils.video_utils import VideoWriter
            os.makedirs(vis_dir, exist_ok=True)
            video_writer = VideoWriter(vis_dir, save_video=True, fps=args.video_fps, single_video=True)
            print(f"  可视化: 前 {args.save_vis} 帧用 VideoWriter 写入 {vis_dir} (fps={args.video_fps})，与 eval 同流程")
        except Exception as e:
            print(f"  [提示] VideoWriter 不可用，跳过视频: {e}")
            video_writer = None

    mse_first_list = []
    mse_horizon_list = []
    episode_idx_list = []
    n_total = 0
    n_vis_saved = 0

    for batch in tqdm(loader, desc="Eval on train set"):
        images = batch["image"].to(device)
        instructions = [s for s in batch["instruction"]]
        actions_gt = batch["actions"]  # (B, horizon, 7)
        ep_idx = batch.get("episode_idx")
        if ep_idx is not None:
            if torch.is_tensor(ep_idx):
                ep_idx = ep_idx.cpu().numpy()
            else:
                ep_idx = np.array(ep_idx)
            episode_idx_list.append(ep_idx)

        with torch.no_grad():
            out = model(images, instructions, action_normalize=True)
        actions_pred = out["actions"]  # (B, horizon, 7) or (B, 7)

        if actions_pred.dim() == 2:
            actions_pred = actions_pred.unsqueeze(1)
        actions_pred = actions_pred[:, : actions_gt.size(1), :]
        actions_gt = actions_gt.to(device)

        # 首步 MSE
        mse_first = ((actions_pred[:, 0, :] - actions_gt[:, 0, :]) ** 2).mean(dim=1)
        mse_first_list.append(mse_first.cpu().numpy())
        # 整条 horizon MSE（按步平均）
        mse_h = ((actions_pred - actions_gt) ** 2).mean(dim=(1, 2))
        mse_horizon_list.append(mse_h.cpu().numpy())
        n_total += images.size(0)

        # 保存可视化帧（与 eval 相同：VideoWriter.append_image -> 最后 .save()）
        if video_writer is not None and n_vis_saved < args.save_vis:
            mse_first_np = mse_first.cpu().numpy()
            pred_np = actions_pred.cpu().numpy()
            gt_np = actions_gt.cpu().numpy()
            for i in range(images.size(0)):
                if n_vis_saved >= args.save_vis:
                    break
                frame_rgb = _make_vis_frame(
                    images[i], instructions[i],
                    float(mse_first_np[i]),
                    pred_np[i, 0, :], gt_np[i, 0, :],
                    n_vis_saved,
                )
                if frame_rgb is not None:
                    video_writer.append_image(frame_rgb, idx=0)
                n_vis_saved += 1

    mse_first = np.concatenate(mse_first_list)
    mse_horizon = np.concatenate(mse_horizon_list)

    results = {
        "checkpoint": os.path.abspath(args.checkpoint),
        "dataset_repo": args.dataset_repo,
        "split_seed": args.split_seed,
        "n_samples": int(n_total),
        "mse_first_step_mean": float(mse_first.mean()),
        "mse_first_step_std": float(mse_first.std()),
        "mse_first_step_median": float(np.median(mse_first)),
        "mse_horizon_mean": float(mse_horizon.mean()),
        "mse_horizon_std": float(mse_horizon.std()),
        "mse_horizon_median": float(np.median(mse_horizon)),
        "timestamp": datetime.now().isoformat(),
    }

    # 按 episode 聚合：每个 episode 的样本数 = max(1, episode_length - action_horizon)
    if episode_idx_list and args.episode_success_threshold is not None:
        episode_ids = np.concatenate(episode_idx_list)
        from collections import defaultdict
        ep_mse = defaultdict(list)
        for i in range(len(episode_ids)):
            ep_mse[episode_ids[i]].append(mse_first[i])
        per_ep_mean = {ep: np.mean(v) for ep, v in ep_mse.items()}
        n_episodes = len(per_ep_mean)
        n_success = sum(1 for m in per_ep_mean.values() if m < args.episode_success_threshold)
        episode_success_rate = n_success / n_episodes if n_episodes else 0.0
        results["n_episodes"] = n_episodes
        results["episode_success_threshold"] = args.episode_success_threshold
        results["episode_success_count"] = n_success
        results["episode_success_rate"] = float(episode_success_rate)
        results["per_episode_mean_mse_first"] = {int(k): float(v) for k, v in sorted(per_ep_mean.items())}

    print("\n" + "=" * 60)
    print("训练集评估结果")
    print("=" * 60)
    print(f"  样本数         : {results['n_samples']}")
    print(f"  首步 MSE (mean): {results['mse_first_step_mean']:.6f}")
    print(f"  首步 MSE (std) : {results['mse_first_step_std']:.6f}")
    print(f"  Horizon MSE    : {results['mse_horizon_mean']:.6f}")
    if args.episode_success_threshold is not None and "n_episodes" in results:
        print(f"  Episode 数     : {results['n_episodes']}")
        print(f"  Episode 达标率 (首步 MSE < {args.episode_success_threshold}): {results['episode_success_rate']:.2%} ({results['episode_success_count']}/{results['n_episodes']})")
    print("=" * 60)

    out_dir = args.output_dir
    if out_dir is None:
        out_dir = os.path.dirname(os.path.abspath(args.checkpoint))
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "eval_train_set_results.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n结果已保存: {out_path}")

    # 与 eval_vla 同一流程：VideoWriter.save() 写出 mp4（此处为静态样本帧序列，eval 为仿真 rollout）
    if video_writer is not None:
        video_writer.save()

    if tee is not None:
        tee.close()


if __name__ == "__main__":
    main()
