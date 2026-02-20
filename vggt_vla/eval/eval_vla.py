#!/usr/bin/env python3
"""
VLA-VGGT 评估脚本 - 在 LIBERO 环境中评估模型

归一化/反归一化说明:
  - 图像: ImageNet normalization (mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
    与训练时 val_transform 保持一致
  - Action: 无归一化 (训练时直接 MSE, action 为原始值)

用法 (从 vggt_vla/ 目录运行):

cd vggt_vla
python eval/eval_vla.py \
    --checkpoint logs/vla_libero_spatial/best_model_libero_spatial_image_20260214_045544_epoch297_step26690_loss0.0017.pt \
    --benchmark libero_spatial \
    --task_ids 6 --num_episodes 1 --num_envs 1

python eval/eval_vla.py \
    --checkpoint logs/vla_libero_spatial/best_model_libero_spatial_image_20260214_045544_epoch297_step26690_loss0.0017.pt \
    --benchmark libero_spatial \
    --task_ids 8 --num_episodes 1 --num_envs 1 \
    --save_videos --output_dir ./eval_results



参考: dataset/LIBERO/libero/lifelong/evaluate.py
"""

import os
import sys
import json
import argparse
from datetime import datetime

# -------------------------------------------------------------------
# 路径设置（从 vggt_vla/ 目录运行）
# -------------------------------------------------------------------
_vggt_vla_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_project_root  = os.path.dirname(_vggt_vla_dir)

sys.path.insert(0, _vggt_vla_dir)                                        # vggt_vla/ → configs, models
sys.path.insert(0, _project_root)                                         # AtlasVLA/
sys.path.insert(0, os.path.join(_project_root, "dataset", "LIBERO"))     # → import libero.*

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# MuJoCo 使用 osmesa（CPU 软渲染），必须在 import robosuite 之前设置
os.environ["MUJOCO_GL"] = "osmesa"
os.environ.pop("PYOPENGL_PLATFORM", None)
os.environ.pop("MUJOCO_EGL_DEVICE_ID", None)
os.environ.pop("EGL_DEVICE_ID", None)

import numpy as np
import torch
from torchvision import transforms
from tqdm import tqdm

# -------------------------------------------------------------------
# LIBERO 导入
# -------------------------------------------------------------------
try:
    from libero.libero import get_libero_path
    from libero.libero.benchmark import get_benchmark
    from libero.libero.envs import OffScreenRenderEnv, SubprocVectorEnv
    from libero.libero.utils.time_utils import Timer
    from libero.libero.utils.video_utils import VideoWriter
except ImportError as e:
    print(f"[错误] LIBERO 导入失败: {e}")
    print(f"  期望路径: {os.path.join(_project_root, 'dataset', 'LIBERO')}")
    print("  请确保 dataset/LIBERO 目录存在且完整")
    sys.exit(1)

# -------------------------------------------------------------------
# VLA 模型导入
# -------------------------------------------------------------------
from configs.model_config import ModelConfig
from models.vla_model import VLAModel


# -------------------------------------------------------------------
# 图像预处理（与训练时 val_transform 完全一致）
# -------------------------------------------------------------------
# 训练：transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
# Action：无归一化（MSE loss 直接在原始 action 空间训练）
IMG_MEAN = [0.485, 0.456, 0.406]
IMG_STD  = [0.229, 0.224, 0.225]

_img_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMG_MEAN, std=IMG_STD),
])


# -------------------------------------------------------------------
# VLAEvaluator
# -------------------------------------------------------------------
class VLAEvaluator:
    """VLA-VGGT 在 LIBERO 环境中的评估器"""

    BENCHMARK_MAP = {
        "libero_spatial": "LIBERO_SPATIAL",
        "libero_object":  "LIBERO_OBJECT",
        "libero_goal":    "LIBERO_GOAL",
        "libero_10":      "LIBERO_10",
    }

    def __init__(self, checkpoint_path: str, benchmark_name: str, device: str = "cuda"):
        self.checkpoint_path = checkpoint_path
        self.benchmark_name  = benchmark_name
        self.device          = device

        print("=" * 60)
        print("VLAEvaluator 初始化")
        print("=" * 60)
        print(f"  Checkpoint : {checkpoint_path}")
        print(f"  Benchmark  : {benchmark_name}")
        print(f"  Device     : {device}")

        self._load_model()
        self._load_benchmark()
        self._setup_paths()

    # ----------------------------------------------------------------
    # 初始化方法
    # ----------------------------------------------------------------
    def _load_model(self):
        print("\n[1/3] 加载模型 ...")
        if not os.path.exists(self.checkpoint_path):
            raise FileNotFoundError(f"Checkpoint 不存在: {self.checkpoint_path}")

        checkpoint = torch.load(
            self.checkpoint_path, map_location=self.device, weights_only=False
        )

        # config 由 trainer.py 保存为 ModelConfig 对象
        cfg = checkpoint.get("config", None)
        if not isinstance(cfg, ModelConfig):
            print("  ⚠ 未找到 ModelConfig，使用默认配置")
            cfg = ModelConfig()
        self.model_config = cfg

        state_dict = (
            checkpoint.get("model_state_dict")
            or checkpoint.get("state_dict")
            or checkpoint
        )

        self.model = VLAModel(self.model_config)
        missing, unexpected = self.model.load_state_dict(state_dict, strict=False)
        if missing:
            print(f"  ⚠ 缺失 keys ({len(missing)}): {missing[:3]}")
        if unexpected:
            print(f"  ⚠ 多余 keys ({len(unexpected)}): {unexpected[:3]}")

        self.model.to(self.device).eval()
        total = sum(p.numel() for p in self.model.parameters())
        print(f"  ✓ 加载完成 | 参数量: {total/1e6:.1f}M")
        print(f"  图像归一化: mean={IMG_MEAN}  std={IMG_STD}")
        print(f"  Action 归一化: 无（与训练一致）")

    def _load_benchmark(self):
        print("\n[2/3] 加载 LIBERO 基准 ...")
        if self.benchmark_name not in self.BENCHMARK_MAP:
            raise ValueError(
                f"未知基准: {self.benchmark_name}. "
                f"支持: {list(self.BENCHMARK_MAP.keys())}"
            )
        key = self.BENCHMARK_MAP[self.benchmark_name]
        self.benchmark = get_benchmark(key)(0)
        names = self.benchmark.get_task_names()
        print(f"  ✓ {key} | {len(names)} 个任务:")
        for i, n in enumerate(names):
            print(f"    [{i:2d}] {n}")

    def _setup_paths(self):
        print("\n[3/3] 获取 LIBERO 数据路径 ...")
        # 优先使用 dataset/LIBERO/libero/libero 下的数据（标准 LIBERO，非 LIBERO-plus）
        # 避免 ~/.libero/config.yaml 指向其他项目路径的问题
        _libero_pkg = os.path.join(_project_root, "dataset", "LIBERO", "libero", "libero")
        _bddl_local = os.path.join(_libero_pkg, "bddl_files")
        _init_local = os.path.join(_libero_pkg, "init_files")

        if os.path.isdir(_bddl_local) and os.path.isdir(_init_local):
            self.bddl_dir        = _bddl_local
            self.init_states_dir = _init_local
        else:
            # fallback: 读 ~/.libero/config.yaml
            self.bddl_dir        = get_libero_path("bddl_files")
            self.init_states_dir = get_libero_path("init_states")

        # datasets 仍从 config.yaml 读（存放 HDF5 数据集）
        try:
            self.datasets_dir = get_libero_path("datasets")
        except Exception:
            self.datasets_dir = os.path.join(_project_root, "dataset", "LIBERO", "datasets")

        print(f"  datasets   : {self.datasets_dir}")
        print(f"  bddl_files : {self.bddl_dir}")
        print(f"  init_states: {self.init_states_dir}")
        print("\n" + "=" * 60 + "\n")

    # ----------------------------------------------------------------
    # 单任务评估
    # ----------------------------------------------------------------
    def evaluate_task(
        self,
        task_id: int,
        num_episodes: int = 10,
        max_steps: int = 600,
        num_envs: int = 20,
        action_chunk_size: int = 1,
        save_videos: bool = False,
        video_folder: str = None,
    ) -> dict:
        """
        评估单个 LIBERO 任务。

        图像流程:
            LIBERO obs["agentview_image"] (H,W,3 uint8 RGB)
            → ToPILImage → Resize(224) → ToTensor → Normalize(ImageNet)
            → 模型输入 (N, 3, 224, 224)

        Action 流程:
            模型输出 (N, horizon, 7) → 取 t=0 → 直接送入 env.step
            无需反归一化（训练时 action 未归一化）
        """
        task      = self.benchmark.get_task(task_id)
        task_name = task.language

        print(f"\n{'='*60}")
        print(f"Task {task_id}: {task_name}")
        print(f"  回合数={num_episodes}  最大步={max_steps}  并行环境={num_envs}  chunk={action_chunk_size}")
        print(f"{'='*60}")

        env_args = {
            "bddl_file_name": os.path.join(
                self.bddl_dir, task.problem_folder, task.bddl_file
            ),
            "camera_heights": 256,  # 与训练数据（HF lerobot/libero_spatial_image）一致
            "camera_widths":  256,
        }

        init_states_path = os.path.join(
            self.init_states_dir, task.problem_folder, task.init_states_file
        )
        init_states = torch.load(init_states_path, weights_only=False)
        num_init    = init_states.shape[0]

        # 并行环境
        env = SubprocVectorEnv(
            [lambda: OffScreenRenderEnv(**env_args) for _ in range(num_envs)]
        )
        env.seed(42)

        num_success    = 0
        episode_results = []

        # 视频保存：每个 episode 单独保存到 video_folder/task_{task_id}_ep_{ep}.mp4
        if save_videos and video_folder:
            os.makedirs(video_folder, exist_ok=True)

        try:
            with Timer() as timer:
                with torch.no_grad():
                    for ep in tqdm(range(num_episodes), desc=f"  Task {task_id}"):
                        # 初始化环境
                        # 参考 dataset/LIBERO/libero/lifelong/evaluate.py 的写法
                        # set_init_state 需要 numpy array，shape (num_envs, state_dim)
                        indices    = np.arange(num_envs) % num_init
                        # 若只跑1回合，让所有env用同一个init_state (idx=ep%num_init)
                        indices    = (ep + indices) % num_init
                        # init_state = init_states[indices].numpy()  # (num_envs, state_dim)
                        init_state = init_states[indices] 
                        if torch.is_tensor(init_state):
                            init_state = init_state.numpy()
                        obs = env.set_init_state(init_state)

                        # 预模拟 5 步让物理稳定（忽略 done）
                        for _ in range(5):
                            obs, _, _, _ = env.step(np.zeros((num_envs, 7)))

                        dones      = [False] * num_envs
                        ep_success = False
                        steps      = 0

                        # 每个 episode 独立的 VideoWriter
                        if save_videos and video_folder:
                            ep_video_dir = os.path.join(video_folder, f"task{task_id}_ep{ep:03d}")
                            video_writer = VideoWriter(ep_video_dir, save_video=True, fps=30, single_video=False)
                        else:
                            video_writer = None

                        while steps < max_steps:
                            # ---- 图像预处理 ----
                            imgs = [
                                _img_transform(single_obs["agentview_image"])
                                for single_obs in obs
                            ]
                            images = torch.stack(imgs, dim=0).to(self.device)

                            # ---- 模型推断（一次推断，连续执行多步 = action chunking）----
                            instructions = [task_name] * num_envs
                            outputs = self.model(images, instructions)
                            actions = outputs["actions"]  # (num_envs, action_horizon, 7) or (num_envs, 7)

                            # 转为 numpy: shape (num_envs, action_horizon, 7) or (num_envs, 7)
                            if actions.dim() == 3:
                                actions_np = actions.cpu().numpy()          # (B, H, 7)
                                chunk_len  = min(action_chunk_size, actions_np.shape[1])
                            else:
                                actions_np = actions.cpu().numpy()[:, None, :]  # (B, 1, 7)
                                chunk_len  = 1

                            # ---- 连续执行 chunk_len 步 ----
                            for t in range(chunk_len):
                                if steps >= max_steps or all(dones):
                                    break
                                steps += 1

                                action_np = actions_np[:, t, :]  # (B, 7)
                                safe_actions = np.array([
                                    np.zeros(7) if dones[k] else action_np[k]
                                    for k in range(num_envs)
                                ])
                                obs, reward, done, info = env.step(safe_actions)

                                # 视频帧
                                if video_writer is not None:
                                    video_writer.append_vector_obs(
                                        obs, dones, camera_name="agentview_image"
                                    )

                                # 更新完成状态
                                for k in range(num_envs):
                                    if done[k] and not dones[k]:
                                        dones[k]   = True
                                        ep_success = True

                            if all(dones):
                                break

                        # 保存本 episode 视频
                        if video_writer is not None:
                            video_writer.save()

                        if ep_success:
                            num_success += 1

                        episode_results.append({
                            "episode": ep,
                            "success": ep_success,
                            "steps":   steps,
                        })

        finally:
            env.close()

        success_rate = num_success / num_episodes if num_episodes > 0 else 0.0
        elapsed      = timer.get_elapsed_time()

        result = {
            "task_id":         task_id,
            "task_name":       task_name,
            "num_success":     num_success,
            "num_episodes":    num_episodes,
            "success_rate":    success_rate,
            "elapsed_time":    elapsed,
            "episode_results": episode_results,
        }

        print(
            f"  ✓ {num_success}/{num_episodes} = {success_rate*100:.1f}%  "
            f"({elapsed:.1f}s)"
        )
        return result

    # ----------------------------------------------------------------
    # 基准评估（多任务）
    # ----------------------------------------------------------------
    def evaluate_benchmark(
        self,
        task_ids=None,
        num_episodes: int = 10,
        max_steps: int = 600,
        num_envs: int = 20,
        action_chunk_size: int = 1,
        save_videos: bool = False,
        output_dir: str = "./eval_results",
    ) -> dict:
        if task_ids is None:
            task_ids = list(range(len(self.benchmark.get_task_names())))

        os.makedirs(output_dir, exist_ok=True)

        print(f"\n{'='*60}")
        print(f"基准评估: {self.benchmark_name}")
        print(f"  任务={task_ids}  回合/任务={num_episodes}  并行={num_envs}  chunk={action_chunk_size}")
        print(f"  输出: {output_dir}")
        print(f"{'='*60}")

        all_results    = {}
        total_success  = 0
        total_episodes = 0

        for task_id in task_ids:
            vfolder = (
                os.path.join(output_dir, f"videos_task_{task_id}")
                if save_videos else None
            )
            result = self.evaluate_task(
                task_id,
                num_episodes=num_episodes,
                max_steps=max_steps,
                num_envs=num_envs,
                action_chunk_size=action_chunk_size,
                save_videos=save_videos,
                video_folder=vfolder,
            )
            all_results[f"task_{task_id}"] = result
            total_success  += result["num_success"]
            total_episodes += result["num_episodes"]

        overall_rate = total_success / total_episodes if total_episodes > 0 else 0.0

        summary = {
            "benchmark":             self.benchmark_name,
            "checkpoint":            self.checkpoint_path,
            "num_tasks":             len(task_ids),
            "num_episodes_per_task": num_episodes,
            "overall_success_rate":  overall_rate,
            "total_success":         total_success,
            "total_episodes":        total_episodes,
            "timestamp":             datetime.now().isoformat(),
            "results":               all_results,
        }

        # 文件名含时间戳，避免覆盖之前的结果
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        result_file = os.path.join(output_dir, f"eval_results_{ts}.json")
        with open(result_file, "w") as f:
            json.dump(summary, f, indent=2)

        print(f"\n{'='*60}")
        print(f"评估完成 | 总成功率: {overall_rate*100:.1f}%  ({total_success}/{total_episodes})")
        print(f"\n  各任务:")
        for tid in task_ids:
            r = all_results[f"task_{tid}"]
            print(f"    Task {tid:2d}: {r['success_rate']*100:.1f}%  {r['task_name']}")
        print(f"\n  结果保存: {result_file}")
        print(f"{'='*60}\n")

        return summary


# -------------------------------------------------------------------
# 命令行接口
# -------------------------------------------------------------------
def _load_yaml_config(config_path: str) -> dict:
    """加载 eval_config.yaml，返回 dict（缺少 yaml 库时返回空 dict）"""
    try:
        import yaml
        with open(config_path, "r") as f:
            cfg = yaml.safe_load(f) or {}
        print(f"[配置] 读取 {config_path}")
        return cfg
    except ImportError:
        print("[配置] 未安装 pyyaml，跳过 yaml 配置读取")
        return {}
    except FileNotFoundError:
        return {}


def parse_args():
    # 默认 config 路径：eval/eval_config.yaml（与脚本同目录）
    _default_config = os.path.join(os.path.dirname(os.path.abspath(__file__)), "eval_config.yaml")

    parser = argparse.ArgumentParser(
        description="VLA-VGGT LIBERO 评估脚本（从 vggt_vla/ 目录运行）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
优先级: 命令行参数 > eval_config.yaml > 内置默认值

示例:
  # 使用默认 eval_config.yaml
  python eval/eval_vla.py

  # 覆盖 yaml 中的某个参数
  python eval/eval_vla.py --task_ids 0 --num_episodes 2 --num_envs 1

  # 指定其他 config 文件
  python eval/eval_vla.py --config path/to/my_config.yaml
        """,
    )
    parser.add_argument("--config", type=str, default=_default_config,
                        help=f"eval 配置文件路径（默认: eval/eval_config.yaml）")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="检查点路径 (.pt)，覆盖 yaml 中的值")
    parser.add_argument("--benchmark", type=str, default=None,
                        choices=["libero_spatial", "libero_object",
                                 "libero_goal", "libero_10", None],
                        help="LIBERO 基准，覆盖 yaml 中的值")
    parser.add_argument("--task_ids", type=int, nargs="*", default=None,
                        help="任务 ID 列表，覆盖 yaml 中的值")
    parser.add_argument("--num_episodes", type=int, default=None,
                        help="每个任务的回合数，覆盖 yaml 中的值")
    parser.add_argument("--max_steps", type=int, default=None,
                        help="每回合最大步数，覆盖 yaml 中的值")
    parser.add_argument("--num_envs", type=int, default=None,
                        help="并行环境数，覆盖 yaml 中的值")
    parser.add_argument("--action_chunk_size", type=int, default=None,
                        help="action chunking 步数（1=每步重推断, 10=全部chunk执行），覆盖 yaml 中的值")
    parser.add_argument("--save_videos", action="store_true", default=None,
                        help="保存评估视频，覆盖 yaml 中的值")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="输出目录，覆盖 yaml 中的值")
    parser.add_argument("--device", type=str, default=None,
                        help="推理设备，覆盖 yaml 中的值")
    return parser.parse_args()


def main():
    args = parse_args()

    # 读 yaml 配置（内置默认值）
    yaml_cfg = _load_yaml_config(args.config)

    # 命令行参数优先：不为 None 时覆盖 yaml
    def _get(cli_val, yaml_key, default):
        if cli_val is not None:
            return cli_val
        return yaml_cfg.get(yaml_key, default)

    checkpoint   = _get(args.checkpoint,   "checkpoint",   None)
    benchmark    = _get(args.benchmark,    "benchmark",    "libero_spatial")
    task_ids     = _get(args.task_ids,     "task_ids",     None)
    num_episodes = _get(args.num_episodes, "num_episodes", 10)
    max_steps    = _get(args.max_steps,    "max_steps",    600)
    num_envs          = _get(args.num_envs,          "num_envs",          20)
    action_chunk_size = _get(args.action_chunk_size, "action_chunk_size", 1)
    save_videos       = _get(args.save_videos,       "save_videos",       False)
    output_dir   = _get(args.output_dir,   "output_dir",   "./eval_results")
    device       = _get(args.device,       "device",       "cuda")

    if checkpoint is None:
        raise ValueError("必须指定 checkpoint，通过 --checkpoint 或 eval_config.yaml 的 checkpoint 字段")

    evaluator = VLAEvaluator(
        checkpoint_path=checkpoint,
        benchmark_name=benchmark,
        device=device,
    )
    return evaluator.evaluate_benchmark(
        task_ids=task_ids,
        num_episodes=num_episodes,
        max_steps=max_steps,
        num_envs=num_envs,
        action_chunk_size=action_chunk_size,
        save_videos=save_videos,
        output_dir=output_dir,
    )


if __name__ == "__main__":
    main()
