#!/usr/bin/env python3
"""
VLA-VGGT 评估脚本 - **双视角版本**（默认 agentview + wrist）

与 eval_vla.py 相同环境与流程，仅输入图像不同：
  - 本脚本默认使用 **单帧双视角**：agentview_image + robot0_eye_in_hand_image，
    拼成 [B, 2, 3, 224, 224] 送入模型，与训练时 use_multi_view=True 一致。
  - 加 --no_multi_view 则退化为单视角（仅 agentview），与 eval_vla.py 行为一致。

用法 (从 vggt_vla/ 目录运行):

cd vggt_vla
python eval/eval_vla_multiview.py --checkpoint logs/xxx.pt --benchmark libero_spatial --task_ids 0 1
python eval/eval_vla_multiview.py --checkpoint logs/xxx.pt --benchmark libero_spatial --no_multi_view  # 单视角
"""

import os
import sys
import json
import argparse
from datetime import datetime

_vggt_vla_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_project_root  = os.path.dirname(_vggt_vla_dir)

sys.path.insert(0, _vggt_vla_dir)
sys.path.insert(0, _project_root)
sys.path.insert(0, os.path.join(_project_root, "dataset", "LIBERO"))

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["MUJOCO_GL"] = "osmesa"
os.environ.pop("PYOPENGL_PLATFORM", None)
os.environ.pop("MUJOCO_EGL_DEVICE_ID", None)
os.environ.pop("EGL_DEVICE_ID", None)

import numpy as np
import torch
from torchvision import transforms
from tqdm import tqdm

try:
    from libero.libero import get_libero_path
    from libero.libero.benchmark import get_benchmark
    from libero.libero.envs import OffScreenRenderEnv, SubprocVectorEnv, DummyVectorEnv
    from libero.libero.utils.time_utils import Timer
    from libero.libero.utils.video_utils import VideoWriter
except ImportError as e:
    print(f"[错误] LIBERO 导入失败: {e}")
    sys.exit(1)

from configs.model_config import ModelConfig
from models.vla_model import VLAModel

IMG_MEAN = [0.485, 0.456, 0.406]
IMG_STD  = [0.229, 0.224, 0.225]

_img_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMG_MEAN, std=IMG_STD),
])


def _safe_torch_load(path, map_location=None, **kwargs):
    try:
        return torch.load(path, map_location=map_location, weights_only=False, **kwargs)
    except TypeError:
        return torch.load(path, map_location=map_location, **kwargs)


TASK_MAX_STEPS = {
    "libero_spatial": 220,
    "libero_object":  280,
    "libero_goal":    300,
    "libero_10":      520,
    "libero_90":      400,
}


class VLAEvaluatorMultiview:
    """VLA-VGGT 评估器：默认双视角 (agentview + wrist)，与训练 use_multi_view 一致"""

    BENCHMARK_MAP = {
        "libero_spatial": "LIBERO_SPATIAL",
        "libero_object":  "LIBERO_OBJECT",
        "libero_goal":    "LIBERO_GOAL",
        "libero_10":      "LIBERO_10",
        "libero_90":      "LIBERO_90",
    }

    def __init__(self, checkpoint_path: str, benchmark_name: str, device: str = "cuda", use_multi_view: bool = True):
        self.checkpoint_path = checkpoint_path
        self.benchmark_name  = benchmark_name
        self.device          = device
        self.use_multi_view  = use_multi_view

        print("=" * 60)
        print("VLAEvaluator 初始化 (eval_vla_multiview)")
        print("=" * 60)
        print(f"  Checkpoint   : {checkpoint_path}")
        print(f"  Benchmark   : {benchmark_name}")
        print(f"  Device      : {device}")
        print(f"  双视角(默认): {use_multi_view} (agentview + wrist)")
        self._load_model()
        self._load_benchmark()
        self._setup_paths()

    def _load_model(self):
        print("\n[1/3] 加载模型 ...")
        if not os.path.exists(self.checkpoint_path):
            raise FileNotFoundError(f"Checkpoint 不存在: {self.checkpoint_path}")
        checkpoint = _safe_torch_load(self.checkpoint_path, map_location=self.device)
        cfg = checkpoint.get("config", None)
        if not isinstance(cfg, ModelConfig):
            cfg = ModelConfig()
        self.model_config = cfg
        state_dict = checkpoint.get("model_state_dict") or checkpoint.get("state_dict") or checkpoint
        self.model = VLAModel(self.model_config)
        self.model.load_state_dict(state_dict, strict=False)
        self.model.to(self.device).eval()
        total = sum(p.numel() for p in self.model.parameters())
        print(f"  ✓ 加载完成 | 参数量: {total/1e6:.1f}M")

    def _load_benchmark(self):
        print("\n[2/3] 加载 LIBERO 基准 ...")
        if self.benchmark_name not in self.BENCHMARK_MAP:
            raise ValueError(f"未知基准: {self.benchmark_name}. 支持: {list(self.BENCHMARK_MAP.keys())}")
        key = self.BENCHMARK_MAP[self.benchmark_name]
        self.benchmark = get_benchmark(key)(0)
        names = self.benchmark.get_task_names()
        print(f"  ✓ {key} | {len(names)} 个任务")

    def _setup_paths(self):
        print("\n[3/3] 获取 LIBERO 数据路径 ...")
        _libero_pkg = os.path.join(_project_root, "dataset", "LIBERO", "libero", "libero")
        _bddl_local = os.path.join(_libero_pkg, "bddl_files")
        _init_local = os.path.join(_libero_pkg, "init_files")
        if os.path.isdir(_bddl_local) and os.path.isdir(_init_local):
            self.bddl_dir = _bddl_local
            self.init_states_dir = _init_local
        else:
            self.bddl_dir = get_libero_path("bddl_files")
            self.init_states_dir = get_libero_path("init_states")
        try:
            self.datasets_dir = get_libero_path("datasets")
        except Exception:
            self.datasets_dir = os.path.join(_project_root, "dataset", "LIBERO", "datasets")
        print(f"  bddl_files : {self.bddl_dir}\n  init_states: {self.init_states_dir}\n" + "=" * 60 + "\n")

    def _obs_to_images(self, obs, num_envs: int, use_multi_view: bool):
        """从 obs list 构建模型输入：use_multi_view 时 [B,2,3,H,W]，否则 [B,3,H,W]。"""
        if use_multi_view:
            imgs = []
            for single_obs in obs:
                agent = _img_transform(single_obs["agentview_image"])
                wrist = _img_transform(single_obs["robot0_eye_in_hand_image"])
                imgs.append(torch.stack([agent, wrist], dim=0))
            return torch.stack(imgs, dim=0).to(self.device)
        else:
            imgs = [_img_transform(o["agentview_image"]) for o in obs]
            return torch.stack(imgs, dim=0).to(self.device)

    def evaluate_task(
        self,
        task_id: int,
        num_episodes: int = 10,
        max_steps: int = 600,
        num_envs: int = 20,
        action_chunk_size: int = 1,
        save_videos: bool = False,
        video_folder: str = None,
        action_clip: bool = True,
        action_scale: float = 1.0,
        debug_action_stats: bool = False,
        use_multi_view: bool = True,
    ) -> dict:
        task = self.benchmark.get_task(task_id)
        task_name = task.language

        print(f"\n{'='*60}\nTask {task_id}: {task_name}\n  双视角={use_multi_view}  回合={num_episodes}  max_steps={max_steps}  num_envs={num_envs}\n{'='*60}")

        env_args = {
            "bddl_file_name": os.path.join(self.bddl_dir, task.problem_folder, task.bddl_file),
            "camera_heights": 256,
            "camera_widths":  256,
        }
        init_states_path = os.path.join(self.init_states_dir, task.problem_folder, task.init_states_file)
        init_states = _safe_torch_load(init_states_path)
        num_init = init_states.shape[0]

        if num_envs == 1:
            env = DummyVectorEnv([lambda: OffScreenRenderEnv(**env_args) for _ in range(num_envs)])
        else:
            env = SubprocVectorEnv([lambda: OffScreenRenderEnv(**env_args) for _ in range(num_envs)])
        env.seed(42)

        num_success = 0
        episode_results = []
        if save_videos and video_folder:
            os.makedirs(video_folder, exist_ok=True)

        try:
            with Timer() as timer:
                with torch.no_grad():
                    for ep in tqdm(range(num_episodes), desc=f"  Task {task_id}"):
                        indices = (ep + np.arange(num_envs)) % num_init
                        init_state = init_states[indices]
                        if torch.is_tensor(init_state):
                            init_state = init_state.numpy()
                        init_state = np.ascontiguousarray(init_state.astype(np.float64))
                        try:
                            obs = env.set_init_state(init_state)
                        except (BrokenPipeError, EOFError, ConnectionResetError) as e:
                            if num_envs >= 2 and isinstance(env, SubprocVectorEnv):
                                try:
                                    env.close()
                                except Exception:
                                    pass
                                print(f"  [提示] 子进程异常，改用 DummyVectorEnv 继续本 task")
                                env = DummyVectorEnv([lambda: OffScreenRenderEnv(**env_args) for _ in range(num_envs)])
                                env.seed(42)
                                try:
                                    obs = env.set_init_state(init_state)
                                except Exception as e2:
                                    print(f"  [警告] set_init_state 仍失败: {e2}，本 task 跳过")
                                    break
                            else:
                                print(f"  [警告] set_init_state 子进程断开，本 task 跳过")
                                break
                        obs = [obs[i] for i in range(num_envs)]

                        dones = [False] * num_envs
                        try:
                            for _ in range(5):
                                not_done = [k for k in range(num_envs) if not dones[k]]
                                if not not_done:
                                    break
                                obs_partial, _, done_warmup, _ = env.step(np.zeros((len(not_done), 7)), id=not_done)
                                done_warmup = np.atleast_1d(done_warmup)
                                for i, k in enumerate(not_done):
                                    if i < len(done_warmup) and bool(done_warmup[i]):
                                        dones[k] = True
                                    try:
                                        obs[k] = obs_partial[i]
                                    except (TypeError, KeyError):
                                        obs[k] = obs_partial
                        except (BrokenPipeError, EOFError, ConnectionResetError):
                            break
                        obs = [obs[i] for i in range(num_envs)]
                        ep_success = any(dones)
                        steps = 0
                        step_failed = False

                        if save_videos and video_folder:
                            ep_video_dir = os.path.join(video_folder, f"task{task_id}_ep{ep:03d}")
                            video_writer = VideoWriter(ep_video_dir, save_video=True, fps=30, single_video=False)
                        else:
                            video_writer = None

                        while steps < max_steps:
                            images = self._obs_to_images(obs, num_envs, use_multi_view)
                            instructions = [task_name] * num_envs
                            outputs = self.model(images, instructions)
                            actions = outputs["actions"]

                            if actions.dim() == 3:
                                actions_np = actions.cpu().numpy()
                                chunk_len = min(action_chunk_size, actions_np.shape[1])
                                if num_envs > 1:
                                    chunk_len = 1
                            else:
                                actions_np = actions.cpu().numpy()[:, None, :]
                                chunk_len = 1

                            for t in range(chunk_len):
                                if steps >= max_steps or all(dones):
                                    break
                                steps += 1
                                action_np = actions_np[:, t, :].astype(np.float64)
                                if action_scale != 1.0:
                                    action_np = action_np * action_scale
                                if action_clip:
                                    action_np = np.clip(action_np, -1.0, 1.0)
                                if debug_action_stats and ep == 0 and steps == 1:
                                    print(f"  [debug] 第1步 action: min={action_np.min():.4f} max={action_np.max():.4f}")
                                safe_actions = np.array([np.zeros(7) if dones[k] else action_np[k] for k in range(num_envs)])
                                not_done_ids = [k for k in range(num_envs) if not dones[k]]
                                if not not_done_ids:
                                    break
                                actions_for_step = np.array([safe_actions[k] for k in not_done_ids])
                                try:
                                    obs_partial, reward_partial, done_partial, info_partial = env.step(actions_for_step, id=not_done_ids)
                                except (EOFError, ConnectionResetError, BrokenPipeError) as e:
                                    print(f"  [警告] env.step 子进程断开，本 episode 提前结束")
                                    step_failed = True
                                    break
                                done_partial = np.atleast_1d(done_partial)
                                any_just_done = False
                                for i, k in enumerate(not_done_ids):
                                    try:
                                        obs[k] = obs_partial[i]
                                    except (TypeError, KeyError):
                                        obs[k] = obs_partial
                                    if i < len(done_partial) and bool(done_partial[i]) and not dones[k]:
                                        dones[k] = True
                                        ep_success = True
                                        any_just_done = True
                                if any_just_done:
                                    break
                                if video_writer is not None:
                                    video_writer.append_vector_obs(obs, dones, camera_name="agentview_image")

                            if all(dones) or step_failed:
                                break

                        if video_writer is not None:
                            video_writer.save()
                        if ep_success:
                            num_success += 1
                        episode_results.append({"episode": ep, "success": ep_success, "steps": steps})
                        if step_failed:
                            break

        finally:
            try:
                env.close()
            except (ConnectionResetError, EOFError, BrokenPipeError, OSError):
                pass

        success_rate = num_success / num_episodes if num_episodes > 0 else 0.0
        elapsed = timer.get_elapsed_time()
        result = {
            "task_id": task_id,
            "task_name": task_name,
            "num_success": num_success,
            "num_episodes": num_episodes,
            "success_rate": success_rate,
            "elapsed_time": elapsed,
            "episode_results": episode_results,
        }
        print(f"  ✓ {num_success}/{num_episodes} = {success_rate*100:.1f}%  ({elapsed:.1f}s)")
        return result

    def evaluate_benchmark(
        self,
        task_ids=None,
        num_episodes: int = 10,
        max_steps: int = 600,
        num_envs: int = 20,
        action_chunk_size: int = 1,
        save_videos: bool = False,
        output_dir: str = "./eval_results",
        action_clip: bool = True,
        action_scale: float = 1.0,
        debug_action_stats: bool = False,
        use_multi_view: bool = True,
    ) -> dict:
        if task_ids is None:
            task_ids = list(range(len(self.benchmark.get_task_names())))
        if max_steps is None:
            max_steps = TASK_MAX_STEPS.get(self.benchmark_name, 600)
        os.makedirs(output_dir, exist_ok=True)
        print(f"\n{'='*60}\n基准评估: {self.benchmark_name} (双视角={use_multi_view}, max_steps={max_steps})\n  任务={task_ids}  输出={output_dir}\n{'='*60}")

        all_results = {}
        total_success = 0
        total_episodes = 0
        for task_id in task_ids:
            vfolder = os.path.join(output_dir, f"videos_task_{task_id}") if save_videos else None
            result = self.evaluate_task(
                task_id,
                num_episodes=num_episodes,
                max_steps=max_steps,
                num_envs=num_envs,
                action_chunk_size=action_chunk_size,
                save_videos=save_videos,
                video_folder=vfolder,
                action_clip=action_clip,
                action_scale=action_scale,
                debug_action_stats=debug_action_stats,
                use_multi_view=use_multi_view,
            )
            all_results[f"task_{task_id}"] = result
            total_success += result["num_success"]
            total_episodes += result["num_episodes"]

        overall_rate = total_success / total_episodes if total_episodes > 0 else 0.0
        summary = {
            "benchmark": self.benchmark_name,
            "checkpoint": self.checkpoint_path,
            "multi_view": use_multi_view,
            "num_tasks": len(task_ids),
            "num_episodes_per_task": num_episodes,
            "overall_success_rate": overall_rate,
            "total_success": total_success,
            "total_episodes": total_episodes,
            "timestamp": datetime.now().isoformat(),
            "results": all_results,
        }
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        result_file = os.path.join(output_dir, f"eval_multiview_results_{ts}.json")
        with open(result_file, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"\n{'='*60}\n评估完成 | 总成功率: {overall_rate*100:.1f}%  ({total_success}/{total_episodes})\n  结果: {result_file}\n{'='*60}\n")
        return summary


_DEFAULT_CHECKPOINT = "logs/vla_libero_spatial/best_model_libero_spatial_image_20260214_045544_epoch297_step26690_loss0.0017.pt"


def _load_yaml_config(config_path: str) -> dict:
    try:
        import yaml
        with open(config_path, "r") as f:
            return yaml.safe_load(f) or {}
    except (ImportError, FileNotFoundError):
        return {}


def parse_args():
    parser = argparse.ArgumentParser(
        description="VLA-VGGT LIBERO 评估 - 双视角版本 (默认 agentview + wrist)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--checkpoint", type=str, default=None, help=f"默认: {_DEFAULT_CHECKPOINT}")
    parser.add_argument("--benchmark", type=str, default=None,
                        choices=["libero_spatial", "libero_object", "libero_goal", "libero_10", "libero_90", None])
    parser.add_argument("--task_ids", type=int, nargs="*", default=None)
    parser.add_argument("--num_episodes", type=int, default=None)
    parser.add_argument("--max_steps", type=int, default=None)
    parser.add_argument("--num_envs", type=int, default=None)
    parser.add_argument("--action_chunk_size", type=int, default=None)
    parser.add_argument("--save_videos", action="store_true", default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--no_action_clip", action="store_true")
    parser.add_argument("--action_scale", type=float, default=None)
    parser.add_argument("--debug_action_stats", action="store_true")
    parser.add_argument("--no_multi_view", action="store_true", help="禁用双视角，仅用 agentview（与 eval_vla.py 一致）")
    return parser.parse_args()


def main():
    args = parse_args()
    yaml_cfg = _load_yaml_config(args.config) if args.config and os.path.isfile(args.config) else {}

    def _get(cli_val, yaml_key, default):
        return cli_val if cli_val is not None else yaml_cfg.get(yaml_key, default)

    use_multi_view = not getattr(args, "no_multi_view", False)
    checkpoint   = _get(args.checkpoint,   "checkpoint",   _DEFAULT_CHECKPOINT)
    benchmark    = _get(args.benchmark,   "benchmark",    "libero_spatial")
    task_ids     = _get(args.task_ids,    "task_ids",     None)
    num_episodes = _get(args.num_episodes, "num_episodes", 10)
    max_steps    = _get(args.max_steps,   "max_steps",    None)
    num_envs     = _get(args.num_envs,    "num_envs",     1)
    action_chunk_size = _get(args.action_chunk_size, "action_chunk_size", 8)
    save_videos  = _get(args.save_videos, "save_videos",  False)
    output_dir   = _get(args.output_dir,   "output_dir",   "./eval_results")
    device       = _get(args.device,       "device",       "cuda:0")
    action_clip  = False if args.no_action_clip else yaml_cfg.get("action_clip", True)
    action_scale = _get(args.action_scale, "action_scale", 1.0)
    debug_action_stats = getattr(args, "debug_action_stats", False) or yaml_cfg.get("debug_action_stats", False)

    evaluator = VLAEvaluatorMultiview(
        checkpoint_path=checkpoint,
        benchmark_name=benchmark,
        device=device,
        use_multi_view=use_multi_view,
    )
    return evaluator.evaluate_benchmark(
        task_ids=task_ids,
        num_episodes=num_episodes,
        max_steps=max_steps,
        num_envs=num_envs,
        action_chunk_size=action_chunk_size,
        save_videos=save_videos,
        output_dir=output_dir,
        action_clip=action_clip,
        action_scale=action_scale,
        debug_action_stats=debug_action_stats,
        use_multi_view=use_multi_view,
    )


if __name__ == "__main__":
    main()
