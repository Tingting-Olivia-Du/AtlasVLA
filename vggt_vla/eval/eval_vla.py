#!/usr/bin/env python3
"""
VLA-VGGT 评估脚本 - 在 **原始 LIBERO**（非 LIBERO-Plus）环境中评估模型

重要说明:
  - 本脚本评估的是 **原始 LIBERO**（每 suite 10 个任务），使用 AtlasVLA/dataset/LIBERO。
  - **不需要** clone 或安装 LIBERO-Plus；仅需保证 dataset/LIBERO 下存在 bddl_files、init_files 等。

归一化/反归一化说明:
  - 图像: ImageNet normalization (mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
    与训练时 val_transform 保持一致
  - Action: 无归一化 (训练时直接 MSE, action 为原始值)

Action 语义（LIBERO / OSC_POSE）:
  - 环境使用 robosuite 的 OSC_POSE 控制器，接受的是相对位移 (delta)，不是绝对坐标。
  - 动作空间: 7 维 = (dx, dy, dz, droll, dpitch, dyaw, gripper)，均为相对当前状态的增量。
  - 训练数据 (lerobot/libero_spatial_image) 来自 LIBERO 采集，存的就是 env.step() 接受的 delta。
  - 因此：模型预测的 action 即为 delta，eval 时直接送入 env.step()，无需做 delta↔绝对值的转换。

用法 (从 vggt_vla/ 目录运行):

cd vggt_vla
python eval/eval_vla.py \
    --checkpoint logs/vla_libero_spatial/best_model_libero_spatial_image_20260214_045544_epoch297_step26690_loss0.0017.pt \
    --benchmark libero_spatial \
    --task_ids 6 --num_episodes 10 --num_envs 1

python eval/eval_vla.py \
    --checkpoint logs/vla_libero_spatial/best_model_libero_spatial_image_20260214_045544_epoch297_step26690_loss0.0017.pt \
    --benchmark libero_spatial \
    --num_episodes 1 --num_envs 1 \
    --save_videos --output_dir ./eval_results



参考: dataset/LIBERO/libero/lifelong/evaluate.py 与 Spatial-Forcing run_libero_eval.py（原始 LIBERO 流程）

若 eval 全失败可先排查:
  1. Action 范围: 默认已对 action 做 [-1,1] clip（LIBERO/OSC 常用）；可加 --debug_action_stats 看首步幅度
  2. Action 过小: 若 debug 显示 mean/std 很小可试 --action_scale 1.5 等放大
  3. 图像/指令: 确认与训练一致（agentview、224、ImageNet 归一化；指令为 task.language）
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
    from libero.libero.envs import OffScreenRenderEnv, SubprocVectorEnv, DummyVectorEnv
    from libero.libero.utils.time_utils import Timer
    from libero.libero.utils.video_utils import VideoWriter
except ImportError as e:
    err = str(e)
    print(f"[错误] LIBERO 导入失败: {e}")
    print(f"  期望路径: {os.path.join(_project_root, 'dataset', 'LIBERO')}")
    if "No module named" in err:
        print("  若为缺少依赖，请安装 LIBERO 所需包，例如:")
        print("    pip install bddl==1.0.1   # 若报 No module named 'bddl'")
        print("    pip install robosuite==1.4.0   # 若报 No module named 'robosuite'")
        print("  或一次性安装: pip install -r dataset/LIBERO/requirements.txt")
    else:
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
# LIBERO 使用 OSC_POSE：action 为 delta（相对位移），训练数据与 env.step() 一致，eval 直接送模型输出
IMG_MEAN = [0.485, 0.456, 0.406]
IMG_STD  = [0.229, 0.224, 0.225]

_img_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMG_MEAN, std=IMG_STD),
])


def _safe_torch_load(path, map_location=None, **kwargs):
    """兼容 PyTorch < 2.0：无 weights_only 时省略该参数"""
    try:
        return torch.load(path, map_location=map_location, weights_only=False, **kwargs)
    except TypeError:
        return torch.load(path, map_location=map_location, **kwargs)


# -------------------------------------------------------------------
# VLAEvaluator
# -------------------------------------------------------------------
# 与 Spatial-Forcing run_libero_eval.py 一致：各 suite 推荐 max_steps（原始 LIBERO 最长 demo 步数）
TASK_MAX_STEPS = {
    "libero_spatial": 220,
    "libero_object":  280,
    "libero_goal":    300,
    "libero_10":      520,
    "libero_90":      400,
}


class VLAEvaluator:
    """VLA-VGGT 在原始 LIBERO（非 LIBERO-Plus）环境中的评估器"""

    BENCHMARK_MAP = {
        "libero_spatial": "LIBERO_SPATIAL",
        "libero_object":  "LIBERO_OBJECT",
        "libero_goal":    "LIBERO_GOAL",
        "libero_10":      "LIBERO_10",
        "libero_90":      "LIBERO_90",
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

        checkpoint = _safe_torch_load(
            self.checkpoint_path, map_location=self.device
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
        print("\n[3/3] 获取 LIBERO 数据路径（原始 LIBERO，非 LIBERO-Plus）...")
        # 优先使用 dataset/LIBERO/libero/libero 下的 bddl_files、init_files
        # 避免 ~/.libero/config.yaml 指向其他项目或 LIBERO-Plus 路径
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
        max_steps: int = 220,
        num_envs: int = 1,
        action_chunk_size: int = 8,
        save_videos: bool = False,
        video_folder: str = None,
        action_clip: bool = True,
        action_scale: float = 1.0,
        debug_action_stats: bool = False,
    ) -> dict:
        """
        评估单个 LIBERO 任务。

        图像流程:
            LIBERO obs["agentview_image"] (H,W,3 uint8 RGB)
            → ToPILImage → Resize(224) → ToTensor → Normalize(ImageNet)
            → 模型输入 (N, 3, 224, 224)

        Action 流程:
            模型输出 (N, horizon, 7) 为 delta（与训练数据、OSC_POSE 一致）
            → 可选 scale → 可选 clip [-1,1] → 送入 env.step

        Eval 全失败常见原因与排查:
            1. Action 尺度/范围: 模型输出若超出 [-1,1] 会导致不稳定，默认已做 clip；若动作过小可试 --action_scale
            2. 图像/指令与训练不一致: 确认 camera 为 agentview、Resize 224、ImageNet 归一化；指令用 task.language
            3. Action chunk: 默认每步重推断 (chunk=1)；可试 --action_chunk_size 5 看是否更稳
        """
        task      = self.benchmark.get_task(task_id)
        task_name = task.language

        print(f"\n{'='*60}")
        print(f"Task {task_id}: {task_name}")
        print(f"  回合数={num_episodes}  最大步={max_steps}  并行环境={num_envs}  chunk={action_chunk_size}")
        print(f"  action_clip={action_clip}  action_scale={action_scale}")
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
        init_states = _safe_torch_load(init_states_path)
        num_init    = init_states.shape[0]

        # num_envs==1 时用 DummyVectorEnv 避免子进程 BrokenPipeError（与 dataset/LIBERO/libero/lifelong/metric.py 一致）
        if num_envs == 1:
            env = DummyVectorEnv(
                [lambda: OffScreenRenderEnv(**env_args) for _ in range(num_envs)]
            )
        else:
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
                        # init_state: (num_envs, state_dim)，numpy、C-contiguous 便于子进程序列化
                        init_state = init_states[indices]
                        if torch.is_tensor(init_state):
                            init_state = init_state.numpy()
                        init_state = np.ascontiguousarray(init_state.astype(np.float64))
                        try:
                            obs = env.set_init_state(init_state)
                        except (BrokenPipeError, EOFError, ConnectionResetError) as e:
                            # num_envs>=2 时子进程易崩溃，自动回退到 DummyVectorEnv 继续本 task
                            if num_envs >= 2 and isinstance(env, SubprocVectorEnv):
                                try:
                                    env.close()
                                except Exception:
                                    pass
                                print(f"  [提示] 子进程异常 ({type(e).__name__})，改用 DummyVectorEnv (num_envs={num_envs}) 继续本 task")
                                env = DummyVectorEnv(
                                    [lambda: OffScreenRenderEnv(**env_args) for _ in range(num_envs)]
                                )
                                env.seed(42)
                                try:
                                    obs = env.set_init_state(init_state)
                                except Exception as e2:
                                    print(f"  [警告] DummyVectorEnv set_init_state 仍失败: {e2}，本 task 跳过")
                                    break
                            else:
                                print(f"  [警告] set_init_state 子进程断开 ({type(e).__name__})，本 task 剩余 episode 跳过")
                                break
                        # 统一为 list 以便后续只对未结束的 env step 时按索引合并
                        obs = [obs[i] for i in range(num_envs)]

                        # 预模拟 5 步让物理稳定；只对未结束的 env step，否则会触发 robosuite "executing action in terminated episode"
                        dones = [False] * num_envs
                        try:
                            for _ in range(5):
                                not_done = [k for k in range(num_envs) if not dones[k]]
                                if not not_done:
                                    break
                                obs_partial, _, done_warmup, _ = env.step(
                                    np.zeros((len(not_done), 7)), id=not_done
                                )
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
                        if ep == 0 and num_envs > 1:
                            print(f"  [episode 0] warmup 完成，开始策略步进 (chunk={action_chunk_size})...", flush=True)
                        ep_success  = any(dones)  # warmup 中已有 env 成功则记为成功
                        steps       = 0
                        step_failed = False  # 若 env.step 子进程断开则置 True，跳出本 episode

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
                                # 多 env 时每轮只执行 1 步，避免 step 后 dones 未及时更新导致对已终止 env 再 step
                                if num_envs > 1:
                                    chunk_len = 1
                            else:
                                actions_np = actions.cpu().numpy()[:, None, :]  # (B, 1, 7)
                                chunk_len  = 1

                            # ---- 连续执行 chunk_len 步 ----
                            for t in range(chunk_len):
                                if steps >= max_steps or all(dones):
                                    break
                                steps += 1

                                action_np = actions_np[:, t, :].astype(np.float64)  # (B, 7) delta for OSC_POSE
                                if action_scale != 1.0:
                                    action_np = action_np * action_scale
                                if action_clip:
                                    action_np = np.clip(action_np, -1.0, 1.0)
                                if debug_action_stats and ep == 0 and steps == 1:
                                    print(f"  [debug] 第1步 action: min={action_np.min():.4f} max={action_np.max():.4f} mean={action_np.mean():.4f} std={action_np.std():.4f}")
                                safe_actions = np.array([
                                    np.zeros(7) if dones[k] else action_np[k]
                                    for k in range(num_envs)
                                ])

                                # 只对未结束的 env 调用 step，避免 robosuite "executing action in terminated episode"
                                not_done_ids = [k for k in range(num_envs) if not dones[k]]
                                if not not_done_ids:
                                    break
                                actions_for_step = np.array([safe_actions[k] for k in not_done_ids])
                                try:
                                    obs_partial, reward_partial, done_partial, info_partial = env.step(
                                        actions_for_step, id=not_done_ids
                                    )
                                except (EOFError, ConnectionResetError, BrokenPipeError) as e:
                                    print(f"  [警告] env.step 子进程断开 ({type(e).__name__})，本 episode 提前结束")
                                    step_failed = True
                                    break
                                # 合并回完整 obs，并正确对应 done（done_partial 与 not_done_ids 顺序一致）
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
                                # 有 env 本步刚结束则跳出 chunk，下一轮重新推理，避免对已终止 env 再 step
                                if any_just_done:
                                    break

                                # 视频帧（传入 list obs）
                                if video_writer is not None:
                                    video_writer.append_vector_obs(
                                        obs, dones, camera_name="agentview_image"
                                    )

                            if all(dones) or step_failed:
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

                        if step_failed:
                            break  # env 子进程已断，不再用该 env 跑后续 episode，直接进 finally 关环境

        finally:
            try:
                env.close()
            except (ConnectionResetError, EOFError, BrokenPipeError, OSError) as e:
                # 子进程已退出或连接已断开时 close() 可能报错，忽略以便正常输出评估结果
                print(f"  [提示] 关闭环境时忽略: {type(e).__name__}: {e}")

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
        action_clip: bool = True,
        action_scale: float = 1.0,
        debug_action_stats: bool = False,
    ) -> dict:
        if task_ids is None:
            task_ids = list(range(len(self.benchmark.get_task_names())))

        # 默认使用该 suite 推荐 max_steps（与 SF run_libero_eval 一致）
        if max_steps is None:
            max_steps = TASK_MAX_STEPS.get(self.benchmark_name, 600)

        os.makedirs(output_dir, exist_ok=True)

        print(f"\n{'='*60}")
        print(f"基准评估: {self.benchmark_name} (原始 LIBERO, max_steps={max_steps})")
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
                action_clip=action_clip,
                action_scale=action_scale,
                debug_action_stats=debug_action_stats,
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
    """加载 yaml 配置文件，返回 dict（缺少 yaml 库或文件不存在时返回空 dict）"""
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


# 默认与 run_eval.sh 保持一致（唯一配置入口）
_DEFAULT_CHECKPOINT = "logs/vla_libero_spatial/best_model_libero_spatial_image_20260214_045544_epoch297_step26690_loss0.0017.pt"


def parse_args():
    parser = argparse.ArgumentParser(
        description="VLA-VGGT LIBERO 评估脚本（从 vggt_vla/ 目录运行，或由 eval/run_eval.sh 调用）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
配置由 run_eval.sh 或本脚本命令行指定，无 yaml。示例:
  python eval/eval_vla.py
  python eval/eval_vla.py --checkpoint logs/xxx.pt --task_ids 0 1 --num_episodes 5
        """,
    )
    parser.add_argument("--config", type=str, default=None,
                        help="可选：yaml 配置文件路径（若存在则读取，与 CLI 合并）")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help=f"检查点路径 (默认: {_DEFAULT_CHECKPOINT})")
    parser.add_argument("--benchmark", type=str, default=None,
                        choices=["libero_spatial", "libero_object",
                                 "libero_goal", "libero_10", "libero_90", None],
                        help="原始 LIBERO 基准 (默认: libero_spatial)")
    parser.add_argument("--task_ids", type=int, nargs="*", default=None,
                        help="任务 ID 列表，不传则全部任务")
    parser.add_argument("--num_episodes", type=int, default=None,
                        help="每任务回合数 (默认: 10)")
    parser.add_argument("--max_steps", type=int, default=None,
                        help="每回合最大步数 (默认: 按 suite，如 spatial=220)")
    parser.add_argument("--num_envs", type=int, default=None,
                        help="并行环境数 (默认: 1)")
    parser.add_argument("--action_chunk_size", type=int, default=None,
                        help="action chunk 步数 (默认: 8)")
    parser.add_argument("--save_videos", action="store_true", default=None,
                        help="保存评估视频")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="输出目录 (默认: ./eval_results)")
    parser.add_argument("--device", type=str, default=None,
                        help="推理设备 (默认: cuda:1)")
    parser.add_argument("--no_action_clip", action="store_true",
                        help="不对 action 做 [-1,1] 裁剪（默认会裁剪）")
    parser.add_argument("--action_scale", type=float, default=None,
                        help="action 乘数，例如 1.5 放大动作幅度（默认 1.0）")
    parser.add_argument("--debug_action_stats", action="store_true",
                        help="打印第 1 步 action 的 min/max/mean/std 便于排查")
    return parser.parse_args()


def main():
    args = parse_args()

    # 若指定了 --config 且文件存在，则读取 yaml（否则为空）；与 run_eval.sh 一致时可不传 config
    yaml_cfg = _load_yaml_config(args.config) if args.config and os.path.isfile(args.config) else {}

    def _get(cli_val, yaml_key, default):
        if cli_val is not None:
            return cli_val
        return yaml_cfg.get(yaml_key, default)

    checkpoint   = _get(args.checkpoint,   "checkpoint",   _DEFAULT_CHECKPOINT)
    benchmark    = _get(args.benchmark,    "benchmark",    "libero_spatial")
    task_ids     = _get(args.task_ids,     "task_ids",     None)
    num_episodes = _get(args.num_episodes, "num_episodes", 10)
    max_steps    = _get(args.max_steps,    "max_steps",    None)  # None → 在 evaluate_benchmark 中按 suite 设 TASK_MAX_STEPS
    num_envs          = _get(args.num_envs,          "num_envs",          1)
    action_chunk_size = _get(args.action_chunk_size, "action_chunk_size", 8)
    save_videos       = _get(args.save_videos,       "save_videos",       True)
    output_dir   = _get(args.output_dir,   "output_dir",   "./eval_results")
    device       = _get(args.device,       "device",       "cuda:1")
    action_clip  = False if args.no_action_clip else yaml_cfg.get("action_clip", True)
    action_scale = _get(args.action_scale, "action_scale", 1.0)
    debug_action_stats = getattr(args, "debug_action_stats", False) or yaml_cfg.get("debug_action_stats", False)

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
        action_clip=action_clip,
        action_scale=action_scale,
        debug_action_stats=debug_action_stats,
    )


if __name__ == "__main__":
    main()
