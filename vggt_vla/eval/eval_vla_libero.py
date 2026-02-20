#!/usr/bin/env python3
"""
VLA 模型在 LIBERO 仿真环境中的评估脚本

评估类似 best_model_libero_spatial_image_20260213_212324_epoch15_loss0.0356.pth 的模型
在 LIBERO-Spatial 任务上的 success rate。
"""
import os
import sys
import argparse
import json
import numpy as np
import torch
from pathlib import Path
from datetime import datetime

# wandb 可选：未安装时仅禁用 wandb 日志
try:
    import wandb
    _HAS_WANDB = True
except ImportError:
    _HAS_WANDB = False

# 添加 vggt_vla 根目录
VGGT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(VGGT_ROOT))

# 添加 LIBERO 目录
LIBERO_ROOT = VGGT_ROOT.parent / "dataset" / "LIBERO"
sys.path.insert(0, str(LIBERO_ROOT))

# 导入 VideoWriter
try:
    from libero.libero.utils.video_utils import VideoWriter
    _HAS_VIDEO_WRITER = True
except ImportError:
    _HAS_VIDEO_WRITER = False


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate VLA model on LIBERO simulation")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to model checkpoint (.pth or .pt)")
    parser.add_argument("--config", type=str, default=None,
                        help="Path to train config yaml (可选，checkpoint 内含 config 时不需要)")
    parser.add_argument("--benchmark", type=str, default="libero_spatial",
                        choices=["libero_spatial", "libero_object", "libero_goal", "libero_10"])
    parser.add_argument("--task_ids", type=int, nargs="+", default=None,
                        help="Task IDs to evaluate (default: all 0-9)")
    parser.add_argument("--n_eval", type=int, default=20,
                        help="Number of evaluation episodes per task")
    parser.add_argument("--max_steps", type=int, default=600,
                        help="Max steps per episode")
    parser.add_argument("--camera_h", type=int, default=224,
                        help="Camera height (VLA expects 224)")
    parser.add_argument("--camera_w", type=int, default=224,
                        help="Camera width")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--gpus", type=str, default=None,
                        help="多卡并行: 如 '0,1,2,3,4,5,6,7' 将任务分配到多 GPU 并行评估")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Directory to save results (default: same dir as checkpoint)")
    parser.add_argument("--num_procs", type=int, default=8,
                        help="Number of parallel envs for VectorEnv (batch inference, default 8 for 8xRTX6000)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--use_wandb", action="store_true",
                        help="Enable Weights & Biases logging")
    parser.add_argument("--wandb_project", type=str, default="vla-vggt-libero-eval",
                        help="W&B project name")
    parser.add_argument("--wandb_entity", type=str, default=None,
                        help="W&B entity/username (default: use default from wandb login)")
    parser.add_argument("--wandb_run_name", type=str, default=None,
                        help="W&B run name (default: auto-generated)")
    parser.add_argument("--log_dir", type=str, default=None,
                        help="Directory to save log file (default: same as output_dir)")
    parser.add_argument("--save_videos", action="store_true",
                        help="Save evaluation videos (episodes will be recorded)")
    parser.add_argument("--video_fps", type=int, default=30,
                        help="FPS for saved videos (default: 30)")
    return parser.parse_args()


def _config_from_dict(d):
    """从 dict 重建 ModelConfig"""
    from configs.model_config import ModelConfig, VisionConfig, LanguageConfig, VGGTConfig, ActionHeadConfig
    v = d.get("vision") or {}
    l = d.get("language") or {}
    vg = d.get("vggt") or {}
    ah = d.get("action_head") or {}
    if isinstance(v, dict):
        vision = VisionConfig(**{k: v for k, v in v.items() if k in VisionConfig.__dataclass_fields__})
    else:
        vision = v
    if isinstance(l, dict):
        language = LanguageConfig(**{k: v for k, v in l.items() if k in LanguageConfig.__dataclass_fields__})
    else:
        language = l
    if isinstance(vg, dict):
        vggt = VGGTConfig(**{k: v for k, v in vg.items() if k in VGGTConfig.__dataclass_fields__})
    else:
        vggt = vg
    if isinstance(ah, dict):
        action_head = ActionHeadConfig(**{k: v for k, v in ah.items() if k in ActionHeadConfig.__dataclass_fields__})
    else:
        action_head = ah
    return ModelConfig(vision=vision, language=language, vggt=vggt, action_head=action_head, hidden_dim=d.get("hidden_dim", 768))


def load_model_and_config(checkpoint_path, config_path, device):
    """加载 VLA 模型和配置"""
    import yaml
    from configs.model_config import ModelConfig
    from models.vla_model import VLAModel

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if isinstance(ckpt, dict):
        state_dict = ckpt.get("model_state_dict") or ckpt.get("state_dict")
        saved_config = ckpt.get("config")
        if state_dict is None:
            raise ValueError(f"No model_state_dict or state_dict in checkpoint {checkpoint_path}")
    else:
        state_dict = ckpt
        saved_config = None

    if saved_config is not None:
        if hasattr(saved_config, "vision"):
            config = saved_config
        elif isinstance(saved_config, dict):
            config = _config_from_dict(saved_config)
        else:
            config = saved_config
    elif config_path and os.path.exists(config_path):
        with open(config_path) as f:
            cfg = yaml.safe_load(f)
        from types import SimpleNamespace
        args = SimpleNamespace()
        defaults = {"use_vision_tower": False, "vision_tower_name": "facebook/dinov2-base",
            "freeze_vision_tower": True, "language_model": "Qwen/Qwen3-0.6B-Base", "freeze_language": True,
            "use_pretrained_vggt": True, "freeze_vggt": False, "action_horizon": 10, "action_dim": 7}
        for k, v in {**defaults, **cfg}.items():
            setattr(args, k, v)
        from scripts.train_vla import build_config
        config = build_config(args)
    else:
        raise ValueError("No config in checkpoint. Please provide --config with train yaml path.")

    model = VLAModel(config)

    # 尝试严格加载，如果失败则使用宽松模式
    try:
        model.load_state_dict(state_dict, strict=True)
        print(f"  ✓ Loaded all weights from checkpoint (strict mode)")
    except RuntimeError as e:
        error_msg = str(e)
        if 'size mismatch' in error_msg or 'Missing' in error_msg:
            print(f"  ⚠ Size mismatch detected: {error_msg[:150]}...")
            print(f"  Loading with strict=False, skipping mismatched weights...")

            incompatible = model.load_state_dict(state_dict, strict=False)
            print(f"\n  Incompatible keys:")
            print(f"    - Missing keys: {len(incompatible.missing_keys)}")
            if incompatible.missing_keys:
                for k in incompatible.missing_keys[:3]:
                    print(f"      {k}")
                if len(incompatible.missing_keys) > 3:
                    print(f"      ... and {len(incompatible.missing_keys)-3} more")

            print(f"    - Unexpected keys: {len(incompatible.unexpected_keys)}")
            if incompatible.unexpected_keys:
                for k in incompatible.unexpected_keys[:3]:
                    print(f"      {k}")
                if len(incompatible.unexpected_keys) > 3:
                    print(f"      ... and {len(incompatible.unexpected_keys)-3} more")

            print(f"\n  ⚠ WARNING: Some weights could not be loaded.")
            print(f"  The model will use randomly initialized weights for mismatched layers.")
            print(f"  This may affect evaluation performance.")
        else:
            raise

    model = model.to(device)
    model.eval()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Loaded {n_params:,} params from checkpoint")
    return model, config


def preprocess_image(img, device, img_size=224):
    """将 env 图像预处理为 VLA 输入格式"""
    import torchvision.transforms as T
    # img: [H,W,3] uint8 or float, BGR or RGB
    if img.dtype != np.uint8:
        img = (img * 255).astype(np.uint8) if img.max() <= 1.0 else img.astype(np.uint8)
    transform = T.Compose([
        T.ToPILImage(),
        T.Resize((img_size, img_size)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    x = transform(img)
    return x.unsqueeze(0).float().to(device)


def preprocess_image_batch(imgs, device, img_size=224):
    """Batch 预处理多张图像 [N,H,W,3] -> [N,3,H,W]"""
    import torchvision.transforms as T
    transform = T.Compose([
        T.ToPILImage(),
        T.Resize((img_size, img_size)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    out = []
    for i in range(len(imgs)):
        img = imgs[i]
        if img.dtype != np.uint8:
            img = (img * 255).astype(np.uint8) if img.max() <= 1.0 else img.astype(np.uint8)
        out.append(transform(img))
    return torch.stack(out).float().to(device)


def evaluate_one_task(model, task, bddl_path, init_states_path, instruction, n_eval, max_steps,
                      camera_h, camera_w, device, seed, video_writer=None):
    """评估单个任务的 success rate
    
    Args:
        video_writer: VideoWriter 实例，如果提供则保存视频
    """
    import gc
    import sys
    from libero.libero.envs import OffScreenRenderEnv

    print("    Creating env (osmesa may be slow on first run)...", flush=True)
    sys.stdout.flush()
    env_args = {
        "bddl_file_name": bddl_path,
        "camera_heights": camera_h,
        "camera_widths": camera_w,
    }
    env = OffScreenRenderEnv(**env_args)
    env.seed(seed)

    try:
        # pruned_init 含 NumPy 数组，PyTorch 2.6+ 需 weights_only=False
        init_states = torch.load(init_states_path, weights_only=False)
    except Exception as e:
        raise RuntimeError(f"Failed to load init states from {init_states_path}: {e}")

    num_success = 0

    for ep in range(n_eval):
        if ep == 0:
            print("    Starting episodes...", flush=True)
        env.reset()
        idx = ep % init_states.shape[0]
        env.set_init_state(init_states[idx])

        # 初始几步让物理稳定
        dummy = np.zeros(7)
        obs, _, _, _ = env.step(dummy)
        if video_writer:
            video_writer.append_obs(obs, False, idx=ep, camera_name="agentview_image")
        for _ in range(4):
            obs, _, _, _ = env.step(dummy)
            if video_writer:
                video_writer.append_obs(obs, False, idx=ep, camera_name="agentview_image")

        done = False
        for step in range(max_steps - 5):
            o = obs[0] if isinstance(obs, (list, tuple)) and len(obs) > 0 else obs
            img = o.get("agentview_image", o.get("agentview_rgb"))
            img_tensor = preprocess_image(img, device, camera_h)

            with torch.no_grad():
                actions = model.predict_action(img_tensor, [instruction])

            # 调试：在第一个 episode 的前 3 步打印详细信息
            if ep == 0 and step < 3:
                print(f"      [Step {step}] actions shape: {actions.shape}, dtype: {actions.dtype}")
                print(f"               range: [{actions.min():.4f}, {actions.max():.4f}], mean: {actions.mean():.4f}")

            if actions.dim() == 3:
                actions = actions[:, 0, :]
            act = actions.cpu().numpy()[0]

            if ep == 0 and step < 3:
                print(f"               final action: {act}")

            obs, reward, done, info = env.step(act)

            if ep == 0 and step < 3:
                print(f"               reward: {reward}, done: {done}")
            # 保存视频帧
            if video_writer:
                video_writer.append_obs(obs, done, idx=ep, camera_name="agentview_image")
            
            # 处理不同格式的 done（单环境可能返回标量、list 或 array）
            if isinstance(done, (list, tuple)) and len(done) > 0:
                done = bool(done[0])
            elif isinstance(done, np.ndarray):
                done = bool(done.flat[0] if done.size > 0 else False)
            else:
                done = bool(done)
            if done:
                num_success += 1
                break

        if (ep + 1) % 5 == 0 or ep == 0:
            print(f"    Episode {ep+1}/{n_eval}, success so far: {num_success}", flush=True)

    try:
        env.close()
    except Exception:
        pass
    del env
    gc.collect()
    return num_success / n_eval


def evaluate_one_task_vector(model, task, bddl_path, init_states_path, instruction, n_eval, max_steps,
                             camera_h, camera_w, device, seed, num_procs, use_dummy_vector_env=False, video_writer=None):
    """使用 VectorEnv 并行评估单个任务（batch 推理加速）
    
    Args:
        use_dummy_vector_env: 如果 True，强制使用 DummyVectorEnv（用于多进程 worker 中）
        video_writer: VideoWriter 实例，如果提供则保存视频
    """
    import gc
    import sys
    import multiprocessing as mp
    from libero.libero.envs import OffScreenRenderEnv, SubprocVectorEnv, DummyVectorEnv

    env_num = min(num_procs, n_eval)
    eval_loop_num = (n_eval + env_num - 1) // env_num

    # 检测是否在多进程 worker 中（参考 LIBERO 官方做法）
    # 如果在 multiprocessing worker 中，daemonic 进程不能创建子进程，必须使用 DummyVectorEnv
    is_mp_worker = mp.current_process().name != 'MainProcess'
    force_dummy = use_dummy_vector_env or is_mp_worker

    print(f"    Creating {env_num} parallel envs ({'DummyVectorEnv' if force_dummy or env_num == 1 else 'SubprocVectorEnv'})...", flush=True)
    sys.stdout.flush()
    env_args = {
        "bddl_file_name": bddl_path,
        "camera_heights": camera_h,
        "camera_widths": camera_w,
    }
    
    # 参考 LIBERO 官方：重试机制处理环境创建失败
    env_creation = False
    count = 0
    while not env_creation and count < 5:
        try:
            if env_num == 1 or force_dummy:
                env = DummyVectorEnv([lambda: OffScreenRenderEnv(**env_args) for _ in range(env_num)])
            else:
                env = SubprocVectorEnv([lambda: OffScreenRenderEnv(**env_args) for _ in range(env_num)])
            env_creation = True
        except Exception as e:
            if count < 4:
                import time
                print(f"    [WARN] VectorEnv creation failed (attempt {count+1}/5), retrying...", flush=True)
                time.sleep(2)
                count += 1
            else:
                print(f"    [WARN] VectorEnv failed after 5 attempts, falling back to single env: {e}", flush=True)
                return evaluate_one_task(model, task, bddl_path, init_states_path, instruction, n_eval,
                                        max_steps, camera_h, camera_w, device, seed)

    try:
        init_states = torch.load(init_states_path, weights_only=False)
    except Exception as e:
        raise RuntimeError(f"Failed to load init states from {init_states_path}: {e}")

    env.seed(seed)
    num_success = 0

    for i in range(eval_loop_num):
        if i == 0:
            print("    Starting VectorEnv episodes...", flush=True)
        env.reset()
        indices = np.arange(i * env_num, (i + 1) * env_num) % init_states.shape[0]
        init_states_ = init_states[indices]  # always env_num

        obs = env.set_init_state(init_states_)
        dummy = np.zeros((env_num, 7))
        for _ in range(5):
            obs, _, _, _ = env.step(dummy)
            if video_writer:
                video_writer.append_vector_obs(obs, [False] * env_num, camera_name="agentview_image")

        dones = [False] * env_num
        steps = 0

        while steps < max_steps - 5:
            steps += 1
            # obs: list of dicts or stacked; VectorEnv returns (obs_stack, ...)
            if isinstance(obs, np.ndarray) and obs.dtype == object:
                obs_list = [obs[k] for k in range(env_num)]
            elif isinstance(obs, (list, tuple)):
                obs_list = obs[:env_num] if len(obs) > env_num else obs
            else:
                obs_list = [obs] if env_num == 1 else [obs[k] for k in range(env_num)]

            imgs = []
            for o in obs_list:
                oo = o[0] if isinstance(o, (list, tuple)) and len(o) > 0 else o
                if isinstance(oo, dict):
                    img = oo.get("agentview_image", oo.get("agentview_rgb"))
                else:
                    img = oo
                imgs.append(img)

            img_batch = preprocess_image_batch(imgs, device, camera_h)
            instructions = [instruction] * env_num

            with torch.no_grad():
                actions = model.predict_action(img_batch, instructions)
            if actions.dim() == 3:
                actions = actions[:, 0, :]
            act_np = actions.cpu().numpy()

            obs, _, done, _ = env.step(act_np)
            # 保存视频帧
            if video_writer:
                video_writer.append_vector_obs(obs, dones, camera_name="agentview_image")
            # VectorEnv 返回的 done 可能是 numpy array 或 list
            if isinstance(done, np.ndarray):
                # 确保 done 是一维数组
                done_flat = done.flatten() if done.ndim > 1 else done
                for k in range(min(len(done_flat), env_num)):
                    dones[k] = dones[k] or bool(done_flat[k])
            elif isinstance(done, (list, tuple)):
                for k in range(min(len(done), env_num)):
                    dones[k] = dones[k] or bool(done[k])
            else:
                # 单个值，应用到所有环境（不应该发生，但容错处理）
                dones[0] = dones[0] or bool(done)

            if all(dones):
                break

        for k in range(env_num):
            if i * env_num + k < n_eval:
                num_success += int(dones[k])

        if (i + 1) * env_num <= n_eval or (i + 1) % 2 == 0:
            print(f"    Batch {i+1}/{eval_loop_num}, success so far: {num_success}", flush=True)

    try:
        env.close()
    except Exception:
        pass
    del env
    gc.collect()
    return num_success / n_eval


def _eval_worker(args_tuple):
    """多 GPU 并行: 每个 worker 在指定 GPU 上评估一组任务"""
    (gpu_id, task_ids_subset, checkpoint_path, config_path, benchmark_name,
     n_eval, max_steps, camera_h, camera_w, seed, num_procs, save_videos, video_fps, output_dir) = args_tuple

    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    os.environ["MUJOCO_EGL_DEVICE_ID"] = str(gpu_id)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model, _ = load_model_and_config(checkpoint_path, config_path, device)

    from libero.libero.benchmark import get_benchmark
    from libero.libero import get_libero_path
    benchmark_map = {"libero_spatial": "LIBERO_SPATIAL", "libero_object": "LIBERO_OBJECT",
                     "libero_goal": "LIBERO_GOAL", "libero_10": "LIBERO_10"}
    benchmark_cls = get_benchmark(benchmark_map[benchmark_name])
    benchmark = benchmark_cls(task_order_index=0)
    bddl_folder = get_libero_path("bddl_files")
    init_folder = get_libero_path("init_states")

    eval_fn = evaluate_one_task_vector if num_procs > 1 else evaluate_one_task

    local_results = {}
    
    # 策略选择：
    # 1. 如果 num_procs > 1：尝试使用 DummyVectorEnv（虽然慢，但batch推理可能更快）
    # 2. 如果 num_procs = 1：使用单环境（最快，让多卡并行）
    # 
    # 注意：多进程worker中无法使用SubprocVectorEnv（daemonic进程限制）
    # 所以多卡时，要么用DummyVectorEnv（串行但batch推理），要么用单环境（最快）
    
    use_vector_env = num_procs > 1
    if use_vector_env:
        print(f"[Worker GPU {gpu_id}] Using DummyVectorEnv with {num_procs} envs (batch inference, sequential execution)", flush=True)
    else:
        print(f"[Worker GPU {gpu_id}] Using single env per task (multi-GPU parallelization)", flush=True)
    
    for tid in task_ids_subset:
        task = benchmark.get_task(tid)
        bddl_path = os.path.join(bddl_folder, task.problem_folder, task.bddl_file)
        init_path = os.path.join(init_folder, task.problem_folder, task.init_states_file)
        if not os.path.exists(bddl_path) or not os.path.exists(init_path):
            continue
        
        # 创建视频保存目录（如果需要）
        video_writer = None
        if save_videos and _HAS_VIDEO_WRITER:
            video_folder = os.path.join(output_dir, "videos", f"task_{tid}")
            video_writer = VideoWriter(video_folder, save_video=True, fps=video_fps, single_video=False)
        
        try:
            if use_vector_env:
                # 使用 DummyVectorEnv（虽然串行，但batch推理可能更快）
                sr = evaluate_one_task_vector(model, task, bddl_path, init_path, task.language, n_eval, max_steps,
                                             camera_h, camera_w, device, seed + tid, num_procs, use_dummy_vector_env=True, video_writer=video_writer)
            else:
                # 使用单环境（最快，让多卡并行）
                sr = evaluate_one_task(model, task, bddl_path, init_path, task.language, n_eval, max_steps,
                                       camera_h, camera_w, device, seed + tid, video_writer=video_writer)
            local_results[tid] = {"instruction": task.language, "success_rate": float(sr)}
        finally:
            if video_writer:
                video_writer.save()
    return local_results


def _log(msg: str, log_file=None):
    """同时打印并写入日志文件"""
    print(msg)
    if log_file is not None:
        log_file.write(msg + "\n")
        log_file.flush()


def main():
    args = parse_args()
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    # 解析并验证 checkpoint 路径（支持相对/绝对路径）
    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.is_absolute():
        ckpt_path = Path.cwd() / ckpt_path
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    args.checkpoint = str(ckpt_path.resolve())

    # 设置输出目录和日志目录
    output_dir = args.output_dir or str(Path(args.checkpoint).parent)
    log_dir = args.log_dir or output_dir
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    # 创建日志文件
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    checkpoint_stem = Path(args.checkpoint).stem
    log_filename = f"eval_{checkpoint_stem}_{timestamp}.log"
    log_path = os.path.join(log_dir, log_filename)
    log_file = open(log_path, "w", encoding="utf-8")

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    if args.gpus:
        gpu_first = args.gpus.split(",")[0].strip()
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
        os.environ["MUJOCO_EGL_DEVICE_ID"] = gpu_first
        args.device = f"cuda:{gpu_first}" if torch.cuda.is_available() else "cpu"
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # 初始化 wandb
    use_wandb = args.use_wandb and _HAS_WANDB
    if args.use_wandb and not _HAS_WANDB:
        _log("Warning: --use_wandb specified but wandb not installed. Run: pip install wandb", log_file)
    
    if use_wandb:
        wandb_run_name = args.wandb_run_name or f"eval_{checkpoint_stem}_{timestamp}"
        wandb_init_kwargs = {
            "project": args.wandb_project,
            "name": wandb_run_name,
            "dir": log_dir,
            "config": {
                "checkpoint": args.checkpoint,
                "benchmark": args.benchmark,
                "n_eval": args.n_eval,
                "max_steps": args.max_steps,
                "num_procs": args.num_procs,
                "seed": args.seed,
                "camera_h": args.camera_h,
                "camera_w": args.camera_w,
                "device": str(device),
                "gpus": args.gpus,
            }
        }
        if args.wandb_entity:
            wandb_init_kwargs["entity"] = args.wandb_entity
        wandb.init(**wandb_init_kwargs)
        _log(f"Weights & Biases logging enabled (entity: {args.wandb_entity or 'default'}, project: {args.wandb_project}).", log_file)

    _log("=" * 60, log_file)
    _log("VLA-LIBERO Evaluation", log_file)
    _log("=" * 60, log_file)
    _log(f"Checkpoint: {args.checkpoint}", log_file)
    _log(f"Benchmark: {args.benchmark}", log_file)
    _log(f"Device: {device}" + (f" (GPUs: {args.gpus})" if args.gpus else ""), log_file)
    _log(f"num_procs: {args.num_procs} (parallel envs for batch inference)", log_file)
    if args.save_videos:
        if _HAS_VIDEO_WRITER:
            _log(f"Video saving: ENABLED (FPS: {args.video_fps}, saved to {output_dir}/videos/)", log_file)
        else:
            _log("Video saving: DISABLED (VideoWriter not available)", log_file)
    else:
        _log("Video saving: DISABLED", log_file)
    _log(f"Log file: {log_path}", log_file)
    _log("=" * 60, log_file)

    from libero.libero.benchmark import get_benchmark
    from libero.libero import get_libero_path
    benchmark_map = {"libero_spatial": "LIBERO_SPATIAL", "libero_object": "LIBERO_OBJECT",
                     "libero_goal": "LIBERO_GOAL", "libero_10": "LIBERO_10"}
    benchmark_cls = get_benchmark(benchmark_map[args.benchmark])
    benchmark = benchmark_cls(task_order_index=0)
    n_tasks = benchmark.get_num_tasks()
    task_ids = args.task_ids if args.task_ids is not None else list(range(n_tasks))

    results = {}
    gpu_ids = [x.strip() for x in args.gpus.split(",") if x.strip()] if args.gpus else []

    if len(gpu_ids) > 1 and len(task_ids) > 0:
        # 多卡并行: 任务分配到各 GPU
        import multiprocessing as mp
        try:
            mp.set_start_method("spawn", force=True)
        except RuntimeError:
            pass
        n_workers = min(len(gpu_ids), len(task_ids))
        chunks = np.array_split(np.array(task_ids), n_workers)
        worker_args = []
        for i in range(len(chunks)):
            # 确保 chunks[i] 是数组，即使只有一个元素
            chunk_array = np.atleast_1d(chunks[i])
            if len(chunk_array) > 0:
                # 转换为 Python list，确保 numpy 类型不会导致序列化问题
                task_ids_list = [int(tid) for tid in chunk_array]
                worker_args.append((
                    int(gpu_ids[i % len(gpu_ids)]), task_ids_list,
                    args.checkpoint, args.config, args.benchmark, args.n_eval, args.max_steps,
                    args.camera_h, args.camera_w, args.seed, args.num_procs,
                    args.save_videos, args.video_fps, output_dir
                ))
        _log(f"\nMulti-GPU: {len(worker_args)} workers on GPUs {gpu_ids[:len(worker_args)]}", log_file)
        if args.num_procs > 1:
            _log(f"Note: Each worker uses DummyVectorEnv with {args.num_procs} envs (batch inference)", log_file)
            _log("      DummyVectorEnv is sequential but enables batch inference acceleration", log_file)
        else:
            _log("Note: Each worker uses single env (GPU-level parallelization - RECOMMENDED)", log_file)
            _log("      This maximizes multi-GPU efficiency", log_file)
        try:
            with mp.Pool(len(worker_args)) as pool:
                for local in pool.map(_eval_worker, worker_args):
                    results.update(local)
                    # 记录每个任务的结果到 wandb（多卡模式下）
                    if use_wandb:
                        for tid, result in local.items():
                            wandb.log({
                                f"eval/task_{tid}_success_rate": result["success_rate"],
                                "eval/task_id": tid,
                            })
        except Exception as e:
            _log(f"\n[ERROR] Multi-GPU evaluation failed: {e}", log_file)
            import traceback
            _log(traceback.format_exc(), log_file)
            if use_wandb:
                wandb.finish()
            log_file.close()
            raise
    else:
        # 单卡顺序评估（支持 num_procs 并行 env）
        _log("\nLoading model...", log_file)
        model, config = load_model_and_config(args.checkpoint, args.config, device)
        _log("Model loaded.", log_file)
        bddl_folder = get_libero_path("bddl_files")
        init_folder = get_libero_path("init_states")
        eval_fn = evaluate_one_task_vector if args.num_procs > 1 else evaluate_one_task
        for tid in task_ids:
            task = benchmark.get_task(tid)
            instruction = task.language
            bddl_path = os.path.join(bddl_folder, task.problem_folder, task.bddl_file)
            init_path = os.path.join(init_folder, task.problem_folder, task.init_states_file)
            if not os.path.exists(bddl_path):
                _log(f"[WARN] BDDL not found: {bddl_path}, skip task {tid}", log_file)
                continue
            if not os.path.exists(init_path):
                _log(f"[WARN] Init states not found: {init_path}, skip task {tid}", log_file)
                continue
            _log(f"\nEvaluating task {tid}: {instruction[:50]}...", log_file)
            
            # 创建视频保存目录（如果需要）
            video_writer = None
            if args.save_videos and _HAS_VIDEO_WRITER:
                video_folder = os.path.join(output_dir, "videos", f"task_{tid}")
                video_writer = VideoWriter(video_folder, save_video=True, fps=args.video_fps, single_video=False)
            
            try:
                if args.num_procs > 1:
                    sr = eval_fn(model, task, bddl_path, init_path, instruction, args.n_eval,
                        args.max_steps, args.camera_h, args.camera_w, device, args.seed + tid, args.num_procs, use_dummy_vector_env=False, video_writer=video_writer)
                else:
                    sr = eval_fn(model, task, bddl_path, init_path, instruction, args.n_eval,
                        args.max_steps, args.camera_h, args.camera_w, device, args.seed + tid, video_writer=video_writer)
                results[tid] = {"instruction": instruction, "success_rate": float(sr)}
            finally:
                if video_writer:
                    video_writer.save()
            _log(f"  Task {tid} success rate: {sr:.2%}", log_file)
            
            # 记录到 wandb
            if use_wandb:
                wandb.log({
                    f"eval/task_{tid}_success_rate": float(sr),
                    "eval/task_id": tid,
                })

    if not results:
        _log("\n[WARN] No results (all tasks skipped?)", log_file)
        if use_wandb:
            wandb.finish()
        log_file.close()
        return

    # 保存结果
    out_name = f"eval_{checkpoint_stem}_{timestamp}.json"
    out_path = os.path.join(output_dir, out_name)
    avg_sr = float(np.mean([r["success_rate"] for r in results.values()]))
    results_dict = {
        "checkpoint": args.checkpoint,
        "benchmark": args.benchmark,
        "task_ids": sorted(results.keys()),
        "n_eval": args.n_eval,
        "max_steps": args.max_steps,
        "num_procs": args.num_procs,
        "seed": args.seed,
        "results": results,
        "avg_success_rate": avg_sr,
    }
    with open(out_path, "w") as f:
        json.dump(results_dict, f, indent=2)
    
    _log(f"\nResults saved to {out_path}", log_file)
    _log(f"Average success rate: {avg_sr:.2%}", log_file)
    
    # 记录最终结果到 wandb
    if use_wandb:
        # 汇总所有任务的最终结果（避免重复记录单个任务）
        final_log = {
            "eval/avg_success_rate": avg_sr,
            "eval/num_tasks": len(results),
        }
        # 可选：记录每个任务的最终值（用于最终汇总表）
        for tid, result in results.items():
            final_log[f"eval/final_task_{tid}_success_rate"] = result["success_rate"]
        wandb.log(final_log)
        wandb.finish()
    
    _log("=" * 60, log_file)
    _log("Evaluation complete", log_file)
    _log("=" * 60, log_file)
    
    # 确保资源清理
    try:
        log_file.close()
    except Exception:
        pass


if __name__ == "__main__":
    main()
