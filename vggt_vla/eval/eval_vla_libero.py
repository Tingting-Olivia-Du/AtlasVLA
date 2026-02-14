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

# 添加 vggt_vla 根目录
VGGT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(VGGT_ROOT))

# 添加 LIBERO 目录
LIBERO_ROOT = VGGT_ROOT.parent / "dataset" / "LIBERO"
sys.path.insert(0, str(LIBERO_ROOT))


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
    model.load_state_dict(state_dict, strict=True)
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
                      camera_h, camera_w, device, seed):
    """评估单个任务的 success rate"""
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
        for _ in range(4):
            obs, _, _, _ = env.step(dummy)

        done = False
        for step in range(max_steps - 5):
            o = obs[0] if isinstance(obs, (list, tuple)) and len(obs) > 0 else obs
            img = o.get("agentview_image", o.get("agentview_rgb"))
            img_tensor = preprocess_image(img, device, camera_h)

            with torch.no_grad():
                actions = model.predict_action(img_tensor, [instruction])
            if actions.dim() == 3:
                actions = actions[:, 0, :]
            act = actions.cpu().numpy()[0]

            obs, reward, done, info = env.step(act)
            if isinstance(done, (list, np.ndarray)):
                done = bool(done[0])
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
                             camera_h, camera_w, device, seed, num_procs):
    """使用 VectorEnv 并行评估单个任务（batch 推理加速）"""
    import gc
    import sys
    from libero.libero.envs import OffScreenRenderEnv, SubprocVectorEnv, DummyVectorEnv

    env_num = min(num_procs, n_eval)
    eval_loop_num = (n_eval + env_num - 1) // env_num

    print(f"    Creating {env_num} parallel envs (VectorEnv)...", flush=True)
    sys.stdout.flush()
    env_args = {
        "bddl_file_name": bddl_path,
        "camera_heights": camera_h,
        "camera_widths": camera_w,
    }
    try:
        if env_num == 1:
            env = DummyVectorEnv([lambda: OffScreenRenderEnv(**env_args) for _ in range(env_num)])
        else:
            env = SubprocVectorEnv([lambda: OffScreenRenderEnv(**env_args) for _ in range(env_num)])
    except Exception as e:
        print(f"    [WARN] VectorEnv failed, falling back to single env: {e}", flush=True)
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
            if isinstance(done, np.ndarray):
                for k in range(env_num):
                    dones[k] = dones[k] or bool(done[k])
            else:
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
     n_eval, max_steps, camera_h, camera_w, seed, num_procs) = args_tuple

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
    for tid in task_ids_subset:
        task = benchmark.get_task(tid)
        bddl_path = os.path.join(bddl_folder, task.problem_folder, task.bddl_file)
        init_path = os.path.join(init_folder, task.problem_folder, task.init_states_file)
        if not os.path.exists(bddl_path) or not os.path.exists(init_path):
            continue
        if num_procs > 1:
            sr = eval_fn(model, task, bddl_path, init_path, task.language, n_eval, max_steps,
                         camera_h, camera_w, device, seed + tid, num_procs)
        else:
            sr = eval_fn(model, task, bddl_path, init_path, task.language, n_eval, max_steps,
                         camera_h, camera_w, device, seed + tid)
        local_results[tid] = {"instruction": task.language, "success_rate": float(sr)}
    return local_results


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

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    if args.gpus:
        gpu_first = args.gpus.split(",")[0].strip()
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
        os.environ["MUJOCO_EGL_DEVICE_ID"] = gpu_first
        args.device = f"cuda:{gpu_first}" if torch.cuda.is_available() else "cpu"
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    print("=" * 60)
    print("VLA-LIBERO Evaluation")
    print("=" * 60)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Benchmark: {args.benchmark}")
    print(f"Device: {device}" + (f" (GPUs: {args.gpus})" if args.gpus else ""))
    print(f"num_procs: {args.num_procs} (parallel envs for batch inference)")
    print("=" * 60)

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
            if len(chunks[i]) > 0:
                worker_args.append((
                    int(gpu_ids[i % len(gpu_ids)]), list(chunks[i]),
                    args.checkpoint, args.config, args.benchmark, args.n_eval, args.max_steps,
                    args.camera_h, args.camera_w, args.seed, args.num_procs
                ))
        print(f"\nMulti-GPU: {len(worker_args)} workers on GPUs {gpu_ids[:len(worker_args)]}")
        with mp.Pool(len(worker_args)) as pool:
            for local in pool.map(_eval_worker, worker_args):
                results.update(local)
    else:
        # 单卡顺序评估（支持 num_procs 并行 env）
        print("\nLoading model...")
        model, config = load_model_and_config(args.checkpoint, args.config, device)
        print("Model loaded.")
        bddl_folder = get_libero_path("bddl_files")
        init_folder = get_libero_path("init_states")
        eval_fn = evaluate_one_task_vector if args.num_procs > 1 else evaluate_one_task
        for tid in task_ids:
            task = benchmark.get_task(tid)
            instruction = task.language
            bddl_path = os.path.join(bddl_folder, task.problem_folder, task.bddl_file)
            init_path = os.path.join(init_folder, task.problem_folder, task.init_states_file)
            if not os.path.exists(bddl_path):
                print(f"[WARN] BDDL not found: {bddl_path}, skip task {tid}")
                continue
            if not os.path.exists(init_path):
                print(f"[WARN] Init states not found: {init_path}, skip task {tid}")
                continue
            print(f"\nEvaluating task {tid}: {instruction[:50]}...")
            if args.num_procs > 1:
                sr = eval_fn(model, task, bddl_path, init_path, instruction, args.n_eval,
                    args.max_steps, args.camera_h, args.camera_w, device, args.seed + tid, args.num_procs)
            else:
                sr = eval_fn(model, task, bddl_path, init_path, instruction, args.n_eval,
                    args.max_steps, args.camera_h, args.camera_w, device, args.seed + tid)
            results[tid] = {"instruction": instruction, "success_rate": float(sr)}
            print(f"  Task {tid} success rate: {sr:.2%}")

    if not results:
        print("\n[WARN] No results (all tasks skipped?)")
        return

    # 保存结果
    output_dir = args.output_dir or str(Path(args.checkpoint).parent)
    os.makedirs(output_dir, exist_ok=True)
    out_name = f"eval_{Path(args.checkpoint).stem}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    out_path = os.path.join(output_dir, out_name)
    avg_sr = float(np.mean([r["success_rate"] for r in results.values()]))
    with open(out_path, "w") as f:
        json.dump({
            "checkpoint": args.checkpoint,
            "benchmark": args.benchmark,
            "task_ids": sorted(results.keys()),
            "n_eval": args.n_eval,
            "results": results,
            "avg_success_rate": avg_sr,
        }, f, indent=2)
    print(f"\nResults saved to {out_path}")
    print(f"Average success rate: {avg_sr:.2%}")


if __name__ == "__main__":
    main()
