#!/usr/bin/env python3
"""
调试 LIBERO eval 的核心问题
1. 检查动作范围和格式
2. 验证模型权重加载
3. 检查特征流和质量
4. 运行单步调试
"""
import os
import sys
import torch
import numpy as np
from pathlib import Path

# 添加路径
VGGT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(VGGT_ROOT))
LIBERO_ROOT = VGGT_ROOT.parent / "dataset" / "LIBERO"
sys.path.insert(0, str(LIBERO_ROOT))

print("=" * 80)
print("LIBERO Eval Debug Script")
print("=" * 80)

# ============ 第1部分：检查动作范围 ============
print("\n[1] Checking Action Space & Format")
print("-" * 80)

try:
    from libero.libero.envs import OffScreenRenderEnv
    from libero.libero import get_libero_path

    # 获取一个任务的 BDDL 文件
    bddl_folder = get_libero_path("bddl_files")
    bddl_path = os.path.join(bddl_folder, "KITCHEN_SCENE1", "open_the_bottom_drawer_of_the_cabinet.bddl")

    print(f"Creating env with BDDL: {bddl_path}")
    env = OffScreenRenderEnv(
        bddl_file_name=bddl_path,
        camera_heights=224,
        camera_widths=224,
    )

    # 检查动作空间
    print(f"\n✓ Environment created successfully")
    print(f"  - Action dim: {env.env.action_dim if hasattr(env.env, 'action_dim') else 'Unknown'}")
    print(f"  - Observation keys: {list(env.env.reset()[0].keys()) if isinstance(env.env.reset(), tuple) else 'Unknown'}")

    # 测试动作范围
    obs = env.reset()
    obs_dict = obs[0] if isinstance(obs, tuple) else obs
    print(f"\n  - Reset observation type: {type(obs_dict)}")

    # 尝试发送不同范围的动作
    test_actions = [
        np.zeros(7),           # 全0
        np.ones(7),            # 全1
        np.ones(7) * 0.1,      # 0.1
        np.ones(7) * -0.5,     # -0.5
        np.random.randn(7),    # 随机高斯
    ]

    print(f"\n  Testing action ranges:")
    for i, test_act in enumerate(test_actions):
        try:
            obs, reward, done, info = env.step(test_act)
            obs_dict = obs[0] if isinstance(obs, tuple) else obs
            img_key = "agentview_image" if "agentview_image" in obs_dict else "agentview_rgb"
            print(f"    ✓ Action {i}: {test_act[:2]}... | Image shape: {obs_dict[img_key].shape}")
        except Exception as e:
            print(f"    ✗ Action {i} failed: {e}")

    env.close()
    print("\n  ✓ Environment closed successfully")

except Exception as e:
    print(f"✗ Failed to test action space: {e}")
    import traceback
    traceback.print_exc()

# ============ 第2部分：验证模型加载 ============
print("\n" + "=" * 80)
print("[2] Validating Model Loading & Inference")
print("-" * 80)

try:
    from configs.model_config import ModelConfig
    from models.vla_model import VLAModel

    checkpoint_path = str(VGGT_ROOT / "logs/vla_libero_spatial/best_model_libero_spatial_image_20260214_045544_epoch297_step26690_loss0.0017.pt")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if not os.path.exists(checkpoint_path):
        print(f"✗ Checkpoint not found: {checkpoint_path}")
        # 尝试找到一个存在的 checkpoint
        logs_dir = VGGT_ROOT / "logs/vla_libero_spatial"
        pt_files = list(logs_dir.glob("best_model_*.pt"))
        if pt_files:
            checkpoint_path = str(pt_files[0])
            print(f"  Found checkpoint: {checkpoint_path}")
        else:
            raise FileNotFoundError("No checkpoint files found")

    print(f"Loading checkpoint from: {checkpoint_path}")

    # 加载 checkpoint
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)

    if isinstance(ckpt, dict):
        state_dict = ckpt.get("model_state_dict") or ckpt.get("state_dict")
        saved_config = ckpt.get("config")
        print(f"\n  ✓ Checkpoint is dict with keys: {list(ckpt.keys())}")
        print(f"    - state_dict size: {len(state_dict) if state_dict else 'None'}")
        print(f"    - config present: {saved_config is not None}")
    else:
        state_dict = ckpt
        saved_config = None
        print(f"\n  ✓ Checkpoint is tensor with shape: {ckpt.shape if hasattr(ckpt, 'shape') else 'N/A'}")

    # 重建配置
    if saved_config is not None:
        if hasattr(saved_config, 'vision'):
            config = saved_config
            print(f"  ✓ Using saved config from checkpoint")
        else:
            print(f"  ✗ Config format not recognized")
    else:
        print(f"  ✗ No config in checkpoint, creating default config")
        # 创建默认配置
        from types import SimpleNamespace
        defaults = {
            "use_vision_tower": False,
            "vision_tower_name": "facebook/dinov2-base",
            "freeze_vision_tower": True,
            "language_model": "Qwen/Qwen3-0.6B-Base",
            "freeze_language": True,
            "use_pretrained_vggt": True,
            "freeze_vggt": False,
            "action_horizon": 10,
            "action_dim": 7
        }
        args = SimpleNamespace(**defaults)
        from scripts.train_vla import build_config
        config = build_config(args)

    # 创建和加载模型
    print(f"\n  Creating VLA model...")
    model = VLAModel(config)
    print(f"\n  Loading state_dict...")
    model.load_state_dict(state_dict, strict=True)
    print(f"  ✓ State dict loaded successfully")

    model = model.to(device)
    model.eval()

    # 统计参数
    n_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n  ✓ Model loaded to {device}")
    print(f"    - Total params: {n_params:,}")
    print(f"    - Trainable params: {trainable_params:,}")

    # ============ 第3部分：检查推理 ============
    print("\n" + "=" * 80)
    print("[3] Testing Model Inference & Feature Flow")
    print("-" * 80)

    # 创建虚拟输入
    B = 2
    img_size = 224
    dummy_img = torch.randn(B, 3, img_size, img_size).to(device)
    instructions = ["pick up the black bowl and place it on the plate", "open the drawer"]

    print(f"\n  Input shapes:")
    print(f"    - Images: {dummy_img.shape} (dtype: {dummy_img.dtype})")
    print(f"    - Instructions: {len(instructions)} instructions")

    print(f"\n  Running forward pass...")
    with torch.no_grad():
        outputs = model.forward(dummy_img, instructions, return_features=True)

    print(f"\n  ✓ Forward pass successful")
    print(f"    - Output keys: {list(outputs.keys())}")
    print(f"    - Actions shape: {outputs['actions'].shape}")
    print(f"    - Actions dtype: {outputs['actions'].dtype}")
    print(f"    - Actions range: [{outputs['actions'].min():.4f}, {outputs['actions'].max():.4f}]")
    print(f"    - Actions mean: {outputs['actions'].mean():.4f}, std: {outputs['actions'].std():.4f}")

    if 'vision_features' in outputs:
        vf = outputs['vision_features']
        print(f"\n  Vision features:")
        print(f"    - Shape: {vf.shape}")
        print(f"    - Range: [{vf.min():.4f}, {vf.max():.4f}]")
        print(f"    - Contain NaN: {torch.isnan(vf).any()}")
        print(f"    - Contain Inf: {torch.isinf(vf).any()}")

    if 'language_features' in outputs:
        lf = outputs['language_features']
        print(f"\n  Language features:")
        print(f"    - Shape: {lf.shape}")
        print(f"    - Range: [{lf.min():.4f}, {lf.max():.4f}]")
        print(f"    - Contain NaN: {torch.isnan(lf).any()}")
        print(f"    - Contain Inf: {torch.isinf(lf).any()}")

    if 'global_features' in outputs:
        gf = outputs['global_features']
        print(f"\n  Global features (for action):")
        print(f"    - Shape: {gf.shape}")
        print(f"    - Range: [{gf.min():.4f}, {gf.max():.4f}]")
        print(f"    - Contain NaN: {torch.isnan(gf).any()}")
        print(f"    - Contain Inf: {torch.isinf(gf).any()}")

    # 测试 predict_action
    print(f"\n  Testing predict_action method...")
    actions = model.predict_action(dummy_img, instructions)
    print(f"  ✓ predict_action returned shape: {actions.shape}")
    print(f"    - Range: [{actions.min():.4f}, {actions.max():.4f}]")

    # 测试多次推理的一致性
    print(f"\n  Testing inference consistency...")
    with torch.no_grad():
        act1 = model.predict_action(dummy_img, instructions).detach().cpu().numpy()
        act2 = model.predict_action(dummy_img, instructions).detach().cpu().numpy()

    diff = np.abs(act1 - act2).max()
    print(f"    - Max difference between two runs: {diff:.6f}")
    if diff > 1e-5:
        print(f"    ⚠ Actions are not deterministic!")
    else:
        print(f"    ✓ Actions are deterministic")

except Exception as e:
    print(f"✗ Model loading/inference failed: {e}")
    import traceback
    traceback.print_exc()

# ============ 第4部分：完整端到端测试 ============
print("\n" + "=" * 80)
print("[4] End-to-End Episode Test")
print("-" * 80)

try:
    from libero.libero.envs import OffScreenRenderEnv
    from libero.libero import get_libero_path
    import torchvision.transforms as T

    def preprocess_image(img, device, img_size=224):
        """预处理图像"""
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

    # 创建环境
    bddl_folder = get_libero_path("bddl_files")
    bddl_path = os.path.join(bddl_folder, "KITCHEN_SCENE1", "open_the_bottom_drawer_of_the_cabinet.bddl")
    init_folder = get_libero_path("init_states")
    init_path = os.path.join(init_folder, "KITCHEN_SCENE1", "open_the_bottom_drawer_of_the_cabinet.pt")

    print(f"Creating environment...")
    env = OffScreenRenderEnv(
        bddl_file_name=bddl_path,
        camera_heights=224,
        camera_widths=224,
    )

    # 加载初始状态
    init_states = torch.load(init_path, weights_only=False)
    print(f"  ✓ Loaded {init_states.shape[0]} initial states")

    env.reset()
    env.set_init_state(init_states[0])

    obs = env.reset()
    obs_dict = obs[0] if isinstance(obs, tuple) else obs
    img_key = "agentview_image" if "agentview_image" in obs_dict else "agentview_rgb"
    img = obs_dict[img_key]

    print(f"\n  Running 5 steps with model predictions...")
    instruction = "open the bottom drawer of the cabinet"

    for step in range(5):
        # 预处理图像
        img_tensor = preprocess_image(img, device, 224)

        # 模型推理
        with torch.no_grad():
            actions = model.predict_action(img_tensor, [instruction])

        if actions.dim() == 3:
            actions = actions[:, 0, :]

        act = actions.cpu().numpy()[0]

        print(f"\n  Step {step+1}:")
        print(f"    - Image shape: {img.shape}")
        print(f"    - Action: {act}")
        print(f"    - Action range: [{act.min():.4f}, {act.max():.4f}]")

        # 执行动作
        obs, reward, done, info = env.step(act)
        obs_dict = obs[0] if isinstance(obs, tuple) else obs
        img = obs_dict[img_key]

        done_val = bool(done[0]) if isinstance(done, (list, tuple)) else bool(done)
        print(f"    - Done: {done_val}, Reward: {reward if isinstance(reward, (int, float)) else 'N/A'}")

        if done_val:
            print(f"    ✓ Task completed at step {step+1}!")
            break

    env.close()

except Exception as e:
    print(f"✗ End-to-end test failed: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 80)
print("Debug script completed")
print("=" * 80)
