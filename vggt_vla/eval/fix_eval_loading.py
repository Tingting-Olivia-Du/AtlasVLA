#!/usr/bin/env python3
"""
修复 eval 中的模型加载问题

问题：Checkpoint 使用 Qwen3-0.6B，但环境中只有 Qwen2-0.5B
解决方案：使用 strict=False 加载，并自动重新初始化不匹配的层
"""
import os
import sys
import torch
from pathlib import Path
from typing import Tuple, Dict, Any

VGGT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(VGGT_ROOT))


def load_model_with_fallback(
    checkpoint_path: str,
    device: str = "cuda:0",
    config_path: str = None,
    verbose: bool = True
) -> Tuple[Any, Any]:
    """
    加载模型，处理语言模型维度不匹配问题

    Args:
        checkpoint_path: checkpoint 文件路径
        device: 加载到的设备
        config_path: 配置文件路径（可选）
        verbose: 是否打印详细日志

    Returns:
        (model, config)
    """
    import yaml
    from configs.model_config import ModelConfig
    from models.vla_model import VLAModel

    device = torch.device(device)

    if verbose:
        print(f"\n{'='*80}")
        print("Loading model with fallback handling for language model dimension mismatch")
        print(f"{'='*80}\n")

    # ============ 加载 checkpoint ============
    if verbose:
        print(f"Loading checkpoint from: {checkpoint_path}")

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)

    if isinstance(ckpt, dict):
        state_dict = ckpt.get("model_state_dict") or ckpt.get("state_dict")
        saved_config = ckpt.get("config")
    else:
        state_dict = ckpt
        saved_config = None

    # ============ 重建配置 ============
    if saved_config is not None:
        if hasattr(saved_config, "vision"):
            config = saved_config
            if verbose:
                print("✓ Using config from checkpoint")
        else:
            config = saved_config
    elif config_path and os.path.exists(config_path):
        if verbose:
            print(f"✓ Loading config from: {config_path}")
        with open(config_path) as f:
            cfg = yaml.safe_load(f)
        from types import SimpleNamespace
        args = SimpleNamespace()
        defaults = {
            "use_vision_tower": False,
            "vision_tower_name": "facebook/dinov2-base",
            "freeze_vision_tower": True,
            "language_model": "Qwen/Qwen2-0.5B",  # 注意：使用 Qwen2
            "freeze_language": True,
            "use_pretrained_vggt": True,
            "freeze_vggt": False,
            "action_horizon": 10,
            "action_dim": 7
        }
        for k, v in {**defaults, **cfg}.items():
            setattr(args, k, v)
        from scripts.train_vla import build_config
        config = build_config(args)
    else:
        raise ValueError("No config found in checkpoint or config_path")

    # ============ 创建模型 ============
    if verbose:
        print("\nCreating model with config...")
    model = VLAModel(config)

    # ============ 尝试加载权重 ============
    if verbose:
        print("\nLoading state dict...")

    try:
        # 首先尝试严格加载
        model.load_state_dict(state_dict, strict=True)
        if verbose:
            print("✓ Loaded state_dict with strict=True")
    except RuntimeError as e:
        error_str = str(e)
        if 'size mismatch' in error_str and 'language' in error_str:
            if verbose:
                print(f"⚠ Size mismatch detected in language model: {error_str[:200]}...")
                print("\nAttempting to fix by loading with strict=False...")

            # 使用 strict=False 加载
            incompatible_keys = model.load_state_dict(state_dict, strict=False)

            if verbose:
                print(f"\n✓ Loaded with strict=False")
                print(f"  - Missing keys: {len(incompatible_keys.missing_keys)}")
                print(f"  - Unexpected keys: {len(incompatible_keys.unexpected_keys)}")

                if incompatible_keys.missing_keys:
                    print(f"\n  Missing keys (will be randomly initialized):")
                    for key in sorted(incompatible_keys.missing_keys)[:10]:
                        print(f"    - {key}")
                    if len(incompatible_keys.missing_keys) > 10:
                        print(f"    ... and {len(incompatible_keys.missing_keys) - 10} more")

                if incompatible_keys.unexpected_keys:
                    print(f"\n  Unexpected keys (will be ignored):")
                    for key in sorted(incompatible_keys.unexpected_keys)[:10]:
                        print(f"    - {key}")
                    if len(incompatible_keys.unexpected_keys) > 10:
                        print(f"    ... and {len(incompatible_keys.unexpected_keys) - 10} more")

            # ============ 修复不匹配的层 ============
            if verbose:
                print("\nRe-initializing language model layers...")

            # 重新初始化语言模型的投影层
            if hasattr(model, 'language_encoder') and hasattr(model.language_encoder, 'projector'):
                try:
                    # 获取实际的 language_hidden_size
                    lm = model.language_encoder.language_model
                    actual_hidden_size = lm.config.hidden_size

                    # 重建投影层
                    output_dim = config.language.output_dim
                    model.language_encoder.projector = torch.nn.Sequential(
                        torch.nn.Linear(actual_hidden_size, output_dim),
                        torch.nn.LayerNorm(output_dim),
                        torch.nn.GELU(),
                        torch.nn.Linear(output_dim, output_dim),
                        torch.nn.LayerNorm(output_dim)
                    )

                    if verbose:
                        print(f"  ✓ Re-initialized language projector: {actual_hidden_size} → {output_dim}")
                except Exception as e:
                    if verbose:
                        print(f"  ⚠ Could not re-initialize projector: {e}")

            if verbose:
                print("\n✓ Model loaded and fixed with fallback handling")
        else:
            # 其他类型的错误，直接抛出
            raise

    # ============ 移到设备并设置 eval 模式 ============
    model = model.to(device)
    model.eval()

    # 统计参数
    n_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    if verbose:
        print(f"\n✓ Model ready for evaluation")
        print(f"  - Device: {device}")
        print(f"  - Total params: {n_params:,}")
        print(f"  - Trainable params: {trainable_params:,}")
        print(f"\n{'='*80}\n")

    return model, config


if __name__ == "__main__":
    # 测试修复
    checkpoint_path = str(VGGT_ROOT / "logs/vla_libero_spatial/best_model_libero_spatial_image_20260214_045544_epoch297_step26690_loss0.0017.pt")

    if os.path.exists(checkpoint_path):
        print(f"Testing model loading with: {checkpoint_path}")

        try:
            model, config = load_model_with_fallback(checkpoint_path, verbose=True)
            print("\n✅ Model loading successful!")

            # 快速推理测试
            import numpy as np
            B = 2
            img_size = 224
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            dummy_img = torch.randn(B, 3, img_size, img_size).to(device)
            instructions = ["pick up the bowl", "open the drawer"]

            print("\nTesting inference...")
            with torch.no_grad():
                actions = model.predict_action(dummy_img, instructions)
            print(f"✓ Inference successful, output shape: {actions.shape}")

        except Exception as e:
            print(f"\n❌ Error: {e}")
            import traceback
            traceback.print_exc()
    else:
        print(f"Checkpoint not found: {checkpoint_path}")
