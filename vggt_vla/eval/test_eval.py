#!/usr/bin/env python3
"""
测试 eval_vla.py 的基本功能

用法:
    python test_eval.py
"""

import os
import sys
from pathlib import Path

# 设置路径
_repo_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _repo_root)
sys.path.insert(0, os.path.join(_repo_root, '..'))

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch


def test_imports():
    """测试导入"""
    print("\n" + "="*60)
    print("[测试] 导入测试")
    print("="*60)

    try:
        from eval_vla import VLAEvaluator, parse_args
        print("  ✓ 成功导入 VLAEvaluator 和 parse_args")

        from models.vla_model import VLAModel
        print("  ✓ 成功导入 VLAModel")

        from configs.model_config import ModelConfig
        print("  ✓ 成功导入 ModelConfig")

        # 尝试导入 LIBERO 模块
        from dataset.LIBERO.libero.libero import get_libero_path
        print("  ✓ 成功导入 LIBERO 工具")

        from dataset.LIBERO.libero.libero.benchmark import get_benchmark
        print("  ✓ 成功导入 LIBERO 基准")

        print("\n✓ 所有导入测试通过")
        return True

    except Exception as e:
        print(f"\n✗ 导入失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_config():
    """测试配置加载"""
    print("\n" + "="*60)
    print("[测试] 配置加载测试")
    print("="*60)

    try:
        from configs.model_config import ModelConfig

        # 使用默认配置
        config = ModelConfig()
        print("  ✓ 成功创建默认配置")

        # 检查关键属性
        assert hasattr(config, 'vision'), "缺少 vision 配置"
        assert hasattr(config, 'language'), "缺少 language 配置"
        assert hasattr(config, 'vggt'), "缺少 vggt 配置"
        assert hasattr(config, 'action_head'), "缺少 action_head 配置"
        print("  ✓ 所有配置组件都存在")

        # 打印配置概要
        print(f"\n  配置概要:")
        print(f"    - Vision: {config.vision.vision_tower_name if hasattr(config.vision, 'vision_tower_name') else 'N/A'}")
        print(f"    - Language: {config.language.language_model if hasattr(config.language, 'language_model') else 'N/A'}")
        print(f"    - VGGT: {'enabled' if config.vggt.use_pretrained_vggt else 'disabled'}")

        print("\n✓ 配置加载测试通过")
        return True

    except Exception as e:
        print(f"\n✗ 配置加载失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_model():
    """测试模型初始化"""
    print("\n" + "="*60)
    print("[测试] 模型初始化测试")
    print("="*60)

    try:
        from models.vla_model import VLAModel
        from configs.model_config import ModelConfig

        config = ModelConfig()
        print("  ✓ 配置已加载")

        # 尝试初始化模型（不加载 vision tower）
        print("\n  正在初始化模型...")
        model = VLAModel(config)
        print("  ✓ 模型初始化成功")

        # 检查模块
        assert hasattr(model, 'vision_encoder'), "缺少 vision_encoder"
        assert hasattr(model, 'language_encoder'), "缺少 language_encoder"
        assert hasattr(model, 'vggt_backbone'), "缺少 vggt_backbone"
        assert hasattr(model, 'action_head'), "缺少 action_head"
        print("  ✓ 所有模型组件都存在")

        # 计算参数数量
        total_params = sum(p.numel() for p in model.parameters())
        print(f"\n  模型统计:")
        print(f"    - 总参数: {total_params:,}")

        print("\n✓ 模型初始化测试通过")
        return True

    except Exception as e:
        print(f"\n✗ 模型初始化失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_benchmark():
    """测试 LIBERO 基准加载"""
    print("\n" + "="*60)
    print("[测试] LIBERO 基准加载测试")
    print("="*60)

    try:
        from dataset.LIBERO.libero.libero.benchmark import get_benchmark

        # 尝试加载基准
        benchmarks = [
            'LIBERO_SPATIAL',
            'LIBERO_OBJECT',
            'LIBERO_GOAL',
            'LIBERO_10'
        ]

        for benchmark_name in benchmarks:
            try:
                benchmark = get_benchmark(benchmark_name)(0)
                task_names = benchmark.get_task_names()
                print(f"  ✓ {benchmark_name}: {len(task_names)} 个任务")
            except Exception as e:
                print(f"  ⚠ {benchmark_name}: 加载失败 - {e}")

        print("\n✓ 基准加载测试完成")
        return True

    except Exception as e:
        print(f"\n✗ 基准加载失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_checkpoint():
    """测试检查点文件"""
    print("\n" + "="*60)
    print("[测试] 检查点文件测试")
    print("="*60)

    checkpoint_path = "logs/vla_libero_spatial/best_model_libero_spatial_image_20260214_045544_epoch297_step26690_loss0.0017.pt"

    if not os.path.exists(checkpoint_path):
        print(f"  ⚠ 检查点文件不存在: {checkpoint_path}")
        print(f"    预期路径: {os.path.abspath(checkpoint_path)}")
        print("\n  列出可用的检查点文件:")
        logs_dir = "logs"
        if os.path.exists(logs_dir):
            for root, dirs, files in os.walk(logs_dir):
                for file in files:
                    if file.endswith('.pt') or file.endswith('.pth'):
                        full_path = os.path.join(root, file)
                        print(f"    - {full_path}")
        else:
            print(f"    日志目录不存在: {logs_dir}")
        return False
    else:
        # 尝试加载检查点
        print(f"  ✓ 检查点文件存在: {checkpoint_path}")
        try:
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            print(f"  ✓ 检查点加载成功")

            # 检查内容
            if isinstance(checkpoint, dict):
                keys = list(checkpoint.keys())
                print(f"    - 检查点键: {keys}")
            else:
                print(f"    - 检查点类型: {type(checkpoint)}")

            return True
        except Exception as e:
            print(f"  ✗ 检查点加载失败: {e}")
            return False


def main():
    """运行所有测试"""
    print("\n")
    print("╔" + "="*58 + "╗")
    print("║" + " "*58 + "║")
    print("║" + "  VLA-VGGT 评估脚本测试".center(58) + "║")
    print("║" + " "*58 + "║")
    print("╚" + "="*58 + "╝")

    tests = [
        ("导入", test_imports),
        ("配置", test_config),
        ("模型", test_model),
        ("基准", test_benchmark),
        ("检查点", test_checkpoint),
    ]

    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"\n✗ {test_name} 测试异常: {e}")
            import traceback
            traceback.print_exc()
            results.append((test_name, False))

    # 总结
    print("\n" + "="*60)
    print("[总结]")
    print("="*60)

    passed = sum(1 for _, result in results if result)
    total = len(results)

    for test_name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"  {status}: {test_name}")

    print(f"\n  总体: {passed}/{total} 个测试通过")

    if passed == total:
        print("\n✓ 所有测试通过！可以开始评估。")
        return 0
    else:
        print(f"\n✗ 有 {total - passed} 个测试失败。请检查错误消息。")
        return 1


if __name__ == '__main__':
    sys.exit(main())
