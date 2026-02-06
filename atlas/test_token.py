#!/usr/bin/env python3
"""
测试Token传递是否正确
"""
import sys
import os

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

import yaml
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

print("=" * 60)
print("Token传递测试")
print("=" * 60)

# 1. 读取配置
print("\n1. 读取配置文件...")
config = yaml.safe_load(open('atlas/configs/train_config.yaml'))
hf_token = config.get("huggingface", {}).get("token")
print(f"   Token: {hf_token[:20]}... (length: {len(hf_token)})")
logging.info(f"Token from config: {hf_token[:15]}... (length: {len(hf_token)})")

# 2. 设置环境变量
print("\n2. 设置环境变量...")
os.environ['HF_TOKEN'] = hf_token
os.environ['HUGGINGFACE_TOKEN'] = hf_token
print(f"   HF_TOKEN set: {os.environ.get('HF_TOKEN')[:20]}...")
logging.info(f"Environment variable HF_TOKEN: {os.environ.get('HF_TOKEN')[:15]}...")

# 3. 测试AutoConfig加载
print("\n3. 测试AutoConfig.from_pretrained...")
from transformers import AutoConfig
model_name = config['model']['lang_encoder_name']
print(f"   Model: {model_name}")

try:
    # 测试不同的token传递方式
    print("\n   方式1: 仅环境变量...")
    config1 = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
    print(f"   ✓ 成功! Model type: {config1.model_type}")
except Exception as e1:
    print(f"   ✗ 失败: {str(e1)[:100]}...")
    
    try:
        print("\n   方式2: 显式token参数...")
        config2 = AutoConfig.from_pretrained(model_name, token=hf_token, trust_remote_code=True)
        print(f"   ✓ 成功! Model type: {config2.model_type}")
    except Exception as e2:
        print(f"   ✗ 失败: {str(e2)[:100]}...")

# 4. 测试VGGTVLA初始化
print("\n4. 测试VGGTVLA初始化...")
try:
    from atlas.src.models import VGGTVLA
    print("   Creating VGGTVLA instance...")
    logging.info("Creating VGGTVLA with token...")
    
    model = VGGTVLA(
        lang_encoder_name=model_name,
        hf_token=hf_token,
        freeze_vggt=True,
        freeze_lang_encoder=True,
    )
    print("   ✓ 模型创建成功!")
    logging.info("Model created successfully!")
except Exception as e:
    print(f"   ✗ 模型创建失败: {e}")
    logging.error(f"Model creation failed: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 60)
print("测试完成")
print("=" * 60)
