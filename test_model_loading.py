#!/usr/bin/env python3
"""
最终测试：验证模型加载流程
"""
import yaml
import os
import sys
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)

print("=" * 70)
print("Atlas VLA - 模型加载测试")
print("=" * 70)

# 1. 读取配置
logging.info("Step 1: 读取配置文件")
config = yaml.safe_load(open('atlas/configs/train_config.yaml'))
hf_token = config.get('huggingface', {}).get('token')
model_name = config['model']['lang_encoder_name']

logging.info(f"  Model: {model_name}")
logging.info(f"  Token: {hf_token[:20]}... (length: {len(hf_token)})")

# 2. 设置环境变量（这是关键！）
logging.info("Step 2: 设置环境变量")
os.environ['HF_TOKEN'] = hf_token
os.environ['HUGGINGFACE_TOKEN'] = hf_token
logging.info("  环境变量已设置")

# 3. 测试transformers直接加载（验证token有效）
logging.info("Step 3: 测试transformers直接加载")
try:
    from transformers import AutoConfig, AutoModel, AutoTokenizer
    
    # 测试config加载
    logging.info(f"  加载config: {model_name}")
    config_obj = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
    logging.info(f"  ✓ Config加载成功! Model type: {config_obj.model_type}")
    
    # 测试model加载（这会花时间）
    logging.info(f"  加载model（这需要几分钟，正在下载...）")
    model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
    logging.info(f"  ✓ Model加载成功!")
    
    # 测试tokenizer加载
    logging.info(f"  加载tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    logging.info(f"  ✓ Tokenizer加载成功!")
    
    logging.info("=" * 70)
    logging.info("所有测试通过！Token配置正确!")
    logging.info("=" * 70)
    
except Exception as e:
    logging.error(f"✗ 加载失败: {e}")
    logging.error("请检查:")
    logging.error(f"  1. Token是否有效: {hf_token[:20]}...")
    logging.error(f"  2. 是否在HuggingFace网站上同意了license")
    logging.error(f"  3. Token是否有访问该模型的权限")
    sys.exit(1)

print("\n" + "=" * 70)
print("测试完成 - 可以开始训练了！")
print("=" * 70)
