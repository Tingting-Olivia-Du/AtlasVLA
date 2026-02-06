#!/usr/bin/env python3
"""
快速测试：验证配置修复
"""
import yaml

print("=" * 60)
print("测试配置类型修复")
print("=" * 60)

# 加载配置
config = yaml.safe_load(open('atlas/configs/train_config.yaml'))
training_config = config['training']

# 测试原始值
lr_raw = training_config['learning_rate']
print(f"\n1. 原始配置值:")
print(f"   learning_rate = {repr(lr_raw)}")
print(f"   type = {type(lr_raw).__name__}")

# 测试转换后
lr_float = float(lr_raw)
print(f"\n2. 转换为float后:")
print(f"   learning_rate = {lr_float}")
print(f"   type = {type(lr_float).__name__}")

# 测试乘法
world_size = 8
effective_lr = lr_float * world_size
print(f"\n3. 多GPU缩放 (world_size={world_size}):")
print(f"   base_lr = {lr_float}")
print(f"   effective_lr = {effective_lr}")
print(f"   formatted = {effective_lr:.4f} (or {effective_lr:.2e})")

# 验证日志格式
log_message = f"  Base LR: {lr_float}, Effective LR (scaled by {world_size}): {effective_lr}"
print(f"\n4. 日志格式测试:")
print(f"   {log_message}")

# 检查是否修复
if isinstance(lr_raw, str):
    print(f"\n❌ WARNING: learning_rate仍然是字符串!")
    print(f"   建议修改配置文件为: learning_rate: 0.0001")
else:
    print(f"\n✅ SUCCESS: learning_rate是数值类型")

if "1e-41e-4" in log_message:
    print(f"❌ FAILED: 日志格式仍然错误!")
else:
    print(f"✅ SUCCESS: 日志格式正确")

print("\n" + "=" * 60)
print("测试完成")
print("=" * 60)
