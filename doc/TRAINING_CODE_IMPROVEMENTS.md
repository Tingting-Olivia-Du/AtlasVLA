# 训练代码改进文档

本文档说明了对训练代码的改进，包括wandb配置修复和架构改进的集成。

## 🔧 问题修复

### 1. Wandb配置问题修复

#### 问题诊断

**问题1**: Wandb没有生效
- 配置文件中 `wandb.enabled: false`，所以wandb被禁用
- Wandb初始化时缺少关键参数（save_code, resume）
- 没有正确传递训练配置到wandb

**问题2**: Wandb初始化缺少错误处理
- 如果wandb初始化失败，会导致整个训练失败
- 没有日志提示wandb状态

#### 修复内容

1. **添加wandb参数传递** (`train.py:401-428`)
   ```python
   # 传递wandb配置参数
   save_code=wandb_config.get("save_code", True),
   resume=wandb_config.get("resume", "allow"),
   ```

2. **改进wandb初始化** (`trainer.py:161-195`)
   - 添加 `save_code` 和 `resume` 参数
   - 添加错误处理，wandb失败不影响训练
   - 记录更多训练配置到wandb.config
   - 添加wandb URL日志

3. **添加状态日志** (`train.py:401-404`)
   ```python
   if rank == 0:
       if use_wandb:
           logging.info("Wandb enabled - experiment tracking will be saved")
       else:
           logging.info("Wandb disabled - set wandb.enabled: true in config to enable")
   ```

### 2. 架构改进集成

#### 问题诊断

**问题**: 新添加的架构改进没有被使用
- 模型初始化时没有传递新参数
- Dataset初始化时没有传递新参数
- Loss函数调用时没有传递intermediates

#### 修复内容

1. **模型初始化支持新参数** (`train.py:227-240`)
   ```python
   model = VGGTVLA(
       # ... 原有参数 ...
       use_quaternion=model_config.get("use_quaternion", False),
       use_attention_pooling=model_config.get("use_attention_pooling", True),
   )
   ```

2. **Dataset初始化支持新参数** (`train.py:347-361`)
   ```python
   train_dataset = LIBERODataset(
       # ... 原有参数 ...
       num_temporal_frames=data_config.get("num_temporal_frames", 1),
       temporal_stride=data_config.get("temporal_stride", 1),
       normalize_actions=data_config.get("normalize_actions", False),
       action_stats_path=data_config.get("action_stats_path"),
   )
   ```

3. **Loss函数支持辅助损失** (`trainer.py:214-235`)
   ```python
   # 如果需要辅助损失，返回中间特征
   return_intermediates = self.criterion.use_auxiliary_loss
   outputs = self.model(images, language_tasks, return_intermediates=return_intermediates)
   
   # 传递中间特征用于辅助损失
   if return_intermediates and "geometry_features" in outputs:
       loss_kwargs["intermediates"] = {
           "geometry_features": outputs.get("geometry_features"),
           "fused_features": outputs.get("fused_features"),
       }
   ```

4. **VGGTVLA支持新参数** (`vggt_vla.py:196-201, 204-208`)
   ```python
   # 改进5: 支持attention pooling
   self.fusion = MultimodalFusion(
       # ...
       use_attention_pooling=use_attention_pooling
   )
   
   # 改进3: 支持四元数
   self.action_head = ActionHead(
       # ...
       use_quaternion=use_quaternion
   )
   ```

## 📋 使用指南

### 启用Wandb

1. **修改配置文件** (`atlas/configs/train_config.yaml`):
   ```yaml
   wandb:
     enabled: true  # 改为true
     project: "atlas-vla"
     entity: "your-wandb-username"
     save_code: true  # 保存代码到wandb
     resume: "allow"
   ```

2. **登录Wandb**:
   ```bash
   wandb login
   ```

3. **开始训练**:
   ```bash
   python atlas/train.py --config atlas/configs/train_config.yaml
   ```

4. **查看结果**:
   - 训练开始后，会显示wandb URL
   - 或访问 https://wandb.ai/your-username/atlas-vla

### 启用架构改进

1. **启用动作归一化**:
   ```yaml
   data:
     normalize_actions: true
     action_stats_path: null  # null=自动计算
   ```

2. **启用四元数表示**:
   ```yaml
   model:
     use_quaternion: true
     action_dim: 8  # 必须设置为8
   ```

3. **启用多帧时序训练**:
   ```yaml
   data:
     num_temporal_frames: 4  # 使用4帧
     temporal_stride: 1
   ```

4. **启用辅助损失**:
   ```yaml
   training:
     loss:
       use_auxiliary_loss: true
       geom_consistency_weight: 0.1
       feature_reg_weight: 0.01
   ```

## 🔍 验证改进

### 检查Wandb是否工作

训练开始后，应该看到：
```
Wandb enabled - experiment tracking will be saved
Wandb initialized: https://wandb.ai/...
```

如果看到：
```
Wandb disabled - set wandb.enabled: true in config to enable
```
说明wandb未启用，检查配置文件。

### 检查架构改进是否生效

1. **检查日志**:
   - 如果启用动作归一化，应该看到: `动作归一化: 已启用`
   - 如果启用多帧时序，应该看到: `使用多帧时序训练: 4 帧`

2. **检查模型参数**:
   ```python
   # 检查action_head是否使用四元数
   print(model.action_head.use_quaternion)
   
   # 检查fusion是否使用attention pooling
   print(model.fusion.use_attention_pooling)
   ```

3. **检查Loss**:
   ```python
   # 检查是否启用辅助损失
   print(trainer.criterion.use_auxiliary_loss)
   ```

## 🐛 故障排除

### Wandb问题

**Q: Wandb初始化失败？**
- 检查是否已登录: `wandb login`
- 检查网络连接
- 检查API key是否正确

**Q: Wandb没有记录数据？**
- 确认 `wandb.enabled: true`
- 确认是rank 0进程（分布式训练）
- 检查日志中是否有wandb错误信息

**Q: 如何离线使用wandb？**
```bash
export WANDB_MODE=offline
python atlas/train.py --config atlas/configs/train_config.yaml
```

### 架构改进问题

**Q: 动作归一化后loss变小？**
- 这是正常的，因为动作被归一化到更小的范围
- 关注相对改进，而不是绝对loss值

**Q: 四元数模式下action_dim不匹配？**
- 确保 `action_dim: 8` 当 `use_quaternion: true`
- 检查模型和dataset的action_dim是否一致

**Q: 多帧训练显存不足？**
- 减少 `num_temporal_frames`
- 减少 `batch_size`
- 增加 `gradient_accumulation_steps`

## 📊 改进效果

### Wandb改进

- ✅ Wandb正确初始化和记录
- ✅ 代码自动保存到wandb
- ✅ 完整的训练配置记录
- ✅ 错误处理，不影响训练

### 架构改进集成

- ✅ 所有6项改进都已集成到训练代码
- ✅ 可以通过配置文件启用/禁用
- ✅ 向后兼容，默认行为不变

## ✅ 检查清单

训练前确认：

- [ ] Wandb已登录 (`wandb login`)
- [ ] 配置文件中 `wandb.enabled: true`（如果需要）
- [ ] 模型参数配置正确（use_quaternion, action_dim等）
- [ ] Dataset参数配置正确（normalize_actions, num_temporal_frames等）
- [ ] Loss参数配置正确（use_auxiliary_loss等）
- [ ] 检查日志确认所有改进都已启用

---

**最后更新**: 2026-02-07
**作者**: Atlas VLA Team
