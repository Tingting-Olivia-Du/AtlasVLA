# 评估性能优化指南

## 不同模式下的最佳配置

### 1. **单卡模式**（推荐：`num_procs=8`）

**配置**：
```bash
GPUS=0 NUM_PROCS=8 ./eval/run_eval.sh
```

**工作原理**：
- 使用 `SubprocVectorEnv`（真正的并行）
- 8 个环境同时运行，batch 推理加速
- **性能**：✅ 最快（单卡内）

**适用场景**：
- 只有 1 张 GPU
- 想快速评估单个任务

---

### 2. **多卡模式 - 策略A：每卡单环境**（推荐：`num_procs=1`）

**配置**：
```bash
GPUS=0,1,2,3,4,5,6,7 NUM_PROCS=1 ./eval/run_eval.sh
```

**工作原理**：
- 8 个 GPU，每个 GPU 运行 1 个环境
- 8 个任务并行评估（GPU 级并行）
- **性能**：✅ 最快（多卡场景）

**适用场景**：
- 有多个 GPU（≥任务数）
- 想最快完成所有任务评估

**优势**：
- 真正的并行（8 个 GPU 同时工作）
- 避免 DummyVectorEnv 的串行开销

---

### 3. **多卡模式 - 策略B：每卡多环境**（`num_procs>1`）

**配置**：
```bash
GPUS=0,1,2,3,4,5,6,7 NUM_PROCS=4 ./eval/run_eval.sh
```

**工作原理**：
- 8 个 GPU，每个 GPU 内 4 个环境（DummyVectorEnv，串行）
- Batch 推理可能更快，但环境执行是串行的
- **性能**：⚠️ 较慢（DummyVectorEnv 串行）

**适用场景**：
- GPU 数量 < 任务数量
- 想通过 batch 推理加速（但环境串行）

**劣势**：
- DummyVectorEnv 是顺序执行（for-loop）
- 多卡并行 + 单卡串行 = 性能差

---

## 性能对比

| 模式 | GPU数 | num_procs | 环境类型 | 并行度 | 速度 |
|------|-------|-----------|----------|--------|------|
| 单卡 | 1 | 8 | SubprocVectorEnv | 8x | ⭐⭐⭐⭐⭐ |
| 多卡-A | 8 | 1 | 单环境 | 8x (GPU级) | ⭐⭐⭐⭐⭐ |
| 多卡-B | 8 | 8 | DummyVectorEnv | 8x (GPU级) + 8x (串行) | ⭐⭐ |

## 推荐配置

### 场景1：8 张 GPU，10 个任务
```bash
# ✅ 推荐：每卡单环境，多卡并行
GPUS=0,1,2,3,4,5,6,7 NUM_PROCS=1 ./eval/run_eval.sh
```

### 场景2：1 张 GPU，1 个任务
```bash
# ✅ 推荐：单卡多环境，真正并行
GPUS=0 NUM_PROCS=8 ./eval/run_eval.sh
```

### 场景3：2 张 GPU，10 个任务
```bash
# ✅ 推荐：每卡单环境，多卡并行
GPUS=0,1 NUM_PROCS=1 ./eval/run_eval.sh
```

## 为什么多卡时 num_procs=1 更快？

1. **DummyVectorEnv 是串行的**：
   - `DummyVectorEnv` 内部用 for-loop 顺序执行
   - 8 个环境 = 8 倍时间（不是并行）

2. **多卡并行更高效**：
   - 8 个 GPU × 1 个环境 = 真正的 8 倍并行
   - 8 个 GPU × 8 个环境（串行）= 8 倍 GPU + 8 倍串行 = 慢

3. **SubprocVectorEnv 无法在多进程 worker 中使用**：
   - 多进程 worker 是 daemonic 进程
   - 无法创建子进程（SubprocVectorEnv 需要）

## 总结

**最佳实践**：
- **多卡模式**：`NUM_PROCS=1`（让多卡并行）
- **单卡模式**：`NUM_PROCS=8`（使用 SubprocVectorEnv 真正并行）
