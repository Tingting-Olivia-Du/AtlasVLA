# 分布式训练概念详解

## Rank 和 Local Rank 是什么意思？

### 基本概念

在**多GPU分布式训练**中，每个GPU进程都有一个唯一的标识符：

```
Rank: 0, Local rank: 0
Device: cuda:0
```

### 详细解释

#### 1. **Rank（全局Rank）**
- **含义**: 在所有GPU中的全局编号（从0开始）
- **范围**: 0 到 (GPU总数 - 1)
- **作用**: 用于区分不同的GPU进程

**示例（8个GPU）**:
```
GPU 0: Rank = 0
GPU 1: Rank = 1
GPU 2: Rank = 2
...
GPU 7: Rank = 7
```

#### 2. **Local Rank（本地Rank）**
- **含义**: 在当前机器上的GPU编号
- **范围**: 0 到 (当前机器的GPU数 - 1)
- **作用**: 用于指定使用哪个物理GPU

**单机多GPU（你的情况）**:
```
物理GPU 0: Local rank = 0, Rank = 0
物理GPU 1: Local rank = 1, Rank = 1
物理GPU 2: Local rank = 2, Rank = 2
...
物理GPU 7: Local rank = 7, Rank = 7
```

**多机多GPU（如果有多个服务器）**:
```
机器1:
  物理GPU 0: Local rank = 0, Rank = 0
  物理GPU 1: Local rank = 1, Rank = 1

机器2:
  物理GPU 0: Local rank = 0, Rank = 2  ← 注意Rank是全局的
  物理GPU 1: Local rank = 1, Rank = 3
```

#### 3. **Device（设备）**
- **含义**: PyTorch中使用的GPU设备编号
- **格式**: `cuda:0`, `cuda:1`, 等
- **作用**: 告诉PyTorch使用哪个GPU

**注意**: 
- 如果使用 `CUDA_VISIBLE_DEVICES`，设备编号会重新映射
- 例如：`CUDA_VISIBLE_DEVICES=2,3,4,5` 时
  - 物理GPU 2 → `cuda:0` (逻辑设备0)
  - 物理GPU 3 → `cuda:1` (逻辑设备1)
  - 物理GPU 4 → `cuda:2` (逻辑设备2)
  - 物理GPU 5 → `cuda:3` (逻辑设备3)

## 实际例子

### 你的情况（8个GPU，单机）

当你运行：
```bash
torchrun --nproc_per_node=8 atlas/train.py
```

会启动8个进程，每个进程看到：

**进程0（主进程）**:
```
Rank: 0, Local rank: 0
Device: cuda:0
```
- 使用物理GPU 0
- 负责日志输出、保存checkpoint等

**进程1**:
```
Rank: 1, Local rank: 1
Device: cuda:1
```
- 使用物理GPU 1

**进程2**:
```
Rank: 2, Local rank: 2
Device: cuda:2
```
- 使用物理GPU 2

... 以此类推

### 如果指定GPU IDs

如果你运行：
```bash
./train.sh --mode multi --gpu-ids 0,2,4,6
```

会启动4个进程：

**进程0**:
```
Rank: 0, Local rank: 0
Device: cuda:0  ← 这是逻辑设备0，对应物理GPU 0
```

**进程1**:
```
Rank: 1, Local rank: 1
Device: cuda:1  ← 这是逻辑设备1，对应物理GPU 2
```

**进程2**:
```
Rank: 2, Local rank: 2
Device: cuda:2  ← 这是逻辑设备2，对应物理GPU 4
```

**进程3**:
```
Rank: 3, Local rank: 3
Device: cuda:3  ← 这是逻辑设备3，对应物理GPU 6
```

## 为什么需要这些概念？

### 1. **数据分配**
每个rank处理不同的数据batch：
- Rank 0: 处理 batch 0, 8, 16, ...
- Rank 1: 处理 batch 1, 9, 17, ...
- Rank 2: 处理 batch 2, 10, 18, ...
- ...

### 2. **梯度同步**
所有rank计算完梯度后，需要同步：
- 每个rank计算自己的梯度
- 通过NCCL（NVIDIA Collective Communications Library）同步
- 所有rank得到相同的平均梯度

### 3. **日志和保存**
通常只有 **Rank 0**（主进程）会：
- 输出日志到控制台
- 保存checkpoint
- 记录到wandb
- 执行验证

这样可以避免重复输出和文件冲突。

## 在你的代码中

### train.py 中的使用

```python
# 设置分布式训练
is_distributed, rank, world_size, local_rank, device = setup_distributed()

# 只有rank 0打印信息
if rank == 0:
    print("Initializing model...")

# 使用local_rank设置GPU设备
torch.cuda.set_device(local_rank)
device = torch.device(f'cuda:{local_rank}')

# 包装模型时使用local_rank
model = DDP(model, device_ids=[local_rank], output_device=local_rank)
```

### trainer.py 中的使用

```python
# 只有rank 0记录到wandb
use_wandb=config.get("wandb", {}).get("enabled", False) and rank == 0

# 只有rank 0保存checkpoint
if rank == 0:
    self.save_checkpoint("best_model.pt")
```

## 常见问题

### Q: 为什么只看到 Rank 0 的输出？

**A**: 因为代码中设置了 `if rank == 0:`，只有主进程（rank 0）会输出日志，避免重复。

### Q: 如何看到所有rank的输出？

**A**: 可以修改代码，移除 `if rank == 0:` 条件，但输出会很混乱。

### Q: Rank 0 和其他rank有什么区别？

**A**: 
- **Rank 0（主进程）**:
  - 输出日志
  - 保存checkpoint
  - 记录wandb
  - 执行验证
  
- **其他rank（工作进程）**:
  - 只训练，不输出
  - 同步梯度
  - 处理数据

### Q: Local rank 和 Rank 什么时候不同？

**A**: 只在**多机多GPU**训练时不同。单机多GPU时，它们相同。

### Q: 如果训练卡住了，如何知道是哪个GPU的问题？

**A**: 可以临时修改代码，让所有rank都输出：
```python
# 修改前
if rank == 0:
    print(f"Step {step}: Loss={loss}")

# 修改后
print(f"Rank {rank}, Step {step}: Loss={loss}")
```

## 可视化理解

### 单机8GPU训练

```
┌─────────────────────────────────────────┐
│           你的服务器（单机）              │
├─────────────────────────────────────────┤
│                                         │
│  GPU 0  →  Rank 0, Local rank 0        │ ← 主进程（输出日志）
│  GPU 1  →  Rank 1, Local rank 1        │
│  GPU 2  →  Rank 2, Local rank 2        │
│  GPU 3  →  Rank 3, Local rank 3        │
│  GPU 4  →  Rank 4, Local rank 4        │
│  GPU 5  →  Rank 5, Local rank 5        │
│  GPU 6  →  Rank 6, Local rank 6        │
│  GPU 7  →  Rank 7, Local rank 7        │
│                                         │
└─────────────────────────────────────────┘
```

### 多机多GPU训练（示例）

```
┌──────────────────────┐    ┌──────────────────────┐
│   机器1（服务器A）     │    │   机器2（服务器B）     │
├──────────────────────┤    ├──────────────────────┤
│                      │    │                      │
│  GPU 0 → Rank 0      │    │  GPU 0 → Rank 4      │
│         Local 0      │    │         Local 0      │
│                      │    │                      │
│  GPU 1 → Rank 1      │    │  GPU 1 → Rank 5      │
│         Local 1      │    │         Local 1      │
│                      │    │                      │
│  GPU 2 → Rank 2      │    │  GPU 2 → Rank 6      │
│         Local 2      │    │         Local 2      │
│                      │    │                      │
│  GPU 3 → Rank 3      │    │  GPU 3 → Rank 7      │
│         Local 3      │    │         Local 3      │
│                      │    │                      │
└──────────────────────┘    └──────────────────────┘
```

## 总结

| 概念 | 含义 | 你的情况（8GPU单机） |
|------|------|---------------------|
| **Rank** | 全局GPU编号 | 0-7 |
| **Local Rank** | 本地GPU编号 | 0-7（和Rank相同） |
| **Device** | PyTorch设备 | cuda:0 到 cuda:7 |
| **World Size** | 总GPU数 | 8 |

**简单记忆**:
- **Rank**: "我是第几个GPU"（全局）
- **Local Rank**: "我在当前机器上是第几个GPU"（本地）
- **Device**: "PyTorch叫我用哪个GPU"

在你的情况下（单机8GPU），Rank和Local Rank是相同的，都是0-7。
