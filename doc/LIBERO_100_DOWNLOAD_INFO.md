# LIBERO_100 下载信息

## 📦 磁盘空间需求

### LIBERO_100 数据集大小

| 数据集 | 大小 | 说明 |
|--------|------|------|
| **LIBERO_100** | **~100 GB** | 包含LIBERO_10和LIBERO_90 |
| LIBERO_10 | ~10 GB | 10个任务（LIBERO_100的子集） |
| LIBERO_90 | ~90 GB | 90个任务（LIBERO_100的子集） |

### 实际下载大小

- **压缩包大小**: 约50-80 GB（下载时）
- **解压后大小**: 约100 GB
- **HuggingFace缓存**: 可能额外占用20-50 GB（如果使用HuggingFace下载）

### 推荐磁盘空间

- **最小**: 150 GB（仅LIBERO_100数据集）
- **推荐**: 200-300 GB（数据集 + 转换后的Atlas格式 + 训练checkpoints）
- **安全**: 500 GB+（包含所有数据和中间文件）

---

## 📁 下载位置

### 默认下载路径

LIBERO数据集默认下载到以下位置之一：

1. **如果设置了LIBERO配置**:
   ```
   ~/.libero/datasets/
   ```
   或自定义路径（首次运行时会询问）

2. **如果使用项目内路径**:
   ```
   dataset/LIBERO/../datasets/
   ```
   即相对于LIBERO包位置的 `../datasets/` 目录

### 查看默认路径

运行以下命令查看默认下载路径：

```bash
cd dataset/LIBERO
python3 -c "from libero.libero import get_libero_path; import os; print(os.path.abspath(get_libero_path('datasets')))"
```

### 指定自定义下载路径

#### 方法1: 使用命令行参数

```bash
cd dataset/LIBERO
python benchmark_scripts/download_libero_datasets.py \
    --datasets libero_100 \
    --download-dir /path/to/your/custom/directory \
    --use-huggingface
```

#### 方法2: 设置环境变量

```bash
export LIBERO_CONFIG_PATH=/path/to/custom/libero/config
# 首次运行时会创建配置文件并询问数据集路径
```

#### 方法3: 修改配置文件

编辑 `~/.libero/config.yaml`，修改 `datasets` 路径：

```yaml
datasets: /path/to/your/custom/datasets/path
```

---

## 📥 下载命令

### 从HuggingFace下载（推荐）

```bash
cd dataset/LIBERO
python benchmark_scripts/download_libero_datasets.py \
    --datasets libero_100 \
    --use-huggingface
```

**优点**:
- 下载速度快
- 链接稳定
- 自动处理格式

**缺点**:
- 需要HuggingFace账号（可选，但推荐）
- 会缓存到HuggingFace缓存目录

### 从原始链接下载

```bash
cd dataset/LIBERO
python benchmark_scripts/download_libero_datasets.py \
    --datasets libero_100
```

**注意**: 原始链接可能已过期，不推荐使用。

---

## 🔍 检查下载状态

### 检查数据集完整性

```bash
cd dataset/LIBERO
python benchmark_scripts/download_libero_datasets.py \
    --datasets libero_100 \
    --use-huggingface
# 脚本会自动检查数据集完整性
```

### 手动检查

```bash
# 检查下载目录
DOWNLOAD_DIR=$(python3 -c "from libero.libero import get_libero_path; print(get_libero_path('datasets'))")
echo "下载目录: $DOWNLOAD_DIR"

# 检查LIBERO_100是否存在
if [ -d "$DOWNLOAD_DIR/libero_100" ]; then
    echo "✓ LIBERO_100已下载"
    # 统计HDF5文件数量（应该有100个任务文件）
    COUNT=$(find "$DOWNLOAD_DIR/libero_100" -name "*.hdf5" | wc -l)
    echo "  找到 $COUNT 个任务文件（应该是100个）"
    
    # 检查磁盘使用
    du -sh "$DOWNLOAD_DIR/libero_100"
else
    echo "✗ LIBERO_100未找到"
fi
```

---

## 💾 磁盘空间管理

### 下载前检查

```bash
# 检查可用磁盘空间
df -h /path/to/download/directory

# 推荐至少150GB可用空间
```

### 清理HuggingFace缓存（如果使用HuggingFace）

```bash
# 查看HuggingFace缓存位置
python3 -c "from huggingface_hub import HfFolder; print(HfFolder.get_cache_dir())"

# 清理缓存（谨慎操作）
# rm -rf ~/.cache/huggingface/hub/datasets--yifengzhu-hf--LIBERO-datasets
```

### 转换后清理原始数据（可选）

转换完成后，如果不需要保留HDF5格式，可以删除原始数据：

```bash
# 谨慎操作！确保转换成功后再删除
# rm -rf /path/to/libero_100
```

---

## 📊 下载后的目录结构

```
下载目录/
└── libero_100/
    ├── task_0.hdf5
    ├── task_1.hdf5
    ├── ...
    └── task_99.hdf5  (共100个任务文件)
```

每个HDF5文件包含：
- 观测数据（图像）
- 动作数据
- 元数据

---

## ⚠️ 注意事项

1. **磁盘空间**: 确保有足够的磁盘空间（推荐200GB+）
2. **网络连接**: 下载100GB数据需要稳定的网络连接
3. **下载时间**: 根据网速，可能需要数小时到一天
4. **HuggingFace账号**: 虽然不需要登录，但登录后可能有更好的下载速度
5. **路径权限**: 确保对下载目录有写权限

---

## 🚀 快速开始

### 完整下载流程

```bash
# 1. 进入LIBERO目录
cd dataset/LIBERO

# 2. 检查可用空间（推荐至少150GB）
df -h .

# 3. 下载LIBERO_100（从HuggingFace）
python benchmark_scripts/download_libero_datasets.py \
    --datasets libero_100 \
    --use-huggingface

# 4. 检查下载结果
python benchmark_scripts/download_libero_datasets.py \
    --datasets libero_100 \
    --use-huggingface
# 脚本会自动检查完整性

# 5. 查看下载位置
python3 -c "from libero.libero import get_libero_path; import os; print('下载位置:', os.path.abspath(get_libero_path('datasets')))"
```

---

## 📝 总结

| 项目 | 信息 |
|------|------|
| **数据集大小** | ~100 GB |
| **推荐磁盘空间** | 200-300 GB |
| **默认下载位置** | `~/.libero/datasets/` 或 `dataset/LIBERO/../datasets/` |
| **下载方式** | HuggingFace（推荐）或原始链接 |
| **文件格式** | HDF5（需要转换为Atlas格式） |
| **任务数量** | 100个任务（包含LIBERO_10的10个任务） |

下载完成后，使用 `atlas/scripts/convert_libero_to_atlas_format.py` 转换为Atlas格式。
