# Quick Start Guide

## 🚀 Get Started in 3 Steps

### 1. Clone and Install

```bash
# Clone the repository (VGGT code is included directly)
git clone https://github.com/Tingting-Olivia-Du/AtlasVLA.git
cd AtlasVLA

# Install in development mode
pip install -e .
```

### 2. Verify Installation

```bash
# Test imports
python atlas/test_imports.py
```

### 3. Run Example

```bash
# Run example usage (requires GPU and model downloads)
python atlas/example_usage.py
```

## 📚 Next Steps

- **Training**: See [Training Guide](atlas/README_TRAINING.md)
- **Installation**: See [Install Guide](INSTALL.md)
- **Contributing**: See [Contributing Guide](CONTRIBUTING.md)

## 🎯 Common Tasks

### Train on LIBERO Dataset

```bash
# 1. Download and prepare LIBERO dataset
# 2. Update config: atlas/configs/train_config.yaml
# 3. Start training
python atlas/train.py --config atlas/configs/train_config.yaml
```

### Evaluate Model

```bash
python atlas/eval.py \
  --config atlas/configs/train_config.yaml \
  --checkpoint checkpoints/best_model.pt \
  --split val
```

### Use Model in Your Code

```python
from atlas.src.models import VGGTVLA
import torch

model = VGGTVLA(
    lang_encoder_name="meta-llama/Llama-2-7b-hf",
    freeze_vggt=True
)
model = model.to("cuda")

# Your code here...
```

## ⚠️ Important Notes

1. **LLaMA 2 Access**: You need to request access from Meta and authenticate with HuggingFace
2. **GPU Required**: Training and inference require CUDA-capable GPU
3. **VGGT Model**: Will be automatically downloaded on first use (~5GB)

## 🆘 Need Help?

- Check [README.md](README.md) for overview
- Check [INSTALL.md](INSTALL.md) for installation issues
- Open an [Issue](https://github.com/Tingting-Olivia-Du/AtlasVLA/issues) for bugs
