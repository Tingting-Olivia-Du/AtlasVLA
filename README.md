# AtlasVLA

Vision-Language-Action (VLA) model for robot manipulation tasks, built on top of [VGGT](https://github.com/facebookresearch/vggt).

## Overview

AtlasVLA integrates 3D geometric understanding from VGGT with language instructions to predict robot actions. The model combines:

- **VGGT Backbone**: Extracts rich 3D geometric information (depth maps, point clouds, camera poses) from images
- **Language Encoder**: Processes natural language instructions using LLaMA 2 encoder
- **Multimodal Fusion**: Cross-attention mechanism to fuse language and 3D geometry
- **Action Prediction**: Outputs end-effector pose (6-DOF) and gripper actions

## Features

- 🎯 **3D-Aware**: Leverages VGGT's powerful 3D scene understanding
- 🗣️ **Language-Conditioned**: Understands natural language task descriptions
- 🤖 **Action Prediction**: Directly predicts robot actions for manipulation
- 📊 **LIBERO Support**: Ready-to-use training on LIBERO manipulation dataset
- 🔧 **Flexible**: Supports freezing/unfreezing different components for efficient training

## Installation

### Prerequisites

- Python >= 3.8
- PyTorch >= 2.0.0
- CUDA-capable GPU (recommended)

### Install from Source

```bash
# Clone the repository
git clone https://github.com/yourusername/AtlasVLA.git
cd AtlasVLA

# Install in development mode
pip install -e .

# Or install with optional dependencies
pip install -e ".[wandb]"  # For wandb experiment tracking
pip install -e ".[dev]"    # For development tools
```

### Manual Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Install VGGT as a package (optional, for better import handling)
pip install -e vggt/
```

## Quick Start

### Basic Usage

```python
import torch
from atlas.src.models import VGGTVLA

# Initialize model
model = VGGTVLA(
    lang_encoder_name="meta-llama/Llama-2-7b-hf",
    freeze_vggt=True,
    freeze_lang_encoder=False
)
model = model.to("cuda")

# Prepare inputs
images = torch.randn(1, 2, 3, 518, 518).to("cuda")  # [B, S, 3, H, W]
language_instruction = ["Pick up the red block"]

# Forward pass
outputs = model(images, language_instruction)
action = outputs["action"]  # [B, 7] - 6-DOF pose + gripper
pose = outputs["pose"]      # [B, 6]
gripper = outputs["gripper"]  # [B, 1]
```

### Training on LIBERO

1. **Prepare your data**: Download and organize LIBERO dataset (see [Training Guide](atlas/README_TRAINING.md))

2. **Configure training**: Edit `atlas/configs/train_config.yaml`

3. **Start training**:
```bash
python atlas/train.py --config atlas/configs/train_config.yaml
```

4. **Evaluate**:
```bash
python atlas/eval.py \
  --config atlas/configs/train_config.yaml \
  --checkpoint checkpoints/best_model.pt \
  --split val
```

## Project Structure

```
AtlasVLA/
├── atlas/                    # Main Atlas VLA code
│   ├── src/
│   │   ├── models/          # Model definitions
│   │   ├── data/            # Data loaders
│   │   └── training/        # Training utilities
│   ├── configs/             # Configuration files
│   ├── train.py             # Training script
│   ├── eval.py              # Evaluation script
│   └── README_TRAINING.md   # Detailed training guide
│
├── vggt/                    # VGGT dependency (submodule)
│   └── ...                  # VGGT original code
│
├── setup.py                 # Package installation
├── pyproject.toml           # Modern Python project config
├── requirements.txt         # Dependencies
└── README.md                # This file
```

## Model Architecture

```
Input: RGB Images [B, S, 3, H, W] + Language Instructions
  ↓
┌─────────────────────────────────────────┐
│  VGGT Backbone (frozen or trainable)   │
│  - Aggregator extracts visual features │
│  - Outputs: 3D geometry information     │
└─────────────────────────────────────────┘
  ↓
┌─────────────────────────────────────────┐
│  3D Geometry Feature Extractor          │
│  - Token features from VGGT            │
│  - Optional: Point cloud features      │
│  - Optional: Camera pose encoding      │
└─────────────────────────────────────────┘
  ↓                    ↓
┌──────────────┐  ┌──────────────┐
│  Language    │  │  3D Geometry │
│  Encoder     │  │  Features    │
│  (LLaMA 2)   │  │              │
└──────────────┘  └──────────────┘
  ↓                    ↓
┌─────────────────────────────────────────┐
│  Multimodal Fusion (Cross-Attention)    │
│  - Language queries attend to geometry  │
└─────────────────────────────────────────┘
  ↓
┌─────────────────────────────────────────┐
│  Action Head                            │
│  - End-effector pose (6-DOF)            │
│  - Gripper action                       │
└─────────────────────────────────────────┘
```

## Training Strategy

### Phase 1: Freeze VGGT (Recommended Start)
- Freeze VGGT backbone
- Train only fusion and action head
- Fast training, low memory usage

### Phase 2: Unfreeze Language Encoder
- Fine-tune language encoder
- Better task-specific understanding

### Phase 3: End-to-End (Optional)
- Unfreeze VGGT
- Requires more GPU memory
- May improve performance

## Configuration

Key configuration options in `atlas/configs/train_config.yaml`:

- **Model**: VGGT checkpoint, language encoder, freeze options
- **Data**: Dataset path, batch size, image size
- **Training**: Learning rate, epochs, loss weights
- **Checkpointing**: Save directory, intervals

## Datasets

Currently supports:
- **LIBERO**: Manipulation benchmark with 130 tasks
  - RGB images (workspace + wrist cameras)
  - 7-DOF actions (6-DOF pose + gripper)
  - Language task descriptions

## Citation

If you use AtlasVLA in your research, please cite:

```bibtex
@misc{atlasvla2025,
  title={AtlasVLA: Vision-Language-Action Model with 3D Geometric Understanding},
  author={Your Name},
  year={2025},
  howpublished={\url{https://github.com/yourusername/AtlasVLA}}
}
```

And the original VGGT paper:

```bibtex
@inproceedings{wang2025vggt,
  title={VGGT: Visual Geometry Grounded Transformer},
  author={Wang, Jianyuan and Chen, Minghao and Karaev, Nikita and Vedaldi, Andrea and Rupprecht, Christian and Novotny, David},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  year={2025}
}
```

## License

This project follows the VGGT license. See [LICENSE](LICENSE) for details.

## Acknowledgments

- Built on [VGGT](https://github.com/facebookresearch/vggt) by Meta AI
- Uses [LLaMA 2](https://ai.meta.com/llama/) for language encoding
- Trained on [LIBERO](https://libero-project.github.io/) manipulation dataset

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Contact

For questions or issues, please open an issue on GitHub.
