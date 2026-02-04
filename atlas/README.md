# Atlas VLA

Vision-Language-Action model based on VGGT for robot manipulation tasks.

## Architecture

- **VGGT Backbone**: Extracts 3D geometric information from images
- **Language Encoder**: LLaMA 2 7B encoder for processing language instructions
- **Geometry Encoder**: Extracts features from VGGT outputs (tokens, point clouds, camera poses)
- **Multimodal Fusion**: Cross-attention mechanism to fuse language and geometry features
- **Action Head**: Predicts end-effector pose (6-DOF) and gripper action

## Installation

```bash
# Install dependencies
pip install torch torchvision transformers
pip install -e ../vggt  # Install VGGT from parent directory
```

## Usage

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

## Model Components

- `GeometryFeatureExtractor`: Extracts 3D features from VGGT outputs
- `MultimodalFusion`: Fuses language and geometry using cross-attention
- `ActionHead`: Predicts robot actions
- `VGGTVLA`: Main model integrating all components

## Training

See `training/` directory for training scripts (to be implemented).

## License

Follows VGGT license (see parent directory).
