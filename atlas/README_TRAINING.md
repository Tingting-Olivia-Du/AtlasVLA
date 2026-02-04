# Atlas VLA Training Guide

## Overview

This guide explains how to finetune the Atlas VLA model on LIBERO dataset.

## Dataset Preparation

### LIBERO Dataset Format

The LIBERO dataset should be organized as follows:

```
libero_data/
  train/
    episode_000/
      images/
        workspace_000.png
        workspace_001.png
        ...
        wrist_000.png
        wrist_001.png
        ...
      actions.parquet  # or actions.csv
      language_task.txt
    episode_001/
      ...
  val/
    episode_000/
      ...
```

### Data Format

- **Images**: 256×256×3 RGB images (workspace and wrist cameras)
- **Actions**: 7-dimensional vectors (6-DOF end-effector pose + gripper)
- **Language**: Task descriptions in text files

### Downloading LIBERO

You can download LIBERO datasets from:
- [LIBERO Project Website](https://libero-project.github.io/datasets)
- [HuggingFace Datasets](https://huggingface.co/datasets/lerobot/libero_object_image)

## Installation

```bash
# Install dependencies
pip install -r atlas/requirements.txt
pip install wandb  # Optional: for experiment tracking
pip install pyyaml  # For config files

# Install VGGT
pip install -e vggt/
```

## Configuration

Edit `atlas/configs/train_config.yaml` to set:

1. **Data path**: Update `data.data_dir` to your LIBERO dataset path
2. **Model settings**: Adjust model architecture if needed
3. **Training hyperparameters**: Learning rate, batch size, etc.
4. **Loss weights**: Adjust `pose_weight` and `gripper_weight` if needed

## Training

### Basic Training

```bash
python atlas/train.py --config atlas/configs/train_config.yaml
```

### Resume Training

```bash
python atlas/train.py \
  --config atlas/configs/train_config.yaml \
  --resume checkpoints/checkpoint_epoch_10.pt
```

### Multi-GPU Training

```bash
# Using torchrun (recommended)
torchrun --nproc_per_node=4 atlas/train.py --config atlas/configs/train_config.yaml

# Or using DDP
python -m torch.distributed.launch \
  --nproc_per_node=4 \
  atlas/train.py --config atlas/configs/train_config.yaml
```

## Training Strategy

### Phase 1: Freeze VGGT (Recommended Start)

- Freeze VGGT backbone: `freeze_vggt: true`
- Train only fusion and action head
- Faster training, lower memory usage
- Good for initial experiments

### Phase 2: Unfreeze Language Encoder

- Set `freeze_lang_encoder: false`
- Fine-tune language encoder for task-specific understanding
- May improve performance on specific tasks

### Phase 3: End-to-End (Optional)

- Unfreeze VGGT: `freeze_vggt: false`
- Requires more GPU memory
- May improve performance but risk of overfitting

## Monitoring Training

### TensorBoard (if implemented)

```bash
tensorboard --logdir checkpoints/
```

### Wandb

Set `wandb.enabled: true` in config and login:

```bash
wandb login
```

## Checkpoints

Checkpoints are saved in `checkpoints/` directory:
- `checkpoint_step_*.pt`: Periodic checkpoints
- `checkpoint_epoch_*.pt`: End-of-epoch checkpoints
- `best_model.pt`: Best validation model

## Evaluation

See `atlas/eval.py` (to be implemented) for evaluation scripts.

## Troubleshooting

### Out of Memory

- Reduce `batch_size` in config
- Use gradient accumulation (to be implemented)
- Reduce `image_size` (may affect performance)

### Slow Training

- Ensure VGGT is frozen if not needed
- Use mixed precision (already enabled)
- Reduce number of workers if data loading is bottleneck

### Poor Convergence

- Check learning rate (try 1e-5 or 5e-5)
- Adjust loss weights
- Verify data format and labels
- Check if language encoder is working correctly

## Expected Training Time

- Single GPU (RTX 3090): ~2-3 days for 50 epochs on LIBERO-10
- Multi-GPU (4x A100): ~12-18 hours for 50 epochs

## Next Steps

1. Implement evaluation metrics
2. Add data augmentation
3. Implement gradient accumulation
4. Add support for more datasets
5. Implement action space normalization
