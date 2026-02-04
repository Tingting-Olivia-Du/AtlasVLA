# Installation Guide

## Quick Install

```bash
# Clone the repository
git clone https://github.com/Tingting-Olivia-Du/AtlasVLA.git
cd AtlasVLA

# Install in development mode (recommended)
pip install -e .
```

## Detailed Installation

### Option 1: Development Installation (Recommended)

This installs the package in editable mode, so changes to the code are immediately reflected.

```bash
# Install AtlasVLA
pip install -e .

# Install optional dependencies
pip install -e ".[wandb]"  # For experiment tracking
pip install -e ".[dev]"     # For development tools
```

### Option 2: Manual Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Install VGGT as a package (optional, for better import handling)
pip install -e vggt/

# Add to Python path (if not using pip install)
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
```

### Option 3: Using Setup Script

```bash
# Run setup script
bash scripts/setup.sh

# Or with virtual environment
bash scripts/setup.sh --venv
```

## Verify Installation

```bash
# Test imports
python atlas/test_imports.py

# Or test basic functionality
python atlas/example_usage.py
```

## Troubleshooting

### Import Errors

If you encounter import errors:

1. **Make sure you're in the project root directory**
2. **Install in development mode**: `pip install -e .`
3. **Check Python path**: `echo $PYTHONPATH`
4. **Verify VGGT is accessible**: `python -c "import vggt; print(vggt.__file__)"`

### CUDA/GPU Issues

- Ensure PyTorch is installed with CUDA support: `pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118`
- Check GPU availability: `python -c "import torch; print(torch.cuda.is_available())"`

### LLaMA 2 Access

If you need access to LLaMA 2:

1. Request access from Meta: https://ai.meta.com/llama/
2. Authenticate with HuggingFace: `huggingface-cli login`
3. Or use an alternative language encoder (modify config)

## Dependencies

Core dependencies are listed in `requirements.txt`. Key packages:

- **torch**: PyTorch >= 2.0.0
- **transformers**: HuggingFace Transformers >= 4.30.0
- **numpy**: >= 1.24.0
- **einops**: For tensor operations

See `requirements.txt` for the complete list.
