#!/bin/bash
# Setup script for AtlasVLA project

set -e

echo "Setting up AtlasVLA project..."

# Check Python version
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "Python version: $python_version"

# Create virtual environment (optional)
if [ "$1" == "--venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
    source venv/bin/activate
    echo "Virtual environment activated. To activate manually: source venv/bin/activate"
fi

# Install dependencies
echo "Installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

# Install VGGT as a package (optional, for better import handling)
if [ -d "vggt" ]; then
    echo "Installing VGGT as a package..."
    pip install -e vggt/ || echo "Warning: Could not install VGGT as package. Continuing..."
fi

# Install AtlasVLA in development mode
echo "Installing AtlasVLA in development mode..."
pip install -e .

echo ""
echo "Setup complete!"
echo ""
echo "To verify installation, run:"
echo "  python atlas/test_imports.py"
echo ""
echo "To start training, run:"
echo "  python atlas/train.py --config atlas/configs/train_config.yaml"
