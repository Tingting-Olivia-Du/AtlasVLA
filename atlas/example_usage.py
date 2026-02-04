"""
Example usage of Atlas VLA model
"""

import torch
import sys
import os

# Add project root to path for development (if not installed as package)
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

try:
    from atlas.src.models import VGGTVLA
except ImportError:
    # Fallback for direct script execution
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../vggt'))
    from atlas.src.models import VGGTVLA


def main():
    # Check device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Initialize model
    print("Initializing VGGTVLA model...")
    model = VGGTVLA(
        lang_encoder_name="meta-llama/Llama-2-7b-hf",  # You may need to authenticate for this
        freeze_vggt=True,
        freeze_lang_encoder=False,
        geom_output_dim=512,
        fusion_hidden_dim=1024,
        action_dim=7
    )
    model = model.to(device)
    model.eval()
    
    print(f"Model initialized. Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    # Prepare dummy inputs
    batch_size = 1
    num_frames = 2  # Workspace + wrist camera
    images = torch.randn(batch_size, num_frames, 3, 518, 518).to(device)
    language_instruction = ["Pick up the red block"]
    
    print(f"\nInput shapes:")
    print(f"  Images: {images.shape}")
    print(f"  Language: {language_instruction}")
    
    # Forward pass
    print("\nRunning forward pass...")
    with torch.no_grad():
        outputs = model(images, language_instruction, return_intermediates=True)
    
    print(f"\nOutput shapes:")
    print(f"  Action: {outputs['action'].shape}")
    print(f"  Pose: {outputs['pose'].shape}")
    print(f"  Gripper: {outputs['gripper'].shape}")
    print(f"  Geometry features: {outputs['geometry_features'].shape}")
    print(f"  Language features: {outputs['language_features'].shape}")
    print(f"  Fused features: {outputs['fused_features'].shape}")
    
    print("\nExample completed successfully!")


if __name__ == "__main__":
    main()
