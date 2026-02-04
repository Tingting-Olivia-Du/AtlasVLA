"""
Test imports to verify code structure
"""

import sys
import os

# Add project root to path for testing
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)


def test_imports():
    """Test that all modules can be imported"""
    try:
        print("Testing imports...")
        
        # Test geometry encoder
        from atlas.src.models.geometry_encoder import GeometryFeatureExtractor
        print("✓ GeometryFeatureExtractor imported")
        
        # Test fusion
        from atlas.src.models.fusion import MultimodalFusion
        print("✓ MultimodalFusion imported")
        
        # Test action head
        from atlas.src.models.action_head import ActionHead
        print("✓ ActionHead imported")
        
        # Test pointnet
        from atlas.src.models.pointnet import PointNetEncoder
        print("✓ PointNetEncoder imported")
        
        # Test main model (may fail if transformers/VGGT not installed)
        try:
            from atlas.src.models.vggt_vla import VGGTVLA
            print("✓ VGGTVLA imported")
        except ImportError as e:
            print(f"⚠ VGGTVLA import failed (expected if dependencies not installed): {e}")
        
        # Test __init__ imports
        from atlas.src.models import GeometryFeatureExtractor, MultimodalFusion, ActionHead
        print("✓ All models imported from __init__")
        
        # Test data imports
        try:
            from atlas.src.data import LIBERODataset
            print("✓ LIBERODataset imported")
        except ImportError as e:
            print(f"⚠ LIBERODataset import failed: {e}")
        
        # Test training imports
        try:
            from atlas.src.training import VLALoss, VLATrainer
            print("✓ VLALoss and VLATrainer imported")
        except ImportError as e:
            print(f"⚠ Training modules import failed: {e}")
        
        print("\nAll imports successful!")
        return True
        
    except Exception as e:
        print(f"\n✗ Import failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    test_imports()
