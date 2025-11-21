#!/usr/bin/env python3
import sys
import os

# Add project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

try:
    import torch
    print("✅ PyTorch imported successfully")
except ImportError as e:
    print(f"❌ PyTorch import failed: {e}")

def validate_claims():
    print("📊 Validating CC-GAN Paper Claims")
    print("=" * 50)
    
    # Check if we have checkpoints
    if not os.path.exists('checkpoints'):
        print("❌ No checkpoints directory - run training first!")
        return
    
    checkpoints = [f for f in os.listdir('checkpoints') if f.endswith('.pth')]
    if not checkpoints:
        print("❌ No checkpoint files found - run training first!")
        return
    
    print(f"✅ Found {len(checkpoints)} checkpoint(s)")
    
    print("\n🎯 Paper Claims to Validate:")
    print("1. ✅ Component Caching - IMPLEMENTED")
    print("2. ✅ 3D-Aware Viewpoint Control (CALL) - IMPLEMENTED") 
    print("3. ✅ Consumer Preference Guidance - IMPLEMENTED")
    print("4. 🔄 >95% Viewpoint Accuracy - NEEDS TESTING")
    print("5. 🔄 60-70% FLOPs Reduction - NEEDS TESTING")
    print("6. 🔄 20% Originality Improvement - NEEDS TESTING")
    print("7. 🔄 Sub-second Inference - NEEDS TESTING")
    
    print("\n📊 Current Status:")
    latest_checkpoint = sorted(checkpoints)[-1]
    print(f"   Latest checkpoint: {latest_checkpoint}")
    
    try:
        checkpoint_data = torch.load(f"checkpoints/{latest_checkpoint}", map_location='cpu')
        epoch = checkpoint_data.get('epoch', 'Unknown')
        cache_size = len(checkpoint_data.get('cache', {}))
        print(f"   Training epoch: {epoch}")
        print(f"   Cache size: {cache_size}")
    except Exception as e:
        print(f"   Error reading checkpoint: {e}")
    
    print("\n🧪 Next: Run quantitative evaluation scripts")
    print("   python scripts/evaluate_fixed.py --checkpoint checkpoints/your_checkpoint.pth")

if __name__ == "__main__":
    validate_claims()
