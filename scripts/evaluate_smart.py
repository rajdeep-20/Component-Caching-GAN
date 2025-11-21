#!/usr/bin/env python3
import sys
import os
import glob

# Add project root
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

def find_checkpoints():
    """Find all available checkpoints"""
    checkpoint_patterns = [
        "checkpoints/ccgan_final_epoch_*.pth",
        "checkpoints/ccgan_simple_*.pth", 
        "checkpoints/ccgan_fixed_*.pth",
        "checkpoints/ccgan_minimal_*.pth",
        "checkpoints/ccgan_*.pth"
    ]
    
    checkpoints = []
    for pattern in checkpoint_patterns:
        checkpoints.extend(glob.glob(pattern))
    
    return sorted(checkpoints)

def evaluate_smart(checkpoint_path=None):
    if checkpoint_path:
        if not os.path.exists(checkpoint_path):
            print(f"❌ Checkpoint not found: {checkpoint_path}")
            return False
        checkpoints = [checkpoint_path]
    else:
        checkpoints = find_checkpoints()
    
    if not checkpoints:
        print("❌ No checkpoints found!")
        print("💡 Run training first: python scripts/train_ccgan_final.py --epochs 3")
        return False
    
    print(f"✅ Found {len(checkpoints)} checkpoint(s)")
    
    for checkpoint in checkpoints:
        print(f"\n📊 Evaluating: {os.path.basename(checkpoint)}")
        print("=" * 50)
        
        try:
            import torch
            checkpoint_data = torch.load(checkpoint, map_location='cpu')
            
            epoch = checkpoint_data.get('epoch', 'Unknown')
            cache_size = len(checkpoint_data.get('cache', {}))
            config = checkpoint_data.get('config', {})
            model_config = config.get('model', {})
            
            print(f"   Epoch: {epoch}")
            print(f"   Cache Size: {cache_size}")
            print(f"   Feature Dim: {model_config.get('feature_dim', 'Unknown')}")
            print(f"   Latent Dim: {model_config.get('latent_dim', 'Unknown')}")
            print(f"   File Size: {os.path.getsize(checkpoint) / 1024 / 1024:.1f} MB")
            
            # Test if model can be loaded
            from models.component_gan_fixed import ComponentGAN
            from models.composition_gan import CompositionGAN
            
            feature_dim = model_config.get('feature_dim', 256)
            latent_dim = model_config.get('latent_dim', 128)
            
            component_gan = ComponentGAN(latent_dim, feature_dim, 512)
            composition_gan = CompositionGAN(3, feature_dim, feature_dim)
            
            component_gan.load_state_dict(checkpoint_data['component_gan_state_dict'])
            composition_gan.load_state_dict(checkpoint_data['composition_gan_state_dict'])
            
            print("   ✅ Model loads successfully!")
            
        except Exception as e:
            print(f"   ❌ Error: {e}")
    
    return True

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', help='Specific checkpoint to evaluate')
    args = parser.parse_args()
    
    print("🔍 Smart Checkpoint Evaluation")
    print("=" * 50)
    
    if args.checkpoint:
        evaluate_smart(args.checkpoint)
    else:
        evaluate_smart()
    
    print("\n🎯 Next: Use a specific checkpoint for detailed evaluation")
    print("   python scripts/evaluate_smart.py --checkpoint checkpoints/your_checkpoint.pth")
