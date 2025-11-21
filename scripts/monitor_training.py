#!/usr/bin/env python3
import torch
import matplotlib.pyplot as plt
import json
import os

def monitor_training():
    print("📊 Training Monitor")
    print("=" * 50)
    
    # Check if checkpoints directory exists
    if not os.path.exists('checkpoints'):
        print("❌ No 'checkpoints' directory found.")
        print("💡 This means:")
        print("   - No training has been completed yet")
        print("   - Or training didn't save any checkpoints")
        print("   - Or you're in the wrong directory")
        print("\n🎯 Run training first:")
        print("   python scripts/train_ccgan_final.py --epochs 3 --batch-size 4 --feature-dim 256")
        return
    
    checkpoints = [f for f in os.listdir('checkpoints') if f.endswith('.pth')]
    
    if not checkpoints:
        print("❌ No checkpoint files found in 'checkpoints' directory.")
        print("💡 Training might have started but no checkpoints saved yet.")
        return
    
    print(f"✅ Found {len(checkpoints)} checkpoint(s)")
    print("-" * 50)
    
    for checkpoint in sorted(checkpoints):
        try:
            data = torch.load(f"checkpoints/{checkpoint}", map_location='cpu')
            epoch = data.get('epoch', 'Unknown')
            cache_size = len(data.get('cache', {}))
            config = data.get('config', {})
            model_config = config.get('model', {})
            
            print(f"📁 Checkpoint: {checkpoint}")
            print(f"   ├── Epoch: {epoch}")
            print(f"   ├── Cache size: {cache_size}")
            print(f"   ├── Feature dim: {model_config.get('feature_dim', 'Unknown')}")
            print(f"   ├── Latent dim: {model_config.get('latent_dim', 'Unknown')}")
            print(f"   └── File size: {os.path.getsize(f'checkpoints/{checkpoint}') / 1024 / 1024:.2f} MB")
            print()
            
        except Exception as e:
            print(f"❌ Error reading {checkpoint}: {e}")
            print()

def check_training_status():
    print("🔍 Training Environment Check")
    print("=" * 50)
    
    # Check current directory
    current_dir = os.getcwd()
    print(f"Current directory: {current_dir}")
    
    # Check if we're in the right place
    if not os.path.exists('models') or not os.path.exists('scripts'):
        print("❌ Not in project root directory!")
        print("💡 Run: cd /home/jupyter/cc-gan-project")
        return False
    
    print("✅ In project root directory")
    
    # Check for checkpoints directory
    if os.path.exists('checkpoints'):
        print("✅ Checkpoints directory exists")
        checkpoints = [f for f in os.listdir('checkpoints') if f.endswith('.pth')]
        print(f"✅ Found {len(checkpoints)} checkpoint files")
    else:
        print("❌ Checkpoints directory doesn't exist yet")
        print("💡 This is normal if you haven't trained yet")
    
    # Check for other important directories
    for dir_name in ['models', 'scripts', 'data', 'configs']:
        if os.path.exists(dir_name):
            print(f"✅ {dir_name} directory exists")
        else:
            print(f"❌ {dir_name} directory missing")
    
    return True

if __name__ == "__main__":
    print("CC-GAN Training Monitor")
    print()
    
    # First check the environment
    if check_training_status():
        print()
        # Then monitor training
        monitor_training()
    
    print("\n🎯 Next Steps:")
    print("1. If no checkpoints: python scripts/train_ccgan_final.py --epochs 3 --batch-size 4 --feature-dim 256")
    print("2. If checkpoints exist: python scripts/demo_ccgan.py --checkpoint checkpoints/your_checkpoint.pth")
    print("3. Continue training: python scripts/train_ccgan_final.py --resume checkpoints/your_checkpoint.pth")
