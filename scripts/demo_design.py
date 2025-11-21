#!/usr/bin/env python3
import sys
import os

# Add project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

def design_demo():
    print("🎨 CC-GAN Design Applications Demo")
    print("=" * 50)
    
    # Check for checkpoints
    if not os.path.exists('checkpoints'):
        print("❌ No checkpoints found. Run training first!")
        print("   python scripts/train_ccgan_final.py --epochs 3 --batch-size 4 --feature-dim 256")
        return
    
    # Test paper applications
    applications = {
        "Product Design": [
            "ergonomic office chair front view",
            "ergonomic office chair side profile", 
            "ergonomic office chair with mesh back"
        ],
        "Fashion Design": [
            "leather biker jacket front view",
            "leather biker jacket back view",
            "denim jacket with embroidery front"
        ],
        "Furniture Design": [
            "modern sofa living room front",
            "wooden dining table top view", 
            "bookshelf home office side"
        ],
        "Architectural Design": [
            "modern house facade front",
            "apartment building street view",
            "office building aerial view"
        ]
    }
    
    for category, prompts in applications.items():
        print(f"\n🏷️  {category}:")
        for prompt in prompts:
            print(f"   📝 {prompt}")
    
    print("\n🎯 To generate these designs:")
    print("   1. python scripts/train_ccgan_final.py --epochs 10 --batch-size 8 --feature-dim 512")
    print("   2. python scripts/demo_ccgan.py --checkpoint checkpoints/ccgan_final_epoch_10.pth")

if __name__ == "__main__":
    design_demo()
