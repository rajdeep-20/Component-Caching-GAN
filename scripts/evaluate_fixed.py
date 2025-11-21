#!/usr/bin/env python3

import os
import sys
import torch
import argparse
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from models.component_gan_fixed import ComponentGAN, ComponentCache
from models.composition_gan import CompositionGAN
from models.cpp import ConsumerPreferencePredictor
from evaluation.metrics import CCGANEvaluator

def evaluate_fixed(checkpoint_path):
    print(f"Loading checkpoint: {checkpoint_path}")
    
    # Load checkpoint first to get the configuration
    device = 'cpu'
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint.get('config', {})
    
    print("Checkpoint configuration:")
    print(f"  Model: {config.get('model', {})}")
    
    # Use dimensions from checkpoint or defaults
    model_config = config.get('model', {})
    latent_dim = model_config.get('latent_dim', 128)
    feature_dim = model_config.get('feature_dim', 256)
    num_components = model_config.get('num_components', 3)
    
    print(f"Using dimensions from checkpoint:")
    print(f"  latent_dim: {latent_dim}")
    print(f"  feature_dim: {feature_dim}")
    print(f"  num_components: {num_components}")
    
    # Initialize models with correct dimensions
    component_gan = ComponentGAN(
        latent_dim=latent_dim,
        feature_dim=feature_dim,
        text_embed_dim=512
    )
    
    composition_gan = CompositionGAN(
        num_components=num_components,
        output_dim=feature_dim,
        component_feature_dim=feature_dim
    )
    
    cpp = ConsumerPreferencePredictor(
        input_dim=feature_dim
    )
    
    cache = ComponentCache(
        cache_size=model_config.get('cache_size', 100),
        feature_dim=feature_dim
    )
    
    # Load state dicts
    print("Loading model weights...")
    component_gan.load_state_dict(checkpoint['component_gan_state_dict'])
    composition_gan.load_state_dict(checkpoint['composition_gan_state_dict'])
    cpp.load_state_dict(checkpoint['cpp_state_dict'])
    cache.cache = checkpoint['cache']
    
    # Move to device
    component_gan.to(device)
    composition_gan.to(device)
    cpp.to(device)
    
    # Set to evaluation mode
    component_gan.eval()
    composition_gan.eval()
    cpp.eval()
    
    # Initialize evaluator
    evaluator = CCGANEvaluator(device=device)
    
    print("\n Model loaded successfully!")
    print(f" Component GAN parameters: {sum(p.numel() for p in component_gan.parameters()):,}")
    print(f" Composition GAN parameters: {sum(p.numel() for p in composition_gan.parameters()):,}")
    print(f" CPP parameters: {sum(p.numel() for p in cpp.parameters()):,}")
    print(f" Cache size: {len(cache.cache)}")
    
    # Run basic functionality test
    print("\n Running basic functionality test...")
    try:
        # Test component generation
        test_component = component_gan.generate_component("test chair", cache, num_samples=1)
        print(f" Component generation: {test_component.shape}")
        
        # Test composition
        test_components = [test_component[0] for _ in range(3)]  # Repeat for 3 components
        test_text = cache.get_text_embedding("test scene")[0]
        test_composition = composition_gan(test_components, test_text)
        print(f" Composition: {test_composition.shape}")
        
        # Test CPP
        test_pref = cpp(test_composition.unsqueeze(0) if test_composition.dim() == 1 else test_composition)
        print(f" Preference prediction: {test_pref.item():.3f}")
        
        print("\n Evaluation setup successful! The model is working correctly.")
        
        # Show some cache contents
        print(f"\n Cache contents ({len(cache.cache)} items):")
        cache_items = list(cache.cache.keys())[:5]  # Show first 5
        for i, key in enumerate(cache_items):
            print(f"  {i+1}. {cache.retrieve_component('any').shape if cache.cache else 'Empty'}")
            
    except Exception as e:
        print(f" Functionality test failed: {e}")
        return False
    
    return True

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate CC-GAN (Fixed)')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.checkpoint):
        print(f"Error: Checkpoint file {args.checkpoint} does not exist!")
        sys.exit(1)
    
    success = evaluate_fixed(args.checkpoint)
    if success:
        print("\n You can now use this model for:")
        print("   python scripts/demo_ccgan.py --checkpoint your_checkpoint.pth")
        print("   python scripts/train_ccgan_final.py --resume your_checkpoint.pth")
    else:
        print("\n Recommendation: Train a new model with current architecture")
        print("   python scripts/train_ccgan_final.py --epochs 5 --batch-size 4 --feature-dim 256")
