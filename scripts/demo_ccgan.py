#!/usr/bin/env python3

import torch
import yaml
import argparse
from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from models.component_gan import ComponentGAN, ComponentCache
from models.composition_gan import CompositionGAN
from models.call_mechanism import CoupledAttentionLocalization
from models.cpp import ConsumerPreferencePredictor, ContinuousConditionalGAN

def main():
    parser = argparse.ArgumentParser(description='CC-GAN Demo')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--config', type=str, default='configs/base_config.yaml',
                       help='Path to config file')
    args = parser.parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Initialize models
    component_gan = ComponentGAN()
    composition_gan = CompositionGAN()
    cpp = ConsumerPreferencePredictor()
    cache = ComponentCache()
    
    # Load checkpoint
    device = config['hardware']['device']
    checkpoint = torch.load(args.checkpoint, map_location=device)
    
    component_gan.load_state_dict(checkpoint['component_gan_state_dict'])
    composition_gan.load_state_dict(checkpoint['composition_gan_state_dict'])
    cpp.load_state_dict(checkpoint['cpp_state_dict'])
    cache.cache = checkpoint['cache']
    
    # Move to device
    component_gan.to(device)
    composition_gan.to(device)
    cpp.to(device)
    
    # Set to eval mode
    component_gan.eval()
    composition_gan.eval()
    cpp.eval()
    
    # Initialize CALL
    call_mechanism = CoupledAttentionLocalization(composition_gan)
    
    # Demo scenarios
    print("CC-GAN Demo")
    print("=" * 50)
    
    # Demo 1: Component generation and caching
    print("\n1. Component Generation and Caching")
    print("-" * 40)
    
    test_components = ["modern armchair", "wooden table", "brass floor lamp"]
    for comp_desc in test_components:
        feature = component_gan.generate_component(comp_desc, cache)
        cache.store_component(comp_desc, feature[0])
        print(f"✓ Generated and cached: {comp_desc}")
    
    # Demo 2: Scene composition
    print("\n2. Scene Composition")
    print("-" * 40)
    
    scene_description = "living room with modern armchair and brass floor lamp"
    scene_embedding = cache.get_text_embedding(scene_description)
    
    component_features = []
    for comp_desc in ["modern armchair", "brass floor lamp"]:
        feature = cache.retrieve_component(comp_desc)
        if feature is not None:
            component_features.append(feature.to(device))
            print(f"✓ Retrieved component: {comp_desc}")
    
    if len(component_features) >= 2:
        composition = composition_gan(component_features, scene_embedding)
        print(f"✓ Composed scene: {scene_description}")
        
        # Get preference score
        pref_score = cpp(composition.unsqueeze(0))
        print(f"✓ Predicted preference score: {pref_score.item():.3f}")
    
    # Demo 3: Viewpoint control with CALL
    print("\n3. Viewpoint Control with CALL")
    print("-" * 40)
    
    viewpoints = ["front view", "side view", "top view"]
    for viewpoint in viewpoints:
        composition_with_view = call_mechanism(
            component_features, scene_embedding, viewpoint
        )
        print(f"✓ Generated {viewpoint} composition")
    
    # Demo 4: Preference-guided generation
    print("\n4. Preference-Guided Generation")
    print("-" * 40)
    
    cc_gan = ContinuousConditionalGAN(composition_gan, cpp)
    best_composition, best_score = cc_gan.generate_with_preference_guidance(
        component_features, scene_embedding, min_preference=0.7
    )
    
    print(f"✓ Best composition preference score: {best_score.item():.3f}")
    
    print("\n" + "=" * 50)
    print("Demo completed successfully!")
    print(f"Total components in cache: {len(cache.cache)}")

if __name__ == "__main__":
    main()
