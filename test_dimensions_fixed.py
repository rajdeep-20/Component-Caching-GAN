#!/usr/bin/env python3

import torch
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from models.component_gan import ComponentGAN, ComponentCache
from models.composition_gan import CompositionGAN
from models.cpp import ConsumerPreferencePredictor

def test_dimensions():
    print("Testing model dimensions...")
    
    # Test configuration
    latent_dim = 128
    feature_dim = 256
    text_embed_dim = 512
    batch_size = 4
    image_size = 128
    num_components = 3
    
    print(f"Configuration:")
    print(f"  Latent dim: {latent_dim}")
    print(f"  Feature dim: {feature_dim}")
    print(f"  Text embed dim: {text_embed_dim}")
    print(f"  Batch size: {batch_size}")
    print(f"  Image size: {image_size}")
    print(f"  Num components: {num_components}")
    print(f"  Image feature size: {3 * image_size * image_size}")
    
    # Test ComponentGAN
    print("\n1. Testing ComponentGAN...")
    try:
        component_gan = ComponentGAN(latent_dim, feature_dim, text_embed_dim)
        
        # Test forward pass
        z = torch.randn(batch_size, latent_dim)
        text_embed = torch.randn(batch_size, text_embed_dim)
        output = component_gan(z, text_embed)
        print(f"✓ ComponentGAN: input {z.shape} + {text_embed.shape} -> output {output.shape}")
        
        # Test discriminator
        disc_output = component_gan.discriminator(output, text_embed)
        print(f"✓ Discriminator: output {disc_output.shape}")
        
    except Exception as e:
        print(f"✗ ComponentGAN error: {e}")
        return False
    
    # Test CompositionGAN
    print("\n2. Testing CompositionGAN...")
    try:
        composition_gan = CompositionGAN(
            num_components=num_components, 
            output_dim=feature_dim,
            component_feature_dim=feature_dim
        )
        
        # Test forward pass
        component_features = [torch.randn(batch_size, feature_dim) for _ in range(num_components)]
        text_prompt = torch.randn(batch_size, text_embed_dim)
        output = composition_gan(component_features, text_prompt)
        print(f"✓ CompositionGAN: {len(component_features)} components -> output {output.shape}")
        
    except Exception as e:
        print(f"✗ CompositionGAN error: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Test CPP
    print("\n3. Testing ConsumerPreferencePredictor...")
    try:
        cpp = ConsumerPreferencePredictor(input_dim=feature_dim)
        
        # Test forward pass
        features = torch.randn(batch_size, feature_dim)
        output = cpp(features)
        print(f"✓ CPP: input {features.shape} -> output {output.shape}")
        
    except Exception as e:
        print(f"✗ CPP error: {e}")
        return False
    
    # Test full pipeline
    print("\n4. Testing full pipeline...")
    try:
        # Create components
        component_descriptions = ["modern armchair", "wooden table", "floor lamp"]
        cache = ComponentCache(feature_dim=feature_dim)
        
        # Generate components
        generated_components = []
        for desc in component_descriptions:
            component = component_gan.generate_component(desc, cache, num_samples=1)
            cache.store_component(desc, component[0])
            generated_components.append(component[0])
            print(f"  Generated: {desc} - {component.shape}")
        
        # Compose scene
        scene_description = "living room with furniture"
        scene_embedding = cache.get_text_embedding(scene_description)
        
        composition = composition_gan(generated_components, scene_embedding[0])
        print(f"✓ Scene composition: {composition.shape}")
        
        # Test preference prediction
        pref_score = cpp(composition.unsqueeze(0))
        print(f"✓ Preference score: {pref_score.item():.3f}")
        
    except Exception as e:
        print(f"✗ Pipeline error: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("\n✓ All dimension tests passed!")
    return True

if __name__ == "__main__":
    test_dimensions()
