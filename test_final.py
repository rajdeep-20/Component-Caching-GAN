#!/usr/bin/env python3

import torch
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from models.component_gan_fixed import ComponentGAN, ComponentCache
from models.composition_gan import CompositionGAN
from models.cpp import ConsumerPreferencePredictor

def test_final_pipeline():
    print("Testing Final Pipeline...")
    
    # Test configuration
    latent_dim = 128
    feature_dim = 256
    text_embed_dim = 512
    batch_size = 4
    num_components = 3
    
    print(f"Configuration:")
    print(f"  Latent dim: {latent_dim}")
    print(f"  Feature dim: {feature_dim}")
    print(f"  Batch size: {batch_size}")
    print(f"  Num components: {num_components}")
    
    # Test ComponentGAN
    print("\n1. Testing ComponentGAN...")
    try:
        component_gan = ComponentGAN(latent_dim, feature_dim, text_embed_dim)
        cache = ComponentCache(feature_dim=feature_dim)
        
        # Test single sample generation
        component_single = component_gan.generate_component("test component", cache, num_samples=1)
        print(f"✓ Single sample: {component_single.shape}")
        
        # Test multiple sample generation
        component_multi = component_gan.generate_component("test component", cache, num_samples=3)
        print(f"✓ Multiple samples: {component_multi.shape}")
        
        # Test batch generation
        z_batch = torch.randn(batch_size, latent_dim)
        text_batch = torch.randn(batch_size, text_embed_dim)
        output_batch = component_gan(z_batch, text_batch)
        print(f"✓ Batch generation: {output_batch.shape}")
        
    except Exception as e:
        print(f"✗ ComponentGAN error: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Test CompositionGAN
    print("\n2. Testing CompositionGAN...")
    try:
        composition_gan = CompositionGAN(
            num_components=num_components, 
            output_dim=feature_dim,
            component_feature_dim=feature_dim
        )
        
        # Test forward pass with batch
        component_features = [torch.randn(batch_size, feature_dim) for _ in range(num_components)]
        text_prompt = torch.randn(batch_size, text_embed_dim)
        output = composition_gan(component_features, text_prompt)
        print(f"✓ CompositionGAN batch: {len(component_features)} components -> {output.shape}")
        
        # Test forward pass with single sample
        component_single = [torch.randn(1, feature_dim) for _ in range(num_components)]
        text_single = torch.randn(1, text_embed_dim)
        output_single = composition_gan(component_single, text_single)
        print(f"✓ CompositionGAN single: {len(component_single)} components -> {output_single.shape}")
        
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
        print(f"✓ CPP batch: {features.shape} -> {output.shape}")
        
        # Test single sample
        feature_single = torch.randn(1, feature_dim)
        output_single = cpp(feature_single)
        print(f"✓ CPP single: {feature_single.shape} -> {output_single.shape}")
        
    except Exception as e:
        print(f"✗ CPP error: {e}")
        return False
    
    # Test full pipeline
    print("\n4. Testing full pipeline...")
    try:
        # Create multiple components
        component_descriptions = ["modern armchair", "wooden table", "floor lamp"]
        
        # Generate components (one for each description)
        generated_components = []
        for desc in component_descriptions:
            component = component_gan.generate_component(desc, cache, num_samples=1)
            cache.store_component(desc, component[0])
            generated_components.append(component[0])
            print(f"  Generated: {desc} - {component.shape}")
        
        # Compose scene with single sample
        scene_description = "living room with furniture"
        scene_embedding = cache.get_text_embedding(scene_description)
        
        # Stack components for single sample processing
        component_single = [comp.unsqueeze(0) for comp in generated_components]
        composition = composition_gan(component_single, scene_embedding[0])
        print(f"✓ Single scene composition: {composition.shape}")
        
        # Test preference prediction
        pref_score = cpp(composition)
        print(f"✓ Preference score: {pref_score.item():.3f}")
        
        # Test batch composition
        print("\n5. Testing batch composition...")
        batch_components = []
        for i in range(batch_size):
            # Create different component combinations for each batch item
            batch_component_set = []
            for j, desc in enumerate(component_descriptions):
                # Add some variation to components
                base_component = cache.retrieve_component(desc)
                if base_component is not None:
                    # Add small noise for variation
                    varied_component = base_component + torch.randn_like(base_component) * 0.1
                    batch_component_set.append(varied_component.unsqueeze(0))
                else:
                    # Generate new component if not in cache
                    new_component = component_gan.generate_component(desc, cache, num_samples=1)
                    batch_component_set.append(new_component)
            
            # Stack components for this batch item
            batch_component_tensor = torch.cat(batch_component_set, dim=0)
            batch_components.append(batch_component_tensor)
        
        # Stack all batch items
        batch_component_tensor = torch.stack(batch_components)
        
        # Create batch text embeddings
        batch_text = scene_embedding.repeat(batch_size, 1)
        
        # Process each batch item through composition GAN
        batch_compositions = []
        for i in range(batch_size):
            comp_features = [batch_component_tensor[i, j] for j in range(num_components)]
            composition = composition_gan(comp_features, batch_text[i])
            batch_compositions.append(composition)
        
        batch_output = torch.stack(batch_compositions)
        print(f"✓ Batch composition: {batch_output.shape}")
        
        # Test batch preference prediction
        batch_pref = cpp(batch_output)
        print(f"✓ Batch preference scores: {batch_pref.shape}")
        
    except Exception as e:
        print(f"✗ Pipeline error: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("\n🎉 ALL TESTS PASSED! The complete CC-GAN pipeline is working correctly.")
    print("You can now run the training script with confidence.")
    return True

if __name__ == "__main__":
    test_final_pipeline()
