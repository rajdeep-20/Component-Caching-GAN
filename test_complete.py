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

def test_complete_pipeline():
    print("Testing Complete CC-GAN Pipeline...")
    
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
    
    # Initialize models
    print("\n1. Initializing models...")
    try:
        component_gan = ComponentGAN(latent_dim, feature_dim, text_embed_dim)
        composition_gan = CompositionGAN(
            num_components=num_components, 
            output_dim=feature_dim,
            component_feature_dim=feature_dim
        )
        cpp = ConsumerPreferencePredictor(input_dim=feature_dim)
        cache = ComponentCache(feature_dim=feature_dim)
        
        print("✓ All models initialized successfully")
    except Exception as e:
        print(f"✗ Model initialization error: {e}")
        return False
    
    # Test single sample pipeline
    print("\n2. Testing single sample pipeline...")
    try:
        # Create components
        component_descriptions = ["modern armchair", "wooden table", "floor lamp"]
        
        # Generate components
        generated_components = []
        for desc in component_descriptions:
            component = component_gan.generate_component(desc, cache, num_samples=1)
            cache.store_component(desc, component[0])
            generated_components.append(component[0])  # This is a 1D tensor
            print(f"  Generated: {desc} - shape: {component[0].shape}")
        
        # Compose scene
        scene_description = "living room with furniture"
        scene_embedding = cache.get_text_embedding(scene_description)
        
        composition = composition_gan(generated_components, scene_embedding[0])
        print(f"✓ Scene composition: {composition.shape}")
        
        # Test preference prediction
        pref_score = cpp(composition.unsqueeze(0) if composition.dim() == 1 else composition)
        print(f"✓ Preference score: {pref_score.item():.3f}")
        
    except Exception as e:
        print(f"✗ Single sample pipeline error: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Test batch pipeline
    print("\n3. Testing batch pipeline...")
    try:
        # Create batch of components
        batch_component_sets = []
        for i in range(batch_size):
            component_set = []
            for desc in component_descriptions:
                # Retrieve from cache or generate new
                cached = cache.retrieve_component(desc)
                if cached is not None:
                    component_set.append(cached)
                else:
                    new_comp = component_gan.generate_component(desc, cache, num_samples=1)[0]
                    component_set.append(new_comp)
            batch_component_sets.append(component_set)
        
        # Create batch text embeddings
        scene_description = "living room with furniture" 
        scene_embedding = cache.get_text_embedding(scene_description)
        batch_text = scene_embedding.repeat(batch_size, 1)
        
        # Process batch through composition GAN
        batch_compositions = []
        for i in range(batch_size):
            comp_features = batch_component_sets[i]  # List of 1D tensors
            composition = composition_gan(comp_features, batch_text[i])
            batch_compositions.append(composition)
        
        # Stack results
        batch_output = torch.stack(batch_compositions)
        print(f"✓ Batch composition: {batch_output.shape}")
        
        # Test batch preference prediction
        batch_pref = cpp(batch_output)
        print(f"✓ Batch preference scores: {batch_pref.shape}")
        
    except Exception as e:
        print(f"✗ Batch pipeline error: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Test mixed dimensions
    print("\n4. Testing mixed dimension handling...")
    try:
        # Mix of 1D and 2D tensors
        mixed_components = [
            torch.randn(feature_dim),  # 1D
            torch.randn(1, feature_dim),  # 2D with batch=1
            torch.randn(feature_dim)  # 1D
        ]
        
        text_prompt = torch.randn(text_embed_dim)  # 1D
        
        output = composition_gan(mixed_components, text_prompt)
        print(f"✓ Mixed dimension composition: {output.shape}")
        
    except Exception as e:
        print(f"✗ Mixed dimension error: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Test training compatibility
    print("\n5. Testing training compatibility...")
    try:
        # Simulate training batch
        train_components = [torch.randn(batch_size, feature_dim) for _ in range(num_components)]
        train_text = torch.randn(batch_size, text_embed_dim)
        
        train_output = composition_gan(train_components, train_text[0])  # Using first text for all
        print(f"✓ Training batch composition: {train_output.shape}")
        
        # Test with different text for each sample
        train_outputs = []
        for i in range(batch_size):
            output = composition_gan(
                [comp[i].unsqueeze(0) for comp in train_components],  # 2D with batch=1
                train_text[i]  # 1D
            )
            train_outputs.append(output)
        
        stacked_output = torch.stack(train_outputs)
        print(f"✓ Individual text composition: {stacked_output.shape}")
        
    except Exception as e:
        print(f"✗ Training compatibility error: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("\n🎉 COMPLETE PIPELINE TEST PASSED!")
    print("The CC-GAN implementation is fully functional and ready for training.")
    print("\nYou can now run:")
    print("  python scripts/train_final.py --epochs 3 --batch-size 4 --feature-dim 256")
    return True

if __name__ == "__main__":
    test_complete_pipeline()
