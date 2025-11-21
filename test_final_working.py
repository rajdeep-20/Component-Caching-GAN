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

def test_final_working():
    print("Testing Final Working Implementation...")
    
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
    
    # Test 1: Basic functionality
    print("\n2. Testing basic functionality...")
    try:
        # Single sample generation
        component = component_gan.generate_component("test", cache, num_samples=1)
        print(f"✓ Single component: {component.shape}")
        
        # Multiple samples
        components = component_gan.generate_component("test", cache, num_samples=3)
        print(f"✓ Multiple components: {components.shape}")
        
        # Batch generation
        z = torch.randn(batch_size, latent_dim)
        text = torch.randn(batch_size, text_embed_dim)
        batch_components = component_gan(z, text)
        print(f"✓ Batch components: {batch_components.shape}")
        
    except Exception as e:
        print(f"✗ Basic functionality error: {e}")
        return False
    
    # Test 2: Composition GAN with proper 2D tensors
    print("\n3. Testing Composition GAN...")
    try:
        # Create 2D component features
        comp_features = [torch.randn(batch_size, feature_dim) for _ in range(num_components)]
        text_prompt = torch.randn(text_embed_dim)  # 1D text
        
        output = composition_gan(comp_features, text_prompt)
        print(f"✓ Composition GAN 2D: {output.shape}")
        
        # Ensure output is 2D
        assert output.dim() == 2, f"Expected 2D output, got {output.dim()}D"
        assert output.size(0) == batch_size, f"Expected batch size {batch_size}, got {output.size(0)}"
        assert output.size(1) == feature_dim, f"Expected feature dim {feature_dim}, got {output.size(1)}"
        print("✓ Output is proper 2D tensor")
        
    except Exception as e:
        print(f"✗ Composition GAN error: {e}")
        return False
    
    # Test 3: Full pipeline with proper dimensions
    print("\n4. Testing full pipeline...")
    try:
        # Generate components
        descriptions = ["chair", "table", "lamp"]
        generated_comps = []
        
        for desc in descriptions:
            comp = component_gan.generate_component(desc, cache, num_samples=1)
            cache.store_component(desc, comp[0])
            generated_comps.append(comp[0])  # 1D tensor
        
        # Create batch by repeating components
        batch_comp_sets = []
        for i in range(batch_size):
            # Each batch item gets the same components but we'll add some variation
            varied_comps = []
            for comp in generated_comps:
                # Add small noise for variation
                varied = comp + torch.randn_like(comp) * 0.01
                varied_comps.append(varied)
            batch_comp_sets.append(varied_comps)
        
        # Compose scenes
        scene_text = torch.randn(text_embed_dim)
        compositions = []
        
        for i in range(batch_size):
            comp = composition_gan(batch_comp_sets[i], scene_text)
            # Ensure composition is 1D for single sample
            if comp.dim() == 2 and comp.size(0) == 1:
                comp = comp.squeeze(0)
            compositions.append(comp)
        
        # Stack to create proper 2D batch tensor
        compositions_batch = torch.stack(compositions)
        print(f"✓ Compositions batch: {compositions_batch.shape}")
        
        # Test CPP
        pref_scores = cpp(compositions_batch)
        print(f"✓ Preference scores: {pref_scores.shape}")
        
        # Test discriminator (requires 2D input)
        disc_pred = component_gan.discriminator(compositions_batch, scene_text.unsqueeze(0).repeat(batch_size, 1))
        print(f"✓ Discriminator predictions: {disc_pred.shape}")
        
    except Exception as e:
        print(f"✗ Full pipeline error: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Test 5: Training scenario
    print("\n5. Testing training scenario...")
    try:
        # Simulate training data
        real_images = torch.randn(batch_size, 3, 64, 64)  # Simulated image batch
        real_features = real_images.view(batch_size, -1)
        
        # Adaptive pooling to match feature_dim
        if real_features.size(1) != feature_dim:
            real_features = torch.nn.functional.adaptive_avg_pool1d(
                real_features.unsqueeze(1), feature_dim
            ).squeeze(1)
        
        text_descriptions = ["scene " + str(i) for i in range(batch_size)]
        text_embeddings = cache.get_text_embedding(text_descriptions)
        
        # Component GAN training step
        z = torch.randn(batch_size, latent_dim)
        fake_features = component_gan(z, text_embeddings)
        
        # Discriminator forward
        real_pred = component_gan.discriminator(real_features, text_embeddings)
        fake_pred = component_gan.discriminator(fake_features, text_embeddings)
        
        print(f"✓ Training scenario:")
        print(f"  - Real features: {real_features.shape}")
        print(f"  - Fake features: {fake_features.shape}")
        print(f"  - Real pred: {real_pred.shape}")
        print(f"  - Fake pred: {fake_pred.shape}")
        
        # Composition training
        comp_features_train = [torch.randn(batch_size, feature_dim) for _ in range(num_components)]
        target_features = torch.randn(batch_size, feature_dim)
        comp_output = composition_gan(comp_features_train, text_embeddings[0])
        
        print(f"  - Composition output: {comp_output.shape}")
        print(f"  - Target features: {target_features.shape}")
        
    except Exception as e:
        print(f"✗ Training scenario error: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("\n🎉 ALL TESTS PASSED! CC-GAN is ready for training.")
    print("\nYou can now run the training script:")
    print("  python scripts/train_final.py --epochs 3 --batch-size 4 --feature-dim 256")
    return True

if __name__ == "__main__":
    test_final_working()
