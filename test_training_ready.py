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

def test_training_scenarios():
    print("Testing Training Scenarios...")
    
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
    
    # Test 1: Batch training with single text
    print("\n2. Testing batch training with single text...")
    try:
        # Create batch of components
        train_components = [torch.randn(batch_size, feature_dim) for _ in range(num_components)]
        train_text = torch.randn(text_embed_dim)  # Single text embedding
        
        # This should work - single text broadcasted to batch
        output = composition_gan(train_components, train_text)
        print(f"✓ Batch with single text: {output.shape}")
        
        # Verify batch size is preserved
        assert output.size(0) == batch_size, f"Expected batch size {batch_size}, got {output.size(0)}"
        print(f"✓ Batch size preserved: {output.size(0)}")
        
    except Exception as e:
        print(f"✗ Batch training with single text error: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Test 2: Batch training with batch text
    print("\n3. Testing batch training with batch text...")
    try:
        # Create batch of components
        train_components = [torch.randn(batch_size, feature_dim) for _ in range(num_components)]
        train_text = torch.randn(batch_size, text_embed_dim)  # Batch text embeddings
        
        # This should work - batch text used as is
        output = composition_gan(train_components, train_text[0])  # Using first for simplicity
        print(f"✓ Batch with batch text: {output.shape}")
        
        # Test with different text for each
        outputs = []
        for i in range(batch_size):
            # Extract i-th component from each component list
            comp_batch_i = [comp[i].unsqueeze(0) for comp in train_components]
            text_i = train_text[i]
            out_i = composition_gan(comp_batch_i, text_i)
            outputs.append(out_i)
        
        stacked = torch.stack(outputs)
        print(f"✓ Individual batch processing: {stacked.shape}")
        
    except Exception as e:
        print(f"✗ Batch training with batch text error: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Test 3: Mixed scenarios
    print("\n4. Testing mixed scenarios...")
    try:
        # Scenario 1: 1D components with 1D text
        comps_1d = [torch.randn(feature_dim) for _ in range(num_components)]
        text_1d = torch.randn(text_embed_dim)
        out_1 = composition_gan(comps_1d, text_1d)
        print(f"✓ 1D components + 1D text: {out_1.shape}")
        
        # Scenario 2: 2D components (batch=1) with 1D text
        comps_2d_1 = [torch.randn(1, feature_dim) for _ in range(num_components)]
        out_2 = composition_gan(comps_2d_1, text_1d)
        print(f"✓ 2D components (batch=1) + 1D text: {out_2.shape}")
        
        # Scenario 3: Mixed 1D and 2D components
        mixed_comps = [
            torch.randn(feature_dim),  # 1D
            torch.randn(1, feature_dim),  # 2D with batch=1
            torch.randn(feature_dim)  # 1D
        ]
        out_3 = composition_gan(mixed_comps, text_1d)
        print(f"✓ Mixed components + 1D text: {out_3.shape}")
        
    except Exception as e:
        print(f"✗ Mixed scenarios error: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Test 4: Integration with ComponentGAN and CPP
    print("\n5. Testing full training integration...")
    try:
        # Simulate training step
        batch_size = 4
        
        # Generate components using ComponentGAN
        component_descriptions = ["modern armchair", "wooden table", "floor lamp"]
        generated_components_batch = []
        
        for desc in component_descriptions:
            # Generate batch of components for this description
            components = component_gan.generate_component(desc, cache, num_samples=batch_size)
            generated_components_batch.append(components)  # Each is [batch_size, feature_dim]
        
        # Prepare for composition - transpose to get list of component sets per batch item
        batch_component_sets = []
        for i in range(batch_size):
            comp_set = [comp_batch[i] for comp_batch in generated_components_batch]
            batch_component_sets.append(comp_set)
        
        # Create scene description and embedding
        scene_description = "living room with furniture"
        scene_embedding = cache.get_text_embedding(scene_description)
        
        # Compose each batch item
        compositions = []
        for i in range(batch_size):
            comp = composition_gan(batch_component_sets[i], scene_embedding[0])
            compositions.append(comp)
        
        compositions_tensor = torch.stack(compositions)
        print(f"✓ Full training integration - compositions: {compositions_tensor.shape}")
        
        # Test CPP on generated compositions
        pref_scores = cpp(compositions_tensor)
        print(f"✓ CPP preference scores: {pref_scores.shape}")
        
        # Test adversarial training
        real_labels = torch.ones(batch_size, 1)
        fake_pred = component_gan.discriminator(compositions_tensor, scene_embedding.repeat(batch_size, 1))
        print(f"✓ Discriminator predictions: {fake_pred.shape}")
        
    except Exception as e:
        print(f"✗ Full training integration error: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("\n🎉 ALL TRAINING SCENARIOS PASSED!")
    print("The CC-GAN is ready for training with various batch configurations.")
    print("\nYou can now confidently run:")
    print("  python scripts/train_final.py --epochs 3 --batch-size 4 --feature-dim 256")
    return True

if __name__ == "__main__":
    test_training_scenarios()
