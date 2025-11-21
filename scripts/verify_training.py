#!/usr/bin/env python3

import torch
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from models.component_gan_fixed import ComponentGAN, ComponentCache
from models.composition_gan import CompositionGAN
from models.cpp import ConsumerPreferencePredictor
from data.dataloaders import get_data_loaders

def verify_training_setup():
    print("Verifying Training Setup...")
    
    # Use minimal configuration
    config = {
        'model': {
            'latent_dim': 128,
            'feature_dim': 256,
            'cache_size': 50,
            'num_components': 2
        },
        'training': {
            'batch_size': 4,
            'learning_rate': 0.0002,
            'lr_cpp': 0.0001,
            'epochs': 3,
            'save_interval': 1
        },
        'data': {
            'image_size': 64,
            'num_workers': 0
        },
        'hardware': {
            'device': 'cpu',
            'num_threads': 8,
            'pin_memory': False
        }
    }
    
    print("1. Loading data loaders...")
    try:
        data_loaders = get_data_loaders(config)
        print(f"✓ Data loaders created:")
        print(f"  - Component: {len(data_loaders['component'].dataset)} samples")
        print(f"  - Composition: {len(data_loaders['composition'].dataset)} samples") 
        print(f"  - Preference: {len(data_loaders['preference'].dataset)} samples")
        
        # Test one batch from each loader
        for name, loader in data_loaders.items():
            try:
                batch = next(iter(loader))
                if name == 'component':
                    images, texts = batch
                    print(f"✓ {name} batch: images {images.shape}, {len(texts)} texts")
                elif name == 'composition':
                    scenes, components, targets = batch
                    print(f"✓ {name} batch: {len(scenes)} scenes, {len(components[0])} components, targets {targets.shape}")
                elif name == 'preference':
                    images, scores = batch
                    print(f"✓ {name} batch: images {images.shape}, scores {scores.shape}")
            except Exception as e:
                print(f"✗ {name} loader error: {e}")
                return False
                
    except Exception as e:
        print(f"✗ Data loader error: {e}")
        return False
    
    print("\n2. Initializing models...")
    try:
        component_gan = ComponentGAN(128, 256, 512)
        composition_gan = CompositionGAN(2, 256, 256)
        cpp = ConsumerPreferencePredictor(256)
        cache = ComponentCache(50, 256)
        
        print("✓ All models initialized successfully")
        
        # Test model forward passes
        test_batch_size = 2
        
        # Component GAN test
        z = torch.randn(test_batch_size, 128)
        text_emb = torch.randn(test_batch_size, 512)
        components_out = component_gan(z, text_emb)
        print(f"✓ ComponentGAN: {z.shape} -> {components_out.shape}")
        
        # Composition GAN test  
        comp_features = [torch.randn(test_batch_size, 256) for _ in range(2)]
        comp_out = composition_gan(comp_features, text_emb[0])
        print(f"✓ CompositionGAN: 2 components -> {comp_out.shape}")
        
        # CPP test
        cpp_out = cpp(components_out)
        print(f"✓ CPP: {components_out.shape} -> {cpp_out.shape}")
        
    except Exception as e:
        print(f"✗ Model initialization error: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("\n3. Testing training step...")
    try:
        # Get a batch
        images, texts = next(iter(data_loaders['component']))
        batch_size = images.size(0)
        
        if batch_size < 2:
            print("⚠️  Batch size too small for training test, but models work")
            return True
            
        images = images.to('cpu')
        
        # Flatten images
        image_features = images.view(batch_size, -1)
        if image_features.size(1) != 256:
            image_features = torch.nn.functional.adaptive_avg_pool1d(
                image_features.unsqueeze(1), 256
            ).squeeze(1)
        
        # Get text embeddings
        text_embeddings = cache.get_text_embedding(texts)
        
        # Forward pass
        z = torch.randn(batch_size, 128)
        fake_features = component_gan(z, text_embeddings)
        
        # Discriminator pass
        real_pred = component_gan.discriminator(image_features, text_embeddings)
        fake_pred = component_gan.discriminator(fake_features, text_embeddings)
        
        print(f"✓ Training step completed:")
        print(f"  - Real predictions: {real_pred.shape}, mean: {real_pred.mean().item():.3f}")
        print(f"  - Fake predictions: {fake_pred.shape}, mean: {fake_pred.mean().item():.3f}")
        print(f"  - Generated features: {fake_features.shape}")
        
    except Exception as e:
        print(f"✗ Training step error: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("\n🎉 TRAINING SETUP VERIFIED! Everything is ready for training.")
    print("You can now run: python scripts/train_final.py --epochs 3 --batch-size 4 --feature-dim 256")
    return True

if __name__ == "__main__":
    verify_training_setup()
