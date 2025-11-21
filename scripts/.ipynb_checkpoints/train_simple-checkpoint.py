import os
import sys
import yaml
import torch
import argparse
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from models.component_gan import ComponentGAN, ComponentCache
from models.composition_gan import CompositionGAN
from models.cpp import ConsumerPreferencePredictor
from training.trainers import CCGANTrainer
from data.dataloaders import get_data_loaders

def main():
    parser = argparse.ArgumentParser(description='Simple CC-GAN Training')
    parser.add_argument('--epochs', type=int, default=5, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=4, help='Batch size')
    args = parser.parse_args()
    
    # Simple configuration
    config = {
        'model': {
            'latent_dim': 128,  # Reduced for faster training
            'feature_dim': 256,  # Reduced for faster training
            'cache_size': 100,
            'num_components': 3
        },
        'training': {
            'batch_size': args.batch_size,
            'learning_rate': 0.0002,
            'lr_cpp': 0.0001,
            'epochs': args.epochs,
            'save_interval': 2
        },
        'data': {
            'image_size': 128,  # Reduced resolution
            'num_workers': 0
        },
        'hardware': {
            'device': 'cpu',
            'num_threads': 8,
            'pin_memory': False
        }
    }
    
    print("Initializing CC-GAN with simplified configuration...")
    
    # Initialize models with smaller dimensions
    component_gan = ComponentGAN(
        latent_dim=config['model']['latent_dim'],
        feature_dim=config['model']['feature_dim']
    )
    
    composition_gan = CompositionGAN(
        num_components=config['model']['num_components'],
        output_dim=config['model']['feature_dim']
    )
    
    cpp = ConsumerPreferencePredictor(
        input_dim=config['model']['feature_dim'],
        hidden_dims=[512, 256]  # Reduced for faster training
    )
    
    cache = ComponentCache(
        cache_size=config['model']['cache_size'],
        feature_dim=config['model']['feature_dim']
    )
    
    # Initialize trainer
    trainer = CCGANTrainer(config, component_gan, composition_gan, cpp, cache)
    
    # Get data loaders
    data_loaders = get_data_loaders(config)
    
    # Create checkpoints directory
    os.makedirs('checkpoints', exist_ok=True)
    
    print(f"Starting training for {args.epochs} epochs...")
    
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        
        try:
            # Train Component GAN
            d_loss, g_loss = trainer.train_component_gan(data_loaders['component'], epoch + 1)
            
            # Train Composition GAN (skip first epoch to let components train)
            if epoch > 0:
                comp_loss = trainer.train_composition_gan(data_loaders['composition'], epoch + 1)
            else:
                comp_loss = 0.0
                print("Skipping composition training in first epoch")
            
            # Train CPP
            cpp_loss = trainer.train_cpp(data_loaders['preference'], epoch + 1)
            
            # Save checkpoint
            if (epoch + 1) % config['training']['save_interval'] == 0:
                checkpoint_path = f"checkpoints/ccgan_simple_epoch_{epoch+1}.pth"
                trainer.save_checkpoint(epoch + 1, checkpoint_path)
            
            print(f"Epoch {epoch+1} Results:")
            print(f"  Component GAN - D: {d_loss:.4f}, G: {g_loss:.4f}")
            print(f"  Composition GAN - Loss: {comp_loss:.4f}")
            print(f"  CPP - Loss: {cpp_loss:.4f}")
            print(f"  Cache size: {len(cache.cache)}")
            
        except Exception as e:
            print(f"Error in epoch {epoch+1}: {e}")
            break
    
    print("Training completed!")
    
    # Save final model
    final_checkpoint = "checkpoints/ccgan_simple_final.pth"
    trainer.save_checkpoint(args.epochs, final_checkpoint)
    print(f"Final model saved: {final_checkpoint}")

if __name__ == "__main__":
    main()