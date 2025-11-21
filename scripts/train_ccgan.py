#!/usr/bin/env python3

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

def load_config(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def main():
    parser = argparse.ArgumentParser(description='Train CC-GAN')
    parser.add_argument('--config', type=str, default='configs/base_config.yaml', 
                       help='Path to config file')
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to checkpoint to resume from')
    parser.add_argument('--epochs', type=int, default=None,
                       help='Override number of epochs from config')
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Override epochs if specified
    if args.epochs:
        config['training']['epochs'] = args.epochs
    
    # Set device
    device = config['hardware']['device']
    torch.set_num_threads(config['hardware']['num_threads'])
    
    print(f"Using device: {device}")
    print(f"Number of threads: {config['hardware']['num_threads']}")
    
    # Initialize models
    component_gan = ComponentGAN(
        latent_dim=config['model']['latent_dim'],
        feature_dim=config['model']['feature_dim']
    )
    
    composition_gan = CompositionGAN(
        num_components=config['model']['num_components'],
        output_dim=config['model']['feature_dim']
    )
    
    cpp = ConsumerPreferencePredictor(
        input_dim=config['model']['feature_dim']
    )
    
    cache = ComponentCache(
        cache_size=config['model']['cache_size'],
        feature_dim=config['model']['feature_dim']
    )
    
    # Initialize trainer
    trainer = CCGANTrainer(config, component_gan, composition_gan, cpp, cache)
    
    # Resume from checkpoint if specified
    start_epoch = 0
    if args.resume and os.path.exists(args.resume):
        start_epoch = trainer.load_checkpoint(args.resume)
        print(f"Resumed from epoch {start_epoch}")
    
    # Get data loaders
    data_loaders = get_data_loaders(config)
    
    # Create checkpoints directory
    os.makedirs('checkpoints', exist_ok=True)
    
    # Training loop
    total_epochs = config['training']['epochs']
    save_interval = config['training']['save_interval']
    
    print(f"Starting training for {total_epochs} epochs...")
    
    for epoch in range(start_epoch, total_epochs):
        print(f"\n{'='*50}")
        print(f"Epoch {epoch+1}/{total_epochs}")
        print(f"{'='*50}")
        
        # Train Component GAN
        d_loss, g_loss = trainer.train_component_gan(data_loaders['component'], epoch + 1)
        
        # Train Composition GAN
        comp_loss = trainer.train_composition_gan(data_loaders['composition'], epoch + 1)
        
        # Train CPP
        cpp_loss = trainer.train_cpp(data_loaders['preference'], epoch + 1)
        
        # Save checkpoint
        if (epoch + 1) % save_interval == 0:
            checkpoint_path = f"checkpoints/ccgan_epoch_{epoch+1}.pth"
            trainer.save_checkpoint(epoch + 1, checkpoint_path)
        
        print(f"Epoch {epoch+1} Summary:")
        print(f"  Component GAN - D Loss: {d_loss:.4f}, G Loss: {g_loss:.4f}")
        print(f"  Composition GAN - Loss: {comp_loss:.4f}")
        print(f"  CPP - Loss: {cpp_loss:.4f}")
        print(f"  Cache size: {len(cache.cache)}")
    
    print("Training completed!")
    
    # Save final model
    final_checkpoint = "checkpoints/ccgan_final.pth"
    trainer.save_checkpoint(total_epochs, final_checkpoint)
    print(f"Final model saved: {final_checkpoint}")

if __name__ == "__main__":
    main()
