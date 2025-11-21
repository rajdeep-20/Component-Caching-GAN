#!/usr/bin/env python3

import os
import sys
import yaml
import torch
import argparse
from pathlib import Path
import logging

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from models.component_gan import ComponentGAN, ComponentCache
from models.composition_gan import CompositionGAN
from models.cpp import ConsumerPreferencePredictor
from data.dataloaders import get_data_loaders

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class FixedCCGANTrainer:
    def __init__(self, config):
        self.config = config
        self.device = torch.device(config['hardware']['device'])
        
        # Initialize models with consistent dimensions
        latent_dim = config['model']['latent_dim']
        feature_dim = config['model']['feature_dim']
        text_embed_dim = 512  # CLIP embedding dimension
        
        self.component_gan = ComponentGAN(
            latent_dim=latent_dim,
            feature_dim=feature_dim,
            text_embed_dim=text_embed_dim
        )
        
        self.composition_gan = CompositionGAN(
            num_components=config['model']['num_components'],
            output_dim=feature_dim
        )
        
        self.cpp = ConsumerPreferencePredictor(
            input_dim=feature_dim,
            hidden_dims=[512, 256]
        )
        
        self.cache = ComponentCache(
            cache_size=config['model']['cache_size'],
            feature_dim=feature_dim
        )
        
        # Optimizers
        self.optimizer_g = torch.optim.Adam(
            list(self.component_gan.parameters()) + 
            list(self.composition_gan.parameters()),
            lr=config['training']['learning_rate'], 
            betas=(0.5, 0.999)
        )
        
        self.optimizer_d = torch.optim.Adam(
            self.component_gan.discriminator.parameters(),
            lr=config['training']['learning_rate'], 
            betas=(0.5, 0.999)
        )
        
        self.optimizer_cpp = torch.optim.Adam(
            self.cpp.parameters(), 
            lr=config['training']['lr_cpp']
        )
        
        # Loss functions
        self.adversarial_loss = torch.nn.BCELoss()
        self.feature_loss = torch.nn.MSELoss()
        
        # Move models to device
        self.component_gan.to(self.device)
        self.composition_gan.to(self.device)
        self.cpp.to(self.device)
        
    def train_component_gan(self, dataloader, epoch):
        self.component_gan.train()
        total_d_loss = 0
        total_g_loss = 0
        batch_count = 0
        
        from tqdm import tqdm
        pbar = tqdm(dataloader, desc=f"Component GAN Epoch {epoch}")
        
        for batch_idx, (real_images, text_descriptions) in enumerate(pbar):
            try:
                batch_size = real_images.size(0)
                real_images = real_images.to(self.device)
                
                # Get text embeddings - ensure proper shape
                text_embeddings = self.cache.get_text_embedding(text_descriptions).to(self.device)
                
                # Flatten images to match feature dimension
                real_features = real_images.view(batch_size, -1)
                
                # Ensure feature dimension matches
                if real_features.size(1) != self.config['model']['feature_dim']:
                    # Use adaptive pooling or projection to match dimensions
                    real_features = torch.nn.functional.adaptive_avg_pool1d(
                        real_features.unsqueeze(1), 
                        self.config['model']['feature_dim']
                    ).squeeze(1)
                
                # Create labels
                real_labels = torch.ones(batch_size, 1).to(self.device)
                fake_labels = torch.zeros(batch_size, 1).to(self.device)
                
                # Train Discriminator
                self.optimizer_d.zero_grad()
                
                # Real images
                real_pred = self.component_gan.discriminator(real_features, text_embeddings)
                real_loss = self.adversarial_loss(real_pred, real_labels)
                
                # Fake images
                z = torch.randn(batch_size, self.config['model']['latent_dim']).to(self.device)
                fake_features = self.component_gan(z, text_embeddings)
                fake_pred = self.component_gan.discriminator(fake_features.detach(), text_embeddings)
                fake_loss = self.adversarial_loss(fake_pred, fake_labels)
                
                d_loss = (real_loss + fake_loss) / 2
                d_loss.backward()
                self.optimizer_d.step()
                
                # Train Generator
                self.optimizer_g.zero_grad()
                fake_pred = self.component_gan.discriminator(fake_features, text_embeddings)
                g_loss = self.adversarial_loss(fake_pred, real_labels)
                g_loss.backward()
                self.optimizer_g.step()
                
                total_d_loss += d_loss.item()
                total_g_loss += g_loss.item()
                batch_count += 1
                
                # Cache successful components
                if g_loss.item() < 0.5:  # Threshold for successful generation
                    for i, desc in enumerate(text_descriptions):
                        self.cache.store_component(desc, fake_features[i].detach().cpu())
                
                pbar.set_postfix({
                    'D Loss': f'{d_loss.item():.4f}',
                    'G Loss': f'{g_loss.item():.4f}'
                })
                
            except Exception as e:
                logger.error(f"Error in batch {batch_idx}: {e}")
                continue
        
        if batch_count == 0:
            return 0, 0
            
        avg_d_loss = total_d_loss / batch_count
        avg_g_loss = total_g_loss / batch_count
        
        logger.info(f"Component GAN Epoch {epoch} - D Loss: {avg_d_loss:.4f}, G Loss: {avg_g_loss:.4f}")
        
        return avg_d_loss, avg_g_loss
    
    def train_composition_gan(self, dataloader, epoch):
        self.composition_gan.train()
        total_loss = 0
        batch_count = 0
        
        from tqdm import tqdm
        pbar = tqdm(dataloader, desc=f"Composition GAN Epoch {epoch}")
        
        for batch_idx, (scene_descriptions, component_lists, target_images) in enumerate(pbar):
            try:
                batch_size = len(scene_descriptions)
                
                # Get scene text embeddings
                scene_embeddings = self.cache.get_text_embedding(scene_descriptions).to(self.device)
                target_images = target_images.to(self.device)
                
                # Flatten target images to match feature dimension
                target_features = target_images.view(batch_size, -1)
                if target_features.size(1) != self.config['model']['feature_dim']:
                    target_features = torch.nn.functional.adaptive_avg_pool1d(
                        target_features.unsqueeze(1), 
                        self.config['model']['feature_dim']
                    ).squeeze(1)
                
                # Train Composition GAN
                self.optimizer_g.zero_grad()
                
                # Retrieve or generate component features
                component_features = []
                for components in component_lists:
                    scene_components = []
                    for comp_desc in components:
                        if comp_desc:  # Skip empty strings
                            cached_feature = self.cache.retrieve_component(comp_desc)
                            if cached_feature is not None:
                                scene_components.append(cached_feature.to(self.device))
                            else:
                                # Generate component if not in cache
                                new_feature = self.component_gan.generate_component(
                                    comp_desc, self.cache, num_samples=1
                                ).to(self.device)
                                scene_components.append(new_feature.squeeze(0))
                    
                    # Pad to required number of components
                    while len(scene_components) < self.config['model']['num_components']:
                        scene_components.append(torch.zeros(self.config['model']['feature_dim']).to(self.device))
                    
                    component_features.append(torch.stack(scene_components))
                
                # Generate composition
                generated_compositions = []
                for i in range(batch_size):
                    comp = self.composition_gan(component_features[i], scene_embeddings[i])
                    generated_compositions.append(comp)
                
                generated_tensor = torch.stack(generated_compositions)
                
                # Compute reconstruction loss
                comp_loss = self.feature_loss(generated_tensor, target_features)
                comp_loss.backward()
                self.optimizer_g.step()
                
                total_loss += comp_loss.item()
                batch_count += 1
                pbar.set_postfix({'Loss': f'{comp_loss.item():.4f}'})
                
            except Exception as e:
                logger.error(f"Error in composition batch {batch_idx}: {e}")
                continue
        
        if batch_count == 0:
            return 0
            
        avg_loss = total_loss / batch_count
        logger.info(f"Composition GAN Epoch {epoch} - Loss: {avg_loss:.4f}")
        
        return avg_loss
    
    def train_cpp(self, dataloader, epoch):
        self.cpp.train()
        total_loss = 0
        batch_count = 0
        criterion = torch.nn.MSELoss()
        
        from tqdm import tqdm
        pbar = tqdm(dataloader, desc=f"CPP Epoch {epoch}")
        
        for batch_idx, (images, preference_scores) in enumerate(pbar):
            try:
                images = images.to(self.device)
                preference_scores = preference_scores.to(self.device)
                
                # Flatten images for CPP
                image_features = images.view(images.size(0), -1)
                if image_features.size(1) != self.config['model']['feature_dim']:
                    image_features = torch.nn.functional.adaptive_avg_pool1d(
                        image_features.unsqueeze(1), 
                        self.config['model']['feature_dim']
                    ).squeeze(1)
                
                self.optimizer_cpp.zero_grad()
                
                predicted_scores = self.cpp(image_features)
                loss = criterion(predicted_scores, preference_scores.unsqueeze(1))
                loss.backward()
                self.optimizer_cpp.step()
                
                total_loss += loss.item()
                batch_count += 1
                pbar.set_postfix({'Loss': f'{loss.item():.4f}'})
                
            except Exception as e:
                logger.error(f"Error in CPP batch {batch_idx}: {e}")
                continue
        
        if batch_count == 0:
            return 0
            
        avg_loss = total_loss / batch_count
        logger.info(f"CPP Epoch {epoch} - Loss: {avg_loss:.4f}")
        
        return avg_loss
    
    def save_checkpoint(self, epoch, filepath):
        checkpoint = {
            'epoch': epoch,
            'component_gan_state_dict': self.component_gan.state_dict(),
            'composition_gan_state_dict': self.composition_gan.state_dict(),
            'cpp_state_dict': self.cpp.state_dict(),
            'optimizer_g_state_dict': self.optimizer_g.state_dict(),
            'optimizer_d_state_dict': self.optimizer_d.state_dict(),
            'optimizer_cpp_state_dict': self.optimizer_cpp.state_dict(),
            'cache': self.cache.cache,
            'config': self.config
        }
        torch.save(checkpoint, filepath)
        logger.info(f"Checkpoint saved: {filepath}")
    
    def load_checkpoint(self, filepath):
        checkpoint = torch.load(filepath, map_location=self.device)
        self.component_gan.load_state_dict(checkpoint['component_gan_state_dict'])
        self.composition_gan.load_state_dict(checkpoint['composition_gan_state_dict'])
        self.cpp.load_state_dict(checkpoint['cpp_state_dict'])
        self.optimizer_g.load_state_dict(checkpoint['optimizer_g_state_dict'])
        self.optimizer_d.load_state_dict(checkpoint['optimizer_d_state_dict'])
        self.optimizer_cpp.load_state_dict(checkpoint['optimizer_cpp_state_dict'])
        self.cache.cache = checkpoint['cache']
        
        logger.info(f"Checkpoint loaded: {filepath}")
        return checkpoint['epoch']

def main():
    parser = argparse.ArgumentParser(description='Fixed CC-GAN Training')
    parser.add_argument('--epochs', type=int, default=5, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=4, help='Batch size')
    parser.add_argument('--feature-dim', type=int, default=256, help='Feature dimension')
    args = parser.parse_args()
    
    # Configuration with proper dimensions
    config = {
        'model': {
            'latent_dim': 128,
            'feature_dim': args.feature_dim,
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
            'image_size': 128,
            'num_workers': 0
        },
        'hardware': {
            'device': 'cpu',
            'num_threads': 8,
            'pin_memory': False
        }
    }
    
    print("Initializing Fixed CC-GAN...")
    print(f"Feature dimension: {config['model']['feature_dim']}")
    print(f"Image size: {config['data']['image_size']}")
    print(f"Calculated feature size: {3 * config['data']['image_size'] * config['data']['image_size']}")
    
    # Initialize trainer
    trainer = FixedCCGANTrainer(config)
    
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
                checkpoint_path = f"checkpoints/ccgan_fixed_epoch_{epoch+1}.pth"
                trainer.save_checkpoint(epoch + 1, checkpoint_path)
            
            print(f"Epoch {epoch+1} Results:")
            print(f"  Component GAN - D: {d_loss:.4f}, G: {g_loss:.4f}")
            print(f"  Composition GAN - Loss: {comp_loss:.4f}")
            print(f"  CPP - Loss: {cpp_loss:.4f}")
            print(f"  Cache size: {len(trainer.cache.cache)}")
            
        except Exception as e:
            print(f"Error in epoch {epoch+1}: {e}")
            import traceback
            traceback.print_exc()
            break
    
    print("Training completed!")
    
    # Save final model
    final_checkpoint = "checkpoints/ccgan_fixed_final.pth"
    trainer.save_checkpoint(args.epochs, final_checkpoint)
    print(f"Final model saved: {final_checkpoint}")

if __name__ == "__main__":
    main()
