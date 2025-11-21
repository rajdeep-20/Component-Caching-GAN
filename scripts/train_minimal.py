import torch.nn.functional as F
#!/usr/bin/env python3

import os
import sys
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
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MinimalTrainer:
    def __init__(self, config):
        self.config = config
        self.device = torch.device(config['hardware']['device'])
        
        # Model dimensions
        self.latent_dim = config['model']['latent_dim']
        self.feature_dim = config['model']['feature_dim']
        self.text_embed_dim = 512  # Fixed for CLIP
        
        logger.info(f"Initializing models with feature_dim={self.feature_dim}")
        
        # Initialize models
        self.component_gan = ComponentGAN(
            latent_dim=self.latent_dim,
            feature_dim=self.feature_dim,
            text_embed_dim=self.text_embed_dim
        )
        
        self.composition_gan = CompositionGAN(
            num_components=config['model']['num_components'],
            output_dim=self.feature_dim,
            component_feature_dim=self.feature_dim
        )
        
        self.cpp = ConsumerPreferencePredictor(
            input_dim=self.feature_dim,
            hidden_dims=[256, 128]  # Smaller for stability
        )
        
        self.cache = ComponentCache(
            cache_size=config['model']['cache_size'],
            feature_dim=self.feature_dim
        )
        
        # Single optimizer for simplicity
        self.optimizer = torch.optim.Adam(
            list(self.component_gan.parameters()) + 
            list(self.composition_gan.parameters()) +
            list(self.cpp.parameters()),
            lr=config['training']['learning_rate']
        )
        
        # Loss functions
        self.adversarial_loss = torch.nn.BCELoss()
        self.feature_loss = torch.nn.MSELoss()
        
        # Move to device
        self.component_gan.to(self.device)
        self.composition_gan.to(self.device)
        self.cpp.to(self.device)
        
    def train_epoch(self, dataloader, epoch):
        self.component_gan.train()
        self.composition_gan.train()
        self.cpp.train()
        
        total_loss = 0
        batch_count = 0
        
        for batch_idx, (images, descriptions) in enumerate(dataloader):
            try:
                batch_size = images.size(0)
                images = images.to(self.device)
                
                # Get text embeddings
                text_embeddings = self.cache.get_text_embedding(descriptions).to(self.device)
                
                # Flatten images to features
                image_features = images.view(batch_size, -1)
                if image_features.size(1) != self.feature_dim:
                    # Adaptive pooling to match dimension
                    image_features = F.adaptive_avg_pool1d(
                        image_features.unsqueeze(1), 
                        self.feature_dim
                    ).squeeze(1)
                
                # Train step
                self.optimizer.zero_grad()
                
                # Generate components
                z = torch.randn(batch_size, self.latent_dim).to(self.device)
                generated_components = self.component_gan(z, text_embeddings)
                
                # Simple composition: just use the generated component directly
                # (In real training, we'd compose multiple components)
                composition_loss = self.feature_loss(generated_components, image_features)
                
                # Adversarial loss for component GAN
                real_labels = torch.ones(batch_size, 1).to(self.device)
                fake_pred = self.component_gan.discriminator(generated_components, text_embeddings)
                adversarial_loss = self.adversarial_loss(fake_pred, real_labels)
                
                # CPP loss
                preference_scores = self.cpp(generated_components)
                target_scores = torch.ones(batch_size, 1).to(self.device) * 0.8  # Target high preference
                cpp_loss = self.feature_loss(preference_scores, target_scores)
                
                # Total loss
                total_batch_loss = composition_loss + adversarial_loss + cpp_loss
                total_batch_loss.backward()
                self.optimizer.step()
                
                # Cache components
                for i, desc in enumerate(descriptions):
                    self.cache.store_component(desc, generated_components[i].detach().cpu())
                
                total_loss += total_batch_loss.item()
                batch_count += 1
                
                if batch_idx % 10 == 0:
                    logger.info(f"Epoch {epoch}, Batch {batch_idx}: Loss = {total_batch_loss.item():.4f}")
                    
            except Exception as e:
                logger.error(f"Error in batch {batch_idx}: {e}")
                continue
        
        if batch_count == 0:
            return float('inf')
            
        return total_loss / batch_count
    
    def save_checkpoint(self, epoch, filepath):
        checkpoint = {
            'epoch': epoch,
            'component_gan_state_dict': self.component_gan.state_dict(),
            'composition_gan_state_dict': self.composition_gan.state_dict(),
            'cpp_state_dict': self.cpp.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'cache': self.cache.cache,
            'config': self.config
        }
        torch.save(checkpoint, filepath)
        logger.info(f"Checkpoint saved: {filepath}")

def main():
    parser = argparse.ArgumentParser(description='Minimal CC-GAN Training')
    parser.add_argument('--epochs', type=int, default=3, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=2, help='Batch size')
    parser.add_argument('--feature-dim', type=int, default=256, help='Feature dimension')
    args = parser.parse_args()
    
    # Minimal configuration
    config = {
        'model': {
            'latent_dim': 128,
            'feature_dim': args.feature_dim,
            'cache_size': 50,
            'num_components': 2  # Reduced for simplicity
        },
        'training': {
            'batch_size': args.batch_size,
            'learning_rate': 0.0002,
            'epochs': args.epochs,
            'save_interval': 1
        },
        'data': {
            'image_size': 64,  # Smaller for faster training
            'num_workers': 0
        },
        'hardware': {
            'device': 'cpu',
            'num_threads': 8,
            'pin_memory': False
        }
    }
    
    logger.info("Starting Minimal CC-GAN Training")
    logger.info(f"Configuration: {config}")
    
    # Initialize trainer
    trainer = MinimalTrainer(config)
    
    # Get data loaders (using component dataset for simplicity)
    data_loaders = get_data_loaders(config)
    
    # Create checkpoints directory
    os.makedirs('checkpoints', exist_ok=True)
    
    # Training loop
    for epoch in range(args.epochs):
        logger.info(f"\n=== Epoch {epoch+1}/{args.epochs} ===")
        
        try:
            avg_loss = trainer.train_epoch(data_loaders['component'], epoch + 1)
            logger.info(f"Epoch {epoch+1} completed. Average Loss: {avg_loss:.4f}")
            logger.info(f"Cache size: {len(trainer.cache.cache)}")
            
            # Save checkpoint every epoch
            checkpoint_path = f"checkpoints/ccgan_minimal_epoch_{epoch+1}.pth"
            trainer.save_checkpoint(epoch + 1, checkpoint_path)
            
        except Exception as e:
            logger.error(f"Error in epoch {epoch+1}: {e}")
            import traceback
            traceback.print_exc()
            break
    
    logger.info("Training completed!")
    
    # Save final model
    final_checkpoint = "checkpoints/ccgan_minimal_final.pth"
    trainer.save_checkpoint(args.epochs, final_checkpoint)
    logger.info(f"Final model saved: {final_checkpoint}")

if __name__ == "__main__":
    main()
