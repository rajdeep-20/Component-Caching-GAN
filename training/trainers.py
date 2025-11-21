import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import logging
import time
from tqdm import tqdm

class CCGANTrainer:
    def __init__(self, config, component_gan, composition_gan, cpp, cache):
        self.config = config
        self.component_gan = component_gan
        self.composition_gan = composition_gan
        self.cpp = cpp
        self.cache = cache
        
        # Optimizers
        self.optimizer_g = optim.Adam(
            list(self.component_gan.parameters()) + 
            list(self.composition_gan.parameters()),
            lr=config['training']['learning_rate'], 
            betas=(0.5, 0.999)
        )
        
        self.optimizer_d = optim.Adam(
            self.component_gan.discriminator.parameters(),
            lr=config['training']['learning_rate'], 
            betas=(0.5, 0.999)
        )
        
        self.optimizer_cpp = optim.Adam(
            self.cpp.parameters(), 
            lr=config['training']['lr_cpp']
        )
        
        # Loss functions
        self.adversarial_loss = nn.BCELoss()
        self.feature_loss = nn.MSELoss()
        
        self.device = torch.device(config['hardware']['device'])
        
        # Move models to device
        self.component_gan.to(self.device)
        self.composition_gan.to(self.device)
        self.cpp.to(self.device)
        
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
    def train_component_gan(self, dataloader, epoch):
        self.component_gan.train()
        total_d_loss = 0
        total_g_loss = 0
        
        pbar = tqdm(dataloader, desc=f"Component GAN Epoch {epoch}")
        
        for batch_idx, (real_images, text_descriptions) in enumerate(pbar):
            batch_size = real_images.size(0)
            real_images = real_images.to(self.device)
            
            # Get text embeddings
            text_embeddings = self.cache.get_text_embedding(text_descriptions).to(self.device)
            
            # Create labels
            real_labels = torch.ones(batch_size, 1).to(self.device)
            fake_labels = torch.zeros(batch_size, 1).to(self.device)
            
            # Train Discriminator
            self.optimizer_d.zero_grad()
            
            # Real images
            real_pred = self.component_gan.discriminator(real_images, text_embeddings)
            real_loss = self.adversarial_loss(real_pred, real_labels)
            
            # Fake images
            z = torch.randn(batch_size, self.component_gan.latent_dim).to(self.device)
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
            
            # Cache successful components
            if g_loss.item() < 0.5:  # Threshold for successful generation
                for i, desc in enumerate(text_descriptions):
                    self.cache.store_component(desc, fake_features[i].detach().cpu())
            
            pbar.set_postfix({
                'D Loss': f'{d_loss.item():.4f}',
                'G Loss': f'{g_loss.item():.4f}'
            })
        
        avg_d_loss = total_d_loss / len(dataloader)
        avg_g_loss = total_g_loss / len(dataloader)
        
        self.logger.info(f"Component GAN Epoch {epoch} - D Loss: {avg_d_loss:.4f}, G Loss: {avg_g_loss:.4f}")
        
        return avg_d_loss, avg_g_loss
    
    def train_composition_gan(self, dataloader, epoch):
        self.composition_gan.train()
        total_loss = 0
        
        pbar = tqdm(dataloader, desc=f"Composition GAN Epoch {epoch}")
        
        for batch_idx, (scene_descriptions, component_lists, target_images) in enumerate(pbar):
            batch_size = len(scene_descriptions)
            
            # Get scene text embeddings
            scene_embeddings = self.cache.get_text_embedding(scene_descriptions).to(self.device)
            target_images = target_images.to(self.device)
            
            # Train Composition GAN
            self.optimizer_g.zero_grad()
            
            # Retrieve component features from cache
            component_features = []
            for components in component_lists:
                scene_components = []
                for comp_desc in components:
                    cached_feature = self.cache.retrieve_component(comp_desc)
                    if cached_feature is not None:
                        scene_components.append(cached_feature.to(self.device))
                    else:
                        # Generate component if not in cache
                        new_feature = self.component_gan.generate_component(
                            comp_desc, self.cache, num_samples=1
                        ).to(self.device)
                        scene_components.append(new_feature.squeeze(0))
                component_features.append(torch.stack(scene_components))
            
            # Generate composition
            generated_compositions = []
            for i in range(batch_size):
                comp = self.composition_gan(component_features[i], scene_embeddings[i])
                generated_compositions.append(comp)
            
            generated_tensor = torch.stack(generated_compositions)
            
            # Compute reconstruction loss
            comp_loss = self.feature_loss(generated_tensor, target_images)
            comp_loss.backward()
            self.optimizer_g.step()
            
            total_loss += comp_loss.item()
            pbar.set_postfix({'Loss': f'{comp_loss.item():.4f}'})
        
        avg_loss = total_loss / len(dataloader)
        self.logger.info(f"Composition GAN Epoch {epoch} - Loss: {avg_loss:.4f}")
        
        return avg_loss
    
    def train_cpp(self, dataloader, epoch):
        self.cpp.train()
        total_loss = 0
        criterion = nn.MSELoss()
        
        pbar = tqdm(dataloader, desc=f"CPP Epoch {epoch}")
        
        for batch_idx, (images, preference_scores) in enumerate(pbar):
            images = images.to(self.device)
            preference_scores = preference_scores.to(self.device)
            
            self.optimizer_cpp.zero_grad()
            
            predicted_scores = self.cpp(images)
            loss = criterion(predicted_scores, preference_scores.unsqueeze(1))
            loss.backward()
            self.optimizer_cpp.step()
            
            total_loss += loss.item()
            pbar.set_postfix({'Loss': f'{loss.item():.4f}'})
        
        avg_loss = total_loss / len(dataloader)
        self.logger.info(f"CPP Epoch {epoch} - Loss: {avg_loss:.4f}")
        
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
        self.logger.info(f"Checkpoint saved: {filepath}")
    
    def load_checkpoint(self, filepath):
        checkpoint = torch.load(filepath, map_location=self.device)
        self.component_gan.load_state_dict(checkpoint['component_gan_state_dict'])
        self.composition_gan.load_state_dict(checkpoint['composition_gan_state_dict'])
        self.cpp.load_state_dict(checkpoint['cpp_state_dict'])
        self.optimizer_g.load_state_dict(checkpoint['optimizer_g_state_dict'])
        self.optimizer_d.load_state_dict(checkpoint['optimizer_d_state_dict'])
        self.optimizer_cpp.load_state_dict(checkpoint['optimizer_cpp_state_dict'])
        self.cache.cache = checkpoint['cache']
        
        self.logger.info(f"Checkpoint loaded: {filepath}")
        return checkpoint['epoch']
