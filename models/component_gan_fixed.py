import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import CLIPModel, CLIPProcessor

class ComponentCache:
    def __init__(self, cache_size=1000, feature_dim=512):
        self.cache = {}
        self.cache_size = cache_size
        self.feature_dim = feature_dim
        self.text_encoder = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        
        # Freeze text encoder
        for param in self.text_encoder.parameters():
            param.requires_grad = False
        
    def get_text_embedding(self, text):
        if isinstance(text, str):
            text = [text]
        inputs = self.processor(text=text, return_tensors="pt", padding=True)
        with torch.no_grad():
            text_features = self.text_encoder.get_text_features(**inputs)
        return text_features
    
    def store_component(self, text_description, feature_vector):
        if len(self.cache) >= self.cache_size:
            # Remove oldest entry (simple FIFO)
            first_key = next(iter(self.cache))
            del self.cache[first_key]
            
        text_embedding = self.get_text_embedding(text_description)
        key = tuple(text_embedding[0].detach().numpy())
        self.cache[key] = feature_vector.detach()
    
    def retrieve_component(self, text_description):
        text_embedding = self.get_text_embedding(text_description)
        target_key = text_embedding[0].detach().numpy()
        
        if not self.cache:
            return None
            
        # Find nearest neighbor
        best_match = None
        min_distance = float('inf')
        
        for cached_key, feature in self.cache.items():
            distance = torch.dist(torch.tensor(target_key), torch.tensor(cached_key))
            if distance < min_distance:
                min_distance = distance
                best_match = feature
                
        return best_match

class DiscriminatorWithCrossAttention(nn.Module):
    def __init__(self, feature_dim, text_embed_dim=512):
        super().__init__()
        self.feature_proj = nn.Linear(feature_dim, 512)
        self.text_proj = nn.Linear(text_embed_dim, 512)
        self.attention = nn.MultiheadAttention(512, 8, batch_first=True)
        
        self.classifier = nn.Sequential(
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
        
    def forward(self, features, text_embedding):
        # features: (batch_size, feature_dim)
        # text_embedding: (batch_size, text_embed_dim)
        
        feat_proj = self.feature_proj(features).unsqueeze(1)  # (batch_size, 1, 512)
        text_proj = self.text_proj(text_embedding).unsqueeze(1)  # (batch_size, 1, 512)
        
        attended, _ = self.attention(feat_proj, text_proj, text_proj)
        output = self.classifier(attended.squeeze(1))
        return output

class ComponentGAN(nn.Module):
    def __init__(self, latent_dim=512, feature_dim=512, text_embed_dim=512):
        super().__init__()
        self.latent_dim = latent_dim
        self.feature_dim = feature_dim
        self.text_embed_dim = text_embed_dim
        
        # Generator - WITHOUT BatchNorm to avoid single sample issues
        self.generator = nn.Sequential(
            nn.Linear(latent_dim + text_embed_dim, 1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            nn.Linear(1024, 2048),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            nn.Linear(2048, feature_dim),
            nn.Tanh()
        )
        
        # Discriminator with cross-attention
        self.discriminator = DiscriminatorWithCrossAttention(feature_dim, text_embed_dim)
        
    def forward(self, z, text_embedding):
        # Ensure proper shapes
        if z.dim() == 1:
            z = z.unsqueeze(0)
        if text_embedding.dim() == 1:
            text_embedding = text_embedding.unsqueeze(0)
            
        # Check dimensions
        if z.size(1) != self.latent_dim:
            raise ValueError(f"Expected z dimension {self.latent_dim}, got {z.size(1)}")
        if text_embedding.size(1) != self.text_embed_dim:
            raise ValueError(f"Expected text_embedding dimension {self.text_embed_dim}, got {text_embedding.size(1)}")
            
        combined = torch.cat([z, text_embedding], dim=1)
        features = self.generator(combined)
        return features
    
    def generate_component(self, text_description, cache=None, num_samples=1):
        # Set to eval mode to avoid BatchNorm issues
        original_mode = self.training
        self.eval()
        
        with torch.no_grad():
            if cache is not None:
                text_embedding = cache.get_text_embedding(text_description)
            else:
                # Create dummy embedding if cache is not available
                text_embedding = torch.randn(1, self.text_embed_dim)
            
            # Repeat text_embedding to match num_samples
            if num_samples > 1:
                text_embedding = text_embedding.repeat(num_samples, 1)
            
            z = torch.randn(num_samples, self.latent_dim)
            features = self.forward(z, text_embedding)
            
            # Restore original mode
            if original_mode:
                self.train()
                
            return features
