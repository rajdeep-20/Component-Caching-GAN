import torch
import torch.nn as nn
import torch.nn.functional as F

class CrossAttentionLoss(nn.Module):
    """
    Cross-attention loss for text-component mapping
    """
    def __init__(self, lambda_attention=1.0):
        super().__init__()
        self.lambda_attention = lambda_attention
        
    def forward(self, attention_maps, target_regions):
        """
        Args:
            attention_maps: (batch_size, num_tokens, height, width)
            target_regions: (batch_size, num_tokens, height, width) binary masks
        """
        batch_size, num_tokens, h, w = attention_maps.shape
        
        loss = 0.0
        for i in range(batch_size):
            for j in range(num_tokens):
                # Get attention map and target region for this token
                attn_map = attention_maps[i, j]
                target_region = target_regions[i, j]
                
                # Encourage high attention in target region, low elsewhere
                pos_loss = -torch.log(attn_map[target_region == 1] + 1e-8).mean()
                neg_loss = -torch.log(1 - attn_map[target_region == 0] + 1e-8).mean()
                
                loss += (pos_loss + neg_loss) / 2
                
        return self.lambda_attention * loss / (batch_size * num_tokens)

class CcGANLoss(nn.Module):
    """
    Continuous Conditional GAN loss
    """
    def __init__(self):
        super().__init__()
        
    def forward(self, discriminator_output, real_labels, target_scores):
        """
        Args:
            discriminator_output: Discriminator predictions
            real_labels: Whether samples are real or fake
            target_scores: Target preference scores
        """
        batch_size = discriminator_output.size(0)
        
        # Base adversarial loss
        adv_loss = F.binary_cross_entropy(discriminator_output, real_labels)
        
        # Preference alignment loss
        if target_scores is not None:
            pref_loss = F.mse_loss(discriminator_output, target_scores.unsqueeze(1))
            total_loss = adv_loss + pref_loss
        else:
            total_loss = adv_loss
            
        return total_loss

class MultiViewConsistencyLoss(nn.Module):
    """
    Loss for enforcing consistency across different viewpoints
    """
    def __init__(self, lambda_consistency=0.1):
        super().__init__()
        self.lambda_consistency = lambda_consistency
        self.cosine_sim = nn.CosineSimilarity(dim=1)
        
    def forward(self, features_view1, features_view2):
        """
        Compute consistency between features from different views
        """
        consistency_loss = 1 - self.cosine_sim(features_view1, features_view2).mean()
        return self.lambda_consistency * consistency_loss
