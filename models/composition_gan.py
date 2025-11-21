import torch
import torch.nn as nn
import torch.nn.functional as F

class UNetLikeComposer(nn.Module):
    def __init__(self, feature_dim, num_components, output_dim):
        super().__init__()
        self.feature_dim = feature_dim
        self.num_components = num_components
        self.output_dim = output_dim
        
        # Calculate input dimension for encoder
        encoder_input_dim = feature_dim * num_components
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(encoder_input_dim, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(512 + 512, 1024),  # 512 from encoder + 512 text embedding
            nn.ReLU(inplace=True),
            nn.Linear(1024, output_dim),
            nn.Tanh()
        )
        
    def forward(self, component_features, text_embedding):
        # component_features: list of tensors, each can be 1D or 2D
        # Determine batch size from the first component
        first_comp = component_features[0]
        batch_size = first_comp.size(0) if first_comp.dim() > 1 else 1
        
        # Process each component to ensure consistent dimensions
        processed_components = []
        for feat in component_features:
            # Ensure at least 2D and same batch size
            if feat.dim() == 1:
                feat = feat.unsqueeze(0).repeat(batch_size, 1) if batch_size > 1 else feat.unsqueeze(0)
            elif feat.size(0) != batch_size and batch_size > 1:
                # Repeat to match batch size if needed
                feat = feat.repeat(batch_size, 1)
            processed_components.append(feat)
        
        # Concatenate all component features along feature dimension
        combined_features = torch.cat(processed_components, dim=1)
        
        # Check input dimension
        expected_dim = self.feature_dim * self.num_components
        if combined_features.size(1) != expected_dim:
            raise ValueError(f"Expected input dimension {expected_dim}, got {combined_features.size(1)}")
        
        encoded = self.encoder(combined_features)
        
        # Ensure text embedding has correct dimension and batch size
        if text_embedding.dim() == 1:
            text_embedding = text_embedding.unsqueeze(0)
        
        # Repeat text embedding to match batch size if needed
        if text_embedding.size(0) != batch_size and batch_size > 1:
            text_embedding = text_embedding.repeat(batch_size, 1)
        
        # Ensure text embedding has correct dimension (512)
        if text_embedding.size(1) != 512:
            # Simple projection if needed
            if text_embedding.size(1) < 512:
                # Pad with zeros
                padding = torch.zeros(text_embedding.size(0), 512 - text_embedding.size(1))
                text_embedding = torch.cat([text_embedding, padding], dim=1)
            else:
                # Truncate
                text_embedding = text_embedding[:, :512]
        
        decoder_input = torch.cat([encoded, text_embedding], dim=1)
        composed = self.decoder(decoder_input)
        
        return composed

class CompositionGAN(nn.Module):
    def __init__(self, num_components=5, output_dim=512, component_feature_dim=512):
        super().__init__()
        self.num_components = num_components
        self.output_dim = output_dim
        self.component_feature_dim = component_feature_dim
        
        # Component processor
        self.component_processor = nn.Sequential(
            nn.Linear(component_feature_dim, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 512)
        )
        
        # U-Net like architecture for composition
        self.composition_net = UNetLikeComposer(
            feature_dim=512,  # Processed component dimension
            num_components=num_components, 
            output_dim=output_dim
        )
        
        # Cross-attention for CALL
        self.cross_attention = nn.MultiheadAttention(512, 8, batch_first=True)
        
    def forward(self, component_features, text_prompt, viewpoint_tokens=None):
        # Determine batch size from components
        batch_size = None
        for feat in component_features:
            if feat.dim() > 1:
                batch_size = feat.size(0)
                break
        if batch_size is None:
            # All are 1D, so batch size is 1
            batch_size = 1
        
        # Process each component to consistent dimension
        processed_components = []
        for feat in component_features:
            # Ensure the tensor is at least 2D
            if feat.dim() == 1:
                feat = feat.unsqueeze(0)
            
            # Get current feature dimension
            current_dim = feat.size(-1)  # Use last dimension to handle both 1D and 2D
            
            # Process to consistent dimension if needed
            if current_dim != self.component_feature_dim:
                # Use adaptive approach based on tensor dimensions
                if feat.dim() == 2:
                    # 2D tensor: [batch, features]
                    feat = F.adaptive_avg_pool1d(feat.unsqueeze(1), self.component_feature_dim).squeeze(1)
                else:
                    # 1D tensor: [features] - convert to 2D then process
                    feat_2d = feat.unsqueeze(0)
                    feat_2d = F.adaptive_avg_pool1d(feat_2d.unsqueeze(1), self.component_feature_dim).squeeze(1)
                    feat = feat_2d.squeeze(0) if feat.dim() == 1 else feat_2d
            
            processed = self.component_processor(feat)
            processed_components.append(processed)
        
        # Apply CALL if viewpoint tokens provided
        if viewpoint_tokens is not None:
            composed = self.apply_call(processed_components, text_prompt, viewpoint_tokens)
        else:
            composed = self.composition_net(processed_components, text_prompt)
            
        return composed
    
    def compute_cross_attention(self, components, text_prompt):
        # Ensure components are properly stacked
        component_tensors = []
        batch_size = components[0].size(0) if components[0].dim() > 1 else 1
        
        for comp in components:
            if comp.dim() == 1:
                comp = comp.unsqueeze(0).repeat(batch_size, 1) if batch_size > 1 else comp.unsqueeze(0)
            component_tensors.append(comp)
        
        component_tensor = torch.stack(component_tensors, dim=1)  # (batch, num_components, feature_dim)
        
        # Ensure text prompt has correct dimension for attention
        if text_prompt.dim() == 1:
            text_prompt = text_prompt.unsqueeze(0)
        
        # Repeat text prompt to match batch size if needed
        if text_prompt.size(0) != batch_size and batch_size > 1:
            text_prompt = text_prompt.repeat(batch_size, 1)
            
        text_expanded = text_prompt.unsqueeze(1).repeat(1, component_tensor.size(1), 1)
        
        attention_output, attention_weights = self.cross_attention(
            component_tensor, text_expanded, text_expanded
        )
        return attention_weights
    
    def apply_viewpoint_mask(self, attention_weights, viewpoint_tokens):
        # Apply spatial constraints based on viewpoint
        batch_size, num_components, _ = attention_weights.shape
        
        # Create a simple mask based on viewpoint
        masks = []
        for viewpoint in viewpoint_tokens:
            if "top" in viewpoint.lower():
                # Top view mask - focus on center
                mask = torch.ones(num_components)
                mask[:num_components//2] = 0.1  # Reduce attention to first half
            elif "side" in viewpoint.lower():
                # Side view mask
                mask = torch.ones(num_components)
                mask[num_components//2:] = 0.1  # Reduce attention to second half
            else:
                # Front view - uniform attention
                mask = torch.ones(num_components)
            masks.append(mask)
        
        mask_tensor = torch.stack(masks).unsqueeze(-1)
        masked_attention = attention_weights * mask_tensor
        return masked_attention
    
    def apply_call(self, components, text_prompt, viewpoint_tokens):
        # Coupled Attention Localization implementation
        attention_weights = self.compute_cross_attention(components, text_prompt)
        masked_attention = self.apply_viewpoint_mask(attention_weights, viewpoint_tokens)
        
        # Stack components properly
        component_tensors = []
        batch_size = components[0].size(0) if components[0].dim() > 1 else 1
        
        for comp in components:
            if comp.dim() == 1:
                comp = comp.unsqueeze(0).repeat(batch_size, 1) if batch_size > 1 else comp.unsqueeze(0)
            component_tensors.append(comp)
        component_tensor = torch.stack(component_tensors, dim=1)
        
        # Apply the masked attention to components
        attended_components = torch.bmm(masked_attention.transpose(1, 2), component_tensor)
        
        # Convert back to list format for composition net
        attended_list = []
        for i in range(attended_components.size(1)):
            comp = attended_components[:, i, :]
            # Remove batch dimension if original was 1D and batch_size is 1
            if components[i].dim() == 1 and comp.size(0) == 1:
                comp = comp.squeeze(0)
            attended_list.append(comp)
        
        composed = self.composition_net(attended_list, text_prompt)
        
        return composed
