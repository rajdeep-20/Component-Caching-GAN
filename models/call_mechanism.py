import torch
import torch.nn as nn
import torch.nn.functional as F

class CoupledAttentionLocalization:
    """
    Training-free viewpoint control using attention manipulation
    """
    
    def __init__(self, composition_gan):
        self.composition_gan = composition_gan
        
    def __call__(self, components, text_prompt, viewpoint_description):
        """
        Apply CALL for viewpoint control
        
        Args:
            components: List of component features
            text_prompt: Text embedding of the scene description
            viewpoint_description: String describing viewpoint (e.g., "top view", "side profile")
            
        Returns:
            Composed features with applied viewpoint
        """
        viewpoint_tokens = self.parse_viewpoint_tokens(viewpoint_description)
        return self.composition_gan.apply_call(components, text_prompt, viewpoint_tokens)
    
    def parse_viewpoint_tokens(self, viewpoint_description):
        """
        Parse viewpoint description into tokens for CALL
        """
        tokens = []
        if "top" in viewpoint_description.lower():
            tokens.append("top_view")
        if "side" in viewpoint_description.lower():
            tokens.append("side_view") 
        if "front" in viewpoint_description.lower():
            tokens.append("front_view")
        if "back" in viewpoint_description.lower():
            tokens.append("back_view")
        if "left" in viewpoint_description.lower():
            tokens.append("left_view")
        if "right" in viewpoint_description.lower():
            tokens.append("right_view")
            
        # Default to front view if no specific tokens found
        if not tokens:
            tokens.append("front_view")
            
        return tokens
    
    def generate_multiview(self, components, text_prompt, viewpoints=None):
        """
        Generate multiple views of the same composition
        """
        if viewpoints is None:
            viewpoints = ["front view", "side view", "top view", "back view"]
            
        results = {}
        for viewpoint in viewpoints:
            results[viewpoint] = self(components, text_prompt, viewpoint)
            
        return results
