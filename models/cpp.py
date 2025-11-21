import torch
import torch.nn as nn
import torch.nn.functional as F

class ConsumerPreferencePredictor(nn.Module):
    def __init__(self, input_dim=512, hidden_dims=[1024, 512, 256, 128]):
        super().__init__()
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(0.3)
            ])
            prev_dim = hidden_dim
        
        self.feature_extractor = nn.Sequential(*layers)
        
        self.regressor = nn.Sequential(
            nn.Linear(hidden_dims[-1], 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1),
            nn.Sigmoid()  # Output between 0-1
        )
        
    def forward(self, image_features):
        features = self.feature_extractor(image_features)
        preference_score = self.regressor(features)
        return preference_score

class ContinuousConditionalGAN(nn.Module):
    """
    Continuous Conditional GAN for preference-guided generation
    """
    
    def __init__(self, base_generator, preference_predictor):
        super().__init__()
        self.generator = base_generator
        self.preference_predictor = preference_predictor
        
    def forward(self, components, text_prompt, target_preference=0.8):
        """
        Generate composition with target preference score
        
        Args:
            components: Input component features
            text_prompt: Scene description embedding
            target_preference: Target preference score (0-1)
            
        Returns:
            Generated composition and actual preference score
        """
        # Generate composition
        composition = self.generator(components, text_prompt)
        
        # Predict preference score
        preference_score = self.preference_predictor(composition)
        
        return composition, preference_score
    
    def generate_with_preference_guidance(self, components, text_prompt, 
                                        min_preference=0.7, max_iterations=10):
        """
        Generate multiple variations and select based on preference score
        """
        best_composition = None
        best_score = 0.0
        
        for i in range(max_iterations):
            composition, score = self.forward(components, text_prompt)
            
            if score > best_score and score >= min_preference:
                best_score = score
                best_composition = composition
                
            if best_score >= min_preference:
                break
                
        return best_composition, best_score
