import torch
import numpy as np
from scipy import linalg
import torch.nn as nn
from transformers import CLIPModel, CLIPProcessor
import time

class CCGANEvaluator:
    def __init__(self, device='cpu'):
        self.device = torch.device(device)
        
        # Load CLIP for text-image alignment
        self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        self.clip_model.to(self.device)
        self.clip_model.eval()
        
    def compute_clip_score(self, generated_images, text_descriptions):
        """
        Compute CLIP score for text-image alignment
        """
        with torch.no_grad():
            # Process images and text
            inputs = self.clip_processor(
                text=text_descriptions, 
                images=generated_images, 
                return_tensors="pt", 
                padding=True
            )
            
            # Move to device
            for key in inputs:
                inputs[key] = inputs[key].to(self.device)
            
            # Get embeddings
            image_features = self.clip_model.get_image_features(inputs['pixel_values'])
            text_features = self.clip_model.get_text_features(inputs['input_ids'])
            
            # Normalize features
            image_features = image_features / image_features.norm(dim=1, keepdim=True)
            text_features = text_features / text_features.norm(dim=1, keepdim=True)
            
            # Compute cosine similarity
            clip_scores = (image_features * text_features).sum(dim=1)
            
        return clip_scores.mean().item()
    
    def compute_viewpoint_accuracy(self, generated_images, target_viewpoints, viewpoint_classifier=None):
        """
        Compute viewpoint classification accuracy
        """
        if viewpoint_classifier is None:
            # Use a simple heuristic based on image moments
            accuracies = []
            for i, (image, target_view) in enumerate(zip(generated_images, target_viewpoints)):
                # Convert to numpy for processing
                if isinstance(image, torch.Tensor):
                    image_np = image.cpu().numpy()
                else:
                    image_np = image
                
                # Simple heuristic: aspect ratio and symmetry
                h, w = image_np.shape[-2:]
                aspect_ratio = w / h
                
                # Predict viewpoint based on aspect ratio
                if aspect_ratio > 1.2:
                    pred_view = "side_view"
                elif aspect_ratio < 0.8:
                    pred_view = "top_view" 
                else:
                    pred_view = "front_view"
                
                # Simple string matching for accuracy
                accuracy = 1.0 if pred_view in target_view.lower() else 0.0
                accuracies.append(accuracy)
            
            return np.mean(accuracies)
        else:
            # Use provided classifier
            with torch.no_grad():
                predictions = viewpoint_classifier(generated_images)
                correct = (predictions == target_viewpoints).float().mean()
            return correct.item()
    
    def compute_efficiency_metrics(self, model, input_size=(1, 3, 256, 256), num_runs=10):
        """
        Compute FLOPs and latency metrics
        """
        # Latency measurement
        latencies = []
        
        with torch.no_grad():
            for _ in range(num_runs):
                dummy_input = torch.randn(input_size).to(self.device)
                
                start_time = time.time()
                _ = model(dummy_input)
                end_time = time.time()
                
                latencies.append((end_time - start_time) * 1000)  # Convert to ms
        
        avg_latency = np.mean(latencies)
        std_latency = np.std(latencies)
        
        # FLOPs estimation (simplified)
        # Note: Actual FLOPs calculation would require torchprofile or similar
        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        estimated_flops = total_params * input_size[2] * input_size[3]  # Rough estimate
        
        return {
            'avg_latency_ms': avg_latency,
            'latency_std_ms': std_latency,
            'estimated_flops': estimated_flops,
            'total_parameters': total_params
        }
    
    def compute_preference_alignment(self, generated_images, cpp_model, target_threshold=0.7):
        """
        Compute how well generated images align with consumer preferences
        """
        with torch.no_grad():
            preference_scores = cpp_model(generated_images)
            alignment_rate = (preference_scores >= target_threshold).float().mean()
        
        return alignment_rate.item(), preference_scores.mean().item()
    
    def comprehensive_evaluation(self, model, dataloader, cpp_model=None, viewpoint_classifier=None):
        """
        Run comprehensive evaluation on the model
        """
        results = {
            'clip_scores': [],
            'viewpoint_accuracies': [],
            'preference_scores': [],
            'efficiency': None
        }
        
        # Efficiency metrics
        results['efficiency'] = self.compute_efficiency_metrics(model)
        
        # Quality metrics
        for batch_idx, (images, text_descriptions, viewpoints) in enumerate(dataloader):
            images = images.to(self.device)
            
            # CLIP score
            clip_score = self.compute_clip_score(images, text_descriptions)
            results['clip_scores'].append(clip_score)
            
            # Viewpoint accuracy
            viewpoint_acc = self.compute_viewpoint_accuracy(images, viewpoints, viewpoint_classifier)
            results['viewpoint_accuracies'].append(viewpoint_acc)
            
            # Preference scores if CPP available
            if cpp_model is not None:
                with torch.no_grad():
                    pref_scores = cpp_model(images)
                    results['preference_scores'].extend(pref_scores.cpu().numpy())
        
        # Aggregate results
        final_results = {
            'avg_clip_score': np.mean(results['clip_scores']),
            'avg_viewpoint_accuracy': np.mean(results['viewpoint_accuracies']),
            'efficiency_metrics': results['efficiency']
        }
        
        if results['preference_scores']:
            final_results['avg_preference_score'] = np.mean(results['preference_scores'])
            final_results['preference_alignment_rate'] = np.mean([s >= 0.7 for s in results['preference_scores']])
        
        return final_results
