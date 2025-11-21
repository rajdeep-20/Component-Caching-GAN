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
from evaluation.metrics import CCGANEvaluator
from data.dataloaders import get_data_loaders

def evaluate_model(config_path, checkpoint_path):
    # Load configuration
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
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
    
    # Load checkpoint
    device = config['hardware']['device']
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    component_gan.load_state_dict(checkpoint['component_gan_state_dict'])
    composition_gan.load_state_dict(checkpoint['composition_gan_state_dict'])
    cpp.load_state_dict(checkpoint['cpp_state_dict'])
    cache.cache = checkpoint['cache']
    
    # Move models to device
    component_gan.to(device)
    composition_gan.to(device)
    cpp.to(device)
    
    # Set models to evaluation mode
    component_gan.eval()
    composition_gan.eval()
    cpp.eval()
    
    # Initialize evaluator
    evaluator = CCGANEvaluator(device=device)
    
    # Get test data loader
    config['data']['num_workers'] = 2  # Reduce for evaluation
    data_loaders = get_data_loaders(config)
    
    # Run comprehensive evaluation
    print("Running comprehensive evaluation...")
    results = evaluator.comprehensive_evaluation(
        composition_gan, 
        data_loaders['composition'],
        cpp_model=cpp
    )
    
    print("\n" + "="*60)
    print("CC-GAN EVALUATION RESULTS")
    print("="*60)
    
    for metric, value in results.items():
        if metric == 'efficiency_metrics':
            print(f"\nEfficiency Metrics:")
            for eff_metric, eff_value in value.items():
                print(f"  {eff_metric}: {eff_value:.4f}")
        else:
            print(f"{metric.replace('_', ' ').title()}: {value:.4f}")
    
    # Check against paper targets
    print("\n" + "="*60)
    print("PAPER TARGET COMPARISON")
    print("="*60)
    
    targets = {
        'Viewpoint Accuracy': 0.95,
        'CLIP Score': 0.25,  # Typical good CLIP score
        'Preference Alignment': 0.7
    }
    
    for target_name, target_value in targets.items():
        actual_value = None
        if target_name == 'Viewpoint Accuracy':
            actual_value = results.get('avg_viewpoint_accuracy', 0)
        elif target_name == 'CLIP Score':
            actual_value = results.get('avg_clip_score', 0)
        elif target_name == 'Preference Alignment':
            actual_value = results.get('preference_alignment_rate', 0)
        
        if actual_value is not None:
            status = "✓ ACHIEVED" if actual_value >= target_value else "✗ MISSED"
            print(f"{target_name}: {actual_value:.4f} vs Target: {target_value:.4f} {status}")
    
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate CC-GAN')
    parser.add_argument('--config', type=str, default='configs/base_config.yaml',
                       help='Path to config file')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.checkpoint):
        print(f"Error: Checkpoint file {args.checkpoint} does not exist!")
        sys.exit(1)
    
    evaluate_model(args.config, args.checkpoint)
