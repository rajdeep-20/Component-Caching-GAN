#!/usr/bin/env python3

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from data.dataloaders import ComponentDataset, CompositionDataset, PreferenceDataset
from torchvision import transforms

def test_dataloaders():
    print("Testing data loaders...")
    
    # Test transform
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
    ])
    
    try:
        # Test ComponentDataset
        print("Testing ComponentDataset...")
        comp_dataset = ComponentDataset('datasets/component', transform=transform)
        sample = comp_dataset[0]
        print(f"✓ ComponentDataset sample: {type(sample[0])}, {sample[0].shape}, '{sample[1]}'")
        
        # Test CompositionDataset  
        print("Testing CompositionDataset...")
        comp_dataset = CompositionDataset('datasets/composition', transform=transform)
        sample = comp_dataset[0]
        print(f"✓ CompositionDataset sample: {sample[0]}, {len(sample[1])} components, {sample[2].shape}")
        
        # Test PreferenceDataset
        print("Testing PreferenceDataset...")
        pref_dataset = PreferenceDataset('datasets/preference', transform=transform)
        sample = pref_dataset[0]
        print(f"✓ PreferenceDataset sample: {sample[0].shape}, score: {sample[1]}")
        
        print("\n✓ All data loaders working correctly!")
        return True
        
    except Exception as e:
        print(f"✗ Error in data loaders: {e}")
        return False

if __name__ == "__main__":
    test_dataloaders()
