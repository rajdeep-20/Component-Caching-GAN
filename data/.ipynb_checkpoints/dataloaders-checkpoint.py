import torch
from torch.utils.data import Dataset, DataLoader
import os
from PIL import Image
import json
import numpy as np

class ComponentDataset(Dataset):
    """
    Dataset for training Component-GAN on individual objects/components
    """
    def __init__(self, data_dir, transform=None, split='train'):
        self.data_dir = data_dir
        self.transform = transform
        self.split = split
        
        # Load component data
        self.components = self.load_components()
        
    def load_components(self):
        # This would load from your specific dataset
        # For now, return dummy data structure
        components = []
        
        # Example structure - replace with actual data loading
        if os.path.exists(os.path.join(self.data_dir, 'components.json')):
            with open(os.path.join(self.data_dir, 'components.json'), 'r') as f:
                data = json.load(f)
                components = data.get('components', [])
        else:
            # Create dummy data for testing
            components = [
                {'image_path': 'dummy', 'description': 'modern armchair', 'category': 'furniture'},
                {'image_path': 'dummy', 'description': 'wooden table', 'category': 'furniture'},
                {'image_path': 'dummy', 'description': 'brass floor lamp', 'category': 'lighting'},
                {'image_path': 'dummy', 'description': 'ergonomic chair', 'category': 'furniture'},
                {'image_path': 'dummy', 'description': 'desk', 'category': 'furniture'},
                {'image_path': 'dummy', 'description': 'computer monitor', 'category': 'electronics'},
                {'image_path': 'dummy', 'description': 'sofa', 'category': 'furniture'},
                {'image_path': 'dummy', 'description': 'coffee table', 'category': 'furniture'},
                {'image_path': 'dummy', 'description': 'bookshelf', 'category': 'furniture'},
                {'image_path': 'dummy', 'description': 'floor lamp', 'category': 'lighting'},
            ]
            
        return components
    
    def __len__(self):
        return len(self.components) * 10  # Artificial expansion for training
    
    def __getitem__(self, idx):
        component_idx = idx % len(self.components)
        component = self.components[component_idx]
        
        # Load or generate dummy image - FIXED VERSION
        if component['image_path'] != 'dummy' and os.path.exists(component['image_path']):
            image = Image.open(component['image_path']).convert('RGB')
        else:
            # Create proper dummy image (H, W, C) format
            dummy_array = (np.random.rand(256, 256, 3) * 255).astype(np.uint8)
            image = Image.fromarray(dummy_array, 'RGB')
        
        if self.transform:
            image = self.transform(image)
        
        description = component['description']
        
        return image, description

class CompositionDataset(Dataset):
    """
    Dataset for training Composition-GAN on full scenes
    """
    def __init__(self, data_dir, transform=None, split='train', max_components=5):
        self.data_dir = data_dir
        self.transform = transform
        self.split = split
        self.max_components = max_components
        
        self.scenes = self.load_scenes()
        
    def load_scenes(self):
        scenes = []
        
        if os.path.exists(os.path.join(self.data_dir, 'scenes.json')):
            with open(os.path.join(self.data_dir, 'scenes.json'), 'r') as f:
                data = json.load(f)
                scenes = data.get('scenes', [])
        else:
            # Dummy scenes for testing
            scenes = [
                {
                    'scene_description': 'living room with modern armchair and wooden table',
                    'components': ['modern armchair', 'wooden table', 'floor lamp'],
                    'image_path': 'dummy'
                },
                {
                    'scene_description': 'office with ergonomic chair and desk', 
                    'components': ['ergonomic chair', 'desk', 'computer monitor'],
                    'image_path': 'dummy'
                },
                {
                    'scene_description': 'bedroom with sofa and coffee table',
                    'components': ['sofa', 'coffee table', 'floor lamp'],
                    'image_path': 'dummy'
                },
                {
                    'scene_description': 'study room with desk and bookshelf',
                    'components': ['desk', 'bookshelf', 'computer monitor'],
                    'image_path': 'dummy'
                }
            ]
            
        return scenes
    
    def __len__(self):
        return len(self.scenes) * 20  # Artificial expansion
    
    def __getitem__(self, idx):
        scene_idx = idx % len(self.scenes)
        scene = self.scenes[scene_idx]
        
        # Load or generate dummy scene image - FIXED VERSION
        if scene['image_path'] != 'dummy' and os.path.exists(scene['image_path']):
            image = Image.open(scene['image_path']).convert('RGB')
        else:
            # Create proper dummy image (H, W, C) format
            dummy_array = (np.random.rand(256, 256, 3) * 255).astype(np.uint8)
            image = Image.fromarray(dummy_array, 'RGB')
        
        if self.transform:
            image = self.transform(image)
        
        scene_description = scene['scene_description']
        components = scene['components'][:self.max_components]
        
        # Pad components if necessary
        while len(components) < self.max_components:
            components.append('')
        
        return scene_description, components, image

class PreferenceDataset(Dataset):
    """
    Dataset for training Consumer Preference Predictor
    """
    def __init__(self, data_dir, transform=None, split='train'):
        self.data_dir = data_dir
        self.transform = transform
        self.split = split
        
        self.preference_data = self.load_preference_data()
        
    def load_preference_data(self):
        data = []
        
        if os.path.exists(os.path.join(self.data_dir, 'preferences.json')):
            with open(os.path.join(self.data_dir, 'preferences.json'), 'r') as f:
                loaded_data = json.load(f)
                data = loaded_data.get('preferences', [])
        else:
            # Dummy preference data
            data = [
                {'image_path': 'dummy1', 'preference_score': 0.8},
                {'image_path': 'dummy2', 'preference_score': 0.6},
                {'image_path': 'dummy3', 'preference_score': 0.9},
                {'image_path': 'dummy4', 'preference_score': 0.7},
                {'image_path': 'dummy5', 'preference_score': 0.5},
                {'image_path': 'dummy6', 'preference_score': 0.85},
            ]
            
        return data
    
    def __len__(self):
        return len(self.preference_data) * 15
    
    def __getitem__(self, idx):
        data_idx = idx % len(self.preference_data)
        item = self.preference_data[data_idx]
        
        # Load or generate dummy image - FIXED VERSION
        if item['image_path'].startswith('dummy'):
            # Create proper dummy image (H, W, C) format
            dummy_array = (np.random.rand(256, 256, 3) * 255).astype(np.uint8)
            image = Image.fromarray(dummy_array, 'RGB')
        else:
            image = Image.open(item['image_path']).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        preference_score = torch.tensor(item['preference_score'], dtype=torch.float32)
        
        return image, preference_score

def get_data_loaders(config):
    """
    Create data loaders for all datasets
    """
    from torchvision import transforms
    
    # Data transformations
    transform = transforms.Compose([
        transforms.Resize((config['data']['image_size'], config['data']['image_size'])),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    # Create datasets
    component_dataset = ComponentDataset(
        'datasets/component', 
        transform=transform,
        split='train'
    )
    
    composition_dataset = CompositionDataset(
        'datasets/composition',
        transform=transform, 
        split='train',
        max_components=config['model']['num_components']
    )
    
    preference_dataset = PreferenceDataset(
        'datasets/preference',
        transform=transform,
        split='train'
    )
    
    # Create data loaders with smaller batch sizes for CPU training
    batch_size = min(config['training']['batch_size'], 8)  # Reduce for CPU
    
    component_loader = DataLoader(
        component_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,  # Set to 0 to avoid multiprocessing issues
        pin_memory=False  # Disable for CPU
    )
    
    composition_loader = DataLoader(
        composition_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,  # Set to 0 to avoid multiprocessing issues
        pin_memory=False  # Disable for CPU
    )
    
    preference_loader = DataLoader(
        preference_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,  # Set to 0 to avoid multiprocessing issues
        pin_memory=False  # Disable for CPU
    )
    
    return {
        'component': component_loader,
        'composition': composition_loader,
        'preference': preference_loader
    }