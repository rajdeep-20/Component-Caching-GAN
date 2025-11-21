#!/usr/bin/env python3
import os
import torch

print("🧪 Quick Training Test")

# Create checkpoints directory if it doesn't exist
if not os.path.exists('checkpoints'):
    os.makedirs('checkpoints')
    print("✅ Created checkpoints directory")

# Create a dummy checkpoint to test
dummy_checkpoint = {
    'epoch': 1,
    'cache': {'test': torch.randn(256)},
    'config': {
        'model': {
            'feature_dim': 256,
            'latent_dim': 128,
            'num_components': 3
        }
    }
}

# Save dummy checkpoint
torch.save(dummy_checkpoint, 'checkpoints/dummy_test.pth')
print("✅ Created dummy checkpoint: checkpoints/dummy_test.pth")

# Verify it exists
if os.path.exists('checkpoints/dummy_test.pth'):
    print("✅ Checkpoint file verified")
    file_size = os.path.getsize('checkpoints/dummy_test.pth') / 1024
    print(f"✅ File size: {file_size:.2f} KB")
else:
    print("❌ Failed to create checkpoint")
