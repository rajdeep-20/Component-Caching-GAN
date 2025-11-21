#!/usr/bin/env python3
import os
import glob

def track_progress():
    print("📈 Training Progress Tracker")
    print("=" * 50)
    
    # Find all checkpoint files
    checkpoint_files = glob.glob('checkpoints/*.pth')
    
    if not checkpoint_files:
        print("❌ No training progress found.")
        print("💡 Start training: python scripts/train_ccgan_final.py --epochs 3")
        return
    
    print(f"✅ Found {len(checkpoint_files)} checkpoint files")
    print("\n📊 Training Progress:")
    
    epochs = []
    for file in sorted(checkpoint_files):
        filename = os.path.basename(file)
        # Extract epoch number from filename
        if 'epoch' in filename:
            try:
                epoch_num = int(filename.split('_')[-1].split('.')[0])
                epochs.append(epoch_num)
                file_size = os.path.getsize(file) / 1024 / 1024
                print(f"   Epoch {epoch_num}: {file_size:.1f} MB")
            except:
                print(f"   {filename}: {os.path.getsize(file) / 1024 / 1024:.1f} MB")
    
    if epochs:
        max_epoch = max(epochs)
        print(f"\n🎯 Current progress: {max_epoch} epochs completed")
        print(f"📅 Estimated time for 100 epochs: {100/max_epoch * 2:.1f} hours (at current rate)")

if __name__ == "__main__":
    track_progress()
