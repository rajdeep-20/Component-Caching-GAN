#!/bin/bash
echo "🚀 CC-GAN Reliable Workflow"
echo "==========================="

cd /home/jupyter/cc-gan-project

echo "1. 📊 Checking current status..."
python scripts/monitor_training.py

echo -e "\n2. 🎯 Starting/continuing training..."
python scripts/train_ccgan_final.py --epochs 3 --batch-size 4 --feature-dim 256

echo -e "\n3. 📈 Checking progress..."
python scripts/training_progress.py

echo -e "\n4. 🎨 Ready for demo/testing!"
echo "   Run: python scripts/demo_ccgan.py --checkpoint checkpoints/ccgan_final_epoch_3.pth"

echo -e "\n✅ Workflow complete!"
