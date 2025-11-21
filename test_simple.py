#!/usr/bin/env python3
import sys
import os

# Add project root to path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

print("Testing project setup...")
print(f"Working directory: {os.getcwd()}")
print(f"Project root: {project_root}")
print(f"Python path: {sys.path}")

# Test basic imports
try:
    import torch
    print("✅ PyTorch imported")
except ImportError as e:
    print(f"❌ PyTorch import failed: {e}")

# Test model imports
try:
    from models.component_gan_fixed import ComponentGAN
    print("✅ ComponentGAN imported")
except ImportError as e:
    print(f"❌ ComponentGAN import failed: {e}")

try:
    from models.composition_gan import CompositionGAN
    print("✅ CompositionGAN imported")
except ImportError as e:
    print(f"❌ CompositionGAN import failed: {e}")

try:
    from models.cpp import ConsumerPreferencePredictor
    print("✅ CPP imported")
except ImportError as e:
    print(f"❌ CPP import failed: {e}")

print("\n🎯 If all imports work, your project is set up correctly!")
