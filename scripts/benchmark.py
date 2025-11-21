#!/usr/bin/env python3
import sys
import os
import time

# Add the project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

try:
    from models.component_gan_fixed import ComponentGAN, ComponentCache
    import torch
    print("✅ All imports successful!")
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Current Python path:", sys.path)
    sys.exit(1)

def benchmark():
    print("🚀 Benchmarking CC-GAN Performance...")
    
    # Initialize model
    model = ComponentGAN(128, 256, 512)
    cache = ComponentCache(100, 256)
    
    print("✅ Model initialized successfully")
    
    # Benchmark component generation
    times = []
    print("Generating test components...")
    for i in range(3):  # Reduced for quick testing
        try:
            start = time.time()
            component = model.generate_component(f"test component {i}", cache, 1)
            end = time.time()
            times.append(end - start)
            print(f"  ✅ Generated component {i+1}/3 - {component.shape}")
        except Exception as e:
            print(f"  ❌ Failed to generate component {i+1}: {e}")
            continue
    
    if times:
        avg_time = sum(times) / len(times)
        print(f"✅ Average component generation time: {avg_time*1000:.2f} ms")
        print(f"✅ Theoretical FPS: {1/avg_time:.2f}")
    else:
        print("❌ No successful component generations")
        return
    
    # Check memory usage
    param_count = sum(p.numel() for p in model.parameters())
    print(f"✅ Model parameters: {param_count:,}")
    print(f"✅ Estimated model size: {param_count * 4 / 1024 / 1024:.2f} MB")
    
    print("\n📊 Paper Target Comparison:")
    print("   - Viewpoint Accuracy: >95%")
    print("   - FLOPs Reduction: 60-70%")
    print("   - Originality Improvement: 20%")
    print("   - Inference Speed: <1 second")
    print("\n🎯 Next: Run training to get actual performance metrics")

if __name__ == "__main__":
    benchmark()
