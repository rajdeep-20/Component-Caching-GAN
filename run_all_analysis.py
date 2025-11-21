#!/usr/bin/env python3
import sys
import os
import subprocess

# Add project root to Python path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

def run_analysis():
    print("🚀 CC-GAN Complete Analysis Suite")
    print("=" * 60)
    
    scripts_to_run = [
        "scripts/monitor_training.py",
        "scripts/validate_paper.py", 
        "scripts/demo_design.py",
        "scripts/benchmark_comparison.py",
        "scripts/enhance_call.py"
    ]
    
    for script in scripts_to_run:
        if os.path.exists(script):
            print(f"\n📋 Running {script}...")
            print("-" * 40)
            try:
                # Run each script as a separate process to handle imports
                result = subprocess.run([sys.executable, script], capture_output=True, text=True)
                print(result.stdout)
                if result.stderr:
                    print("STDERR:", result.stderr)
            except Exception as e:
                print(f"❌ Error running {script}: {e}")
        else:
            print(f"❌ Script not found: {script}")
    
    print("\n" + "=" * 60)
    print("🎉 Analysis Complete!")
    print("\n🎯 Recommended Next Steps:")
    print("1. Continue training: python scripts/train_ccgan_final.py --epochs 10")
    print("2. Test generation: python scripts/demo_ccgan.py --checkpoint your_checkpoint.pth")
    print("3. Run evaluation: python scripts/evaluate_fixed.py --checkpoint your_checkpoint.pth")

if __name__ == "__main__":
    run_analysis()
