# colab_setup.py
"""
Setup script for Google Colab.
Installs dependencies and configures GPU.
"""
import os
import subprocess
import sys

def check_gpu():
    """Check if GPU is available."""
    try:
        import torch
        if torch.cuda.is_available():
            print(f"✅ GPU Available: {torch.cuda.get_device_name(0)}")
            print(f"   CUDA Version: {torch.version.cuda}")
            print(f"   GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
            return True
        else:
            print("⚠️  GPU not available, using CPU")
            return False
    except ImportError:
        print("⚠️  PyTorch not installed yet")
        return False

def install_dependencies():
    """Install required packages for Colab."""
    print("📦 Installing dependencies...")
    
    packages = [
        "stable-baselines3[extra]",
        "gymnasium",
        "geopandas",
        "osmnx",
        "networkx",
        "pandas",
        "numpy",
        "matplotlib",
        "tqdm",
        "tensorboard",
        "Pillow",
        "shapely",
        "fiona",
        "pyproj",
        "rtree"
    ]
    
    for package in packages:
        print(f"  Installing {package}...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", package])
    
    print("✅ Dependencies installed")

def setup_colab():
    """Complete Colab setup."""
    print("="*60)
    print("Google Colab Setup for RL-WMATA")
    print("="*60)
    
    # Check GPU
    has_gpu = check_gpu()
    
    # Install dependencies
    install_dependencies()
    
    # Verify GPU again after installation
    if has_gpu:
        check_gpu()
    
    print("\n✅ Colab setup complete!")
    print("\nNext steps:")
    print("1. Upload your data folder to Colab")
    print("2. Run: python -m agents.train_ppo --total_timesteps 100000 --use_gpu")
    print("3. Monitor with: tensorboard --logdir ./logs/")

if __name__ == "__main__":
    setup_colab()

