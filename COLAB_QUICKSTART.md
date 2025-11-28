# Google Colab Quick Start Guide

## 🚀 Fast Setup (5 minutes)

### Step 1: Open Colab
1. Go to [Google Colab](https://colab.research.google.com/)
2. Upload `RL_WMATA_Colab.ipynb`
3. Enable GPU: **Runtime → Change runtime type → GPU (T4)**

### Step 2: Run Setup Cells
```python
# Cell 1: Check GPU
import torch
print(f"GPU: {torch.cuda.is_available()}")

# Cell 2: Install
%pip install -q stable-baselines3[extra] gymnasium geopandas osmnx networkx pandas numpy matplotlib tqdm tensorboard Pillow shapely fiona pyproj rtree
```

### Step 3: Upload Data
```python
# Option A: Mount Drive
from google.colab import drive
drive.mount('/content/drive')
!cp -r /content/drive/MyDrive/RL-WMATA/data ./data

# Option B: Direct Upload
from google.colab import files
uploaded = files.upload()  # Upload data.zip
```

### Step 4: Train
```python
# Run data prep
!python setup_data_improvements.py

# Quick test first (recommended)
!python -m agents.train_ppo --total_timesteps 1000

# Full training (uncomment when ready)
# !python -m agents.train_ppo --total_timesteps 1000000

# Train DQN (same pattern)
!python -m agents.train_dqn --total_timesteps 1000
# !python -m agents.train_dqn --total_timesteps 1000000
```

**Note:** Training scripts now:
- ✅ Auto-detect GPU and adjust settings
- ✅ Test environment before training
- ✅ Handle small test runs (no hanging)
- ✅ Provide progress diagnostics

### Step 5: Visualize
```python
# TensorBoard
%load_ext tensorboard
%tensorboard --logdir ./logs/

# Create visualizations
!python visualize_results.py
!python create_training_gif.py --model both
```

### Step 6: Download
```python
from google.colab import files
files.download('models/ppo_station_placement.zip')
files.download('visualizations/station_placements.png')
```

## ⚡ GPU Optimizations Applied

### Automatic Features
- ✅ **GPU Detection**: Automatically uses GPU if available
- ✅ **Parallel Environments**: 4 envs on GPU (vs 1 on CPU for small runs)
- ✅ **Larger Batches**: 128 batch size on GPU (vs 64 on CPU)
- ✅ **Device Management**: Explicit CUDA device assignment
- ✅ **Smart Environment Setup**: Single env for test runs, 4 for full training
- ✅ **Environment Testing**: Validates environment before training starts
- ✅ **Progress Diagnostics**: Clear messages showing training progress

### Performance Gains
- **CPU**: 100k timesteps ≈ 2-3 hours
- **GPU (T4)**: 100k timesteps ≈ 10-15 minutes
- **Speedup**: ~10-20x faster on GPU

## 📊 Expected Training Times

| Timesteps | CPU Time | GPU Time (T4) | GPU Time (V100) |
|-----------|----------|---------------|-----------------|
| 10,000    | 15 min   | 1-2 min       | 30 sec          |
| 100,000   | 2-3 hrs  | 10-15 min     | 5-8 min         |
| 500,000   | 10-15 hrs| 1-2 hrs       | 30-45 min       |

## 🎯 Key Files for Colab

1. **RL_WMATA_Colab.ipynb** - Complete Colab notebook
2. **COLAB_GUIDE.md** - Detailed guide
3. **requirements_colab.txt** - Dependencies
4. **colab_setup.py** - Setup script

## 💡 Pro Tips

1. **Save to Drive**: Save models periodically to Drive
2. **Use Checkpoints**: Save every 50k timesteps
3. **Monitor GPU**: `!nvidia-smi` to check usage
4. **Long Training**: Use Colab Pro for longer sessions

## 🔧 Troubleshooting

### GPU Not Detected
```python
import torch
torch.cuda.is_available()  # Should be True
# If False: Runtime → Change runtime type → GPU
```

### Out of Memory
- Reduce batch size: `batch_size=64` instead of 128
- Reduce parallel envs: `n_envs=2` instead of 4

### Import Errors
```python
%pip install --upgrade stable-baselines3
```

## Summary

✅ **GPU Optimized**: Automatic detection and usage
✅ **Fast Training**: 10-20x speedup on GPU
✅ **Easy Setup**: One notebook with all steps
✅ **Complete**: Training, visualization, download

Ready to train on Colab with GPU acceleration!

