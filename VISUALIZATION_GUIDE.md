# Training Visualization Guide

## Overview
The training scripts now automatically save visualization images every 500 episodes, which can be combined into an animated GIF showing the training progress.

## How It Works

### During Training
1. **Automatic Image Saving**: Every 500 episodes, the training callback saves a visualization image
2. **Image Location**: 
   - PPO: `visualizations/training/ppo/episode_XXXXXX.png`
   - DQN: `visualizations/training/dqn/episode_XXXXXX.png`
3. **Image Content**: Shows selected stations on a map with metrics (reward, coverage, wait time)

### After Training
1. **Create GIF**: Use `create_training_gif.py` to combine images into an animated GIF
2. **View Progress**: See how station selections evolve during training

## Usage

### 1. Train with Visualization
```bash
# Train PPO (images saved automatically)
python -m agents.train_ppo --total_timesteps 100000

# Train DQN (images saved automatically)
python -m agents.train_dqn --total_timesteps 100000
```

During training, you'll see messages like:
```
📸 Training with visualization: saving images every 500 episodes to visualizations/training/ppo/
  💾 Saved training image: visualizations/training/ppo/episode_000500.png
  💾 Saved training image: visualizations/training/ppo/episode_001000.png
  ...
```

### 2. Create Training GIF
```bash
# Create GIF for PPO
python create_training_gif.py --model ppo

# Create GIF for DQN
python create_training_gif.py --model dqn

# Create GIFs for both
python create_training_gif.py --model both
```

### 3. Customize GIF
```bash
# Adjust frame duration (milliseconds)
python create_training_gif.py --model ppo --duration 1000

# Specify output filename
python create_training_gif.py --model ppo --output my_training.gif
```

## Image Frequency

### Default: Every 500 Episodes
- Good balance between detail and file size
- For 100k timesteps (~20k episodes): ~40 images
- For 500k timesteps (~100k episodes): ~200 images

### Adjusting Frequency
Edit `agents/train_ppo.py` or `agents/train_dqn.py`:
```python
viz_callback = VisualizationCallback(
    ...
    episodes_per_image=250,  # Change from 500 to 250 for more frequent images
    ...
)
```

**Recommendations**:
- **250 episodes**: More detailed, larger file size
- **500 episodes**: Balanced (default)
- **1000 episodes**: Less detailed, smaller file size

## Output Files

### Training Images
- **Location**: `visualizations/training/[ppo|dqn]/`
- **Format**: PNG images
- **Naming**: `episode_XXXXXX.png` (6-digit episode number)
- **Size**: ~200-500 KB per image

### GIF Files
- **Location**: `visualizations/`
- **Format**: Animated GIF
- **Naming**: `ppo_training.gif`, `dqn_training.gif`
- **Size**: ~5-50 MB depending on number of frames

## What Each Image Shows

1. **Map View**: 
   - All candidate locations (light blue dots)
   - Selected stations (red stars)
   - DC boundary (if available)

2. **Station Annotations**:
   - Step number (1-5)
   - Candidate ID
   - Yellow highlight boxes

3. **Metrics**:
   - Episode number
   - Total reward
   - Coverage percentage
   - Average wait time

## Tips

### For Best Results
1. **Train for longer**: More episodes = more images = smoother GIF
2. **Use appropriate frequency**: 500 episodes is good for most cases
3. **Monitor file size**: Many images can create large GIFs

### File Management
```bash
# Check how many images were created
ls visualizations/training/ppo/ | wc -l

# Check total size
du -sh visualizations/training/ppo/

# Clean up old images (if needed)
rm visualizations/training/ppo/episode_*.png
```

### Viewing GIFs
- **Browser**: Open the GIF file directly
- **Image viewer**: Most image viewers support animated GIFs
- **Jupyter**: Use `IPython.display.Image` to display in notebooks

## Example Workflow

```bash
# 1. Train PPO for 100k timesteps
python -m agents.train_ppo --total_timesteps 100000

# 2. Wait for training to complete (images saved automatically)

# 3. Create GIF
python create_training_gif.py --model ppo

# 4. View the GIF
open visualizations/ppo_training.gif  # macOS
# or
xdg-open visualizations/ppo_training.gif  # Linux
```

## Troubleshooting

### No Images Created
- Check that training ran long enough (need at least 500 episodes)
- Verify `visualizations/training/` directory exists
- Check training logs for error messages

### GIF Too Large
- Reduce number of images (increase `episodes_per_image`)
- Use `--duration` to make frames faster (smaller file)
- Consider creating separate GIFs for different training phases

### Images Not Updating
- Each image shows a fresh evaluation, so selections may vary
- This is normal - shows exploration during training
- For consistent images, use `deterministic=True` in callback

## Advanced Usage

### Custom Image Frequency
```python
# In train_ppo.py or train_dqn.py
viz_callback = VisualizationCallback(
    ...
    episodes_per_image=100,  # Very frequent (for short training)
    # or
    episodes_per_image=1000,  # Less frequent (for long training)
)
```

### Multiple GIFs
```bash
# Create GIF for first half of training
python create_training_gif.py --model ppo --output ppo_early.gif

# Create GIF for second half (manually select images)
# Or adjust episodes_per_image during training
```

## Summary

✅ **Automatic**: Images saved during training
✅ **Configurable**: Adjust frequency as needed
✅ **Easy**: Simple command to create GIF
✅ **Informative**: See training progress visually

The visualization system helps you understand how the agent learns to place stations over time!

