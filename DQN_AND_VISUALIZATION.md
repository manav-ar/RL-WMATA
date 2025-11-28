# DQN Training and Visualization - Summary

## What Was Added

### 1. DQN Training Script (`agents/train_dqn.py`)
- Complete DQN training implementation
- Uses `MultiInputPolicy` for dict observations (same as PPO)
- Configurable training timesteps via command-line argument
- TensorBoard logging support
- Saves model to `models/dqn_station_placement.zip`

**Usage:**
```bash
python -m agents.train_dqn --total_timesteps 100000
```

### 2. DQN Testing Script (`test_DQN.py`)
- Loads and evaluates trained DQN model
- Displays selected stations with lat/long coordinates
- Shows performance metrics (coverage, wait time, reward)
- Matches the format of `test_PPO.py`

**Usage:**
```bash
python test_DQN.py
```

### 3. Enhanced PPO Testing (`test_PPO.py`)
- Updated to show lat/long coordinates for selected stations
- Displays detailed station information
- Shows performance metrics

**Usage:**
```bash
python test_PPO.py
```

### 4. Visualization Script (`visualize_results.py`)
- Creates side-by-side map comparison of PPO and DQN
- Shows all candidate locations (light blue dots)
- Highlights selected stations (red stars for PPO, green stars for DQN)
- Annotates stations with step number and candidate ID
- Includes performance metrics in titles
- Handles single-model cases (PPO-only or DQN-only)
- Saves high-resolution PNG to `visualizations/station_placements.png`

**Usage:**
```bash
python visualize_results.py
```

### 5. Model Comparison Script (`compare_models.py`)
- Runs both models for multiple episodes (default: 5)
- Compares average performance metrics
- Shows best episode results for each model
- Displays detailed lat/long coordinates in formatted table
- Identifies winner for each metric (reward, coverage, wait time)
- Comprehensive side-by-side comparison

**Usage:**
```bash
python compare_models.py
```

## Output Format

### Station Coordinates
All scripts output selected stations in this format:
```
1. C45: Lat=38.901234, Lon=-77.012345, Population=5678
2. C123: Lat=38.912345, Lon=-77.023456, Population=4321
...
```

Where:
- **C45, C123**: Candidate station IDs
- **Lat**: Latitude in decimal degrees (6 decimal precision)
- **Lon**: Longitude in decimal degrees (6 decimal precision)
- **Population**: Population count at that location

### Visualization Output
- **File**: `visualizations/station_placements.png`
- **Format**: High-resolution PNG (300 DPI)
- **Content**: 
  - Left panel: PPO selected stations
  - Right panel: DQN selected stations
  - DC boundary overlay (if available)
  - All candidate locations shown for context

## Example Workflow

1. **Train both models:**
   ```bash
   python -m agents.train_ppo --total_timesteps 100000
   python -m agents.train_dqn --total_timesteps 100000
   ```

2. **Test individual models:**
   ```bash
   python test_PPO.py
   python test_DQN.py
   ```

3. **Visualize results:**
   ```bash
   python visualize_results.py
   ```

4. **Compare models:**
   ```bash
   python compare_models.py
   ```

## Key Features

### Lat/Long Display
- All scripts extract and display latitude/longitude from candidate data
- Coordinates shown with 6 decimal precision (~0.1m accuracy)
- Easy to copy for use in mapping tools or GIS software

### Performance Metrics
- **Coverage**: Population coverage fraction (0.0 to 1.0)
- **Avg Wait**: Average wait time in minutes
- **Total Reward**: Episode cumulative reward
- **Trips Served**: Daily trips served between selected stations

### Visualization Features
- Color-coded by model (PPO=red, DQN=green)
- Step numbers show placement order
- Candidate IDs for easy identification
- Performance metrics in plot titles
- DC boundary overlay for geographic context

## Files Created

1. `agents/train_dqn.py` - DQN training script
2. `test_DQN.py` - DQN evaluation script
3. `visualize_results.py` - Map visualization script
4. `compare_models.py` - Model comparison script
5. `USAGE_GUIDE.md` - Detailed usage instructions
6. `DQN_AND_VISUALIZATION.md` - This summary

## Dependencies

All scripts require:
- `stable-baselines3` (for PPO and DQN)
- `pandas` (for data handling)
- `geopandas` (for map visualization)
- `matplotlib` (for plotting)
- `numpy` (for numerical operations)

Install with:
```bash
pip install stable-baselines3 geopandas matplotlib pandas numpy
```

## Next Steps

1. **Train for longer**: Use 500k+ timesteps for better performance
2. **Hyperparameter tuning**: Adjust DQN exploration, learning rate, etc.
3. **Compare with baselines**: Run `test_baseline.py` to see how RL compares
4. **Analyze training**: Use TensorBoard to understand learning dynamics
5. **Export coordinates**: Use lat/long output for GIS analysis or Google Maps

## Notes

- Both PPO and DQN use the same environment and observation space
- Models are deterministic during evaluation (no exploration)
- Visualization works even if only one model is trained
- All coordinates are in WGS84 (EPSG:4326) format
- Station selection order matters (shown by step numbers)

