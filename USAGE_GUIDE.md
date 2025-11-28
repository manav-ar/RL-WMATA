# Usage Guide - DQN Training, Testing, and Visualization

This guide explains how to train DQN, test both PPO and DQN models, and visualize their results.

## Quick Start

### 1. Train DQN Agent

```bash
# Train with default 10,000 timesteps
python -m agents.train_dqn

# Train with custom timesteps (recommended: 100,000+)
python -m agents.train_dqn --total_timesteps 100000
```

### 2. Test Individual Models

```bash
# Test PPO model
python test_PPO.py

# Test DQN model
python test_DQN.py
```

Both scripts will output:
- Selected station IDs
- Latitude and Longitude for each selected station
- Performance metrics (coverage, wait time, reward)

### 3. Visualize Results

```bash
# Create side-by-side comparison map
python visualize_results.py
```

This creates a visualization saved to `visualizations/station_placements.png` showing:
- All candidate locations (light blue)
- PPO selected stations (red stars)
- DQN selected stations (green stars)
- Performance metrics for each model

### 4. Compare Models

```bash
# Run comprehensive comparison
python compare_models.py
```

This script:
- Runs both models for 5 episodes each
- Compares average performance metrics
- Shows best episode results
- Displays detailed lat/long coordinates for all selected stations
- Identifies which model performs better on each metric

## Output Format

### Station Coordinates

Each selected station is displayed with:
```
1. C123: Lat=38.901234, Lon=-77.012345, Population=1234
```

Where:
- `C123` is the candidate ID
- `Lat` is latitude (decimal degrees)
- `Lon` is longitude (decimal degrees)
- `Population` is the population at that location

### Performance Metrics

- **Coverage**: Fraction of population within catchment radius (0.0 to 1.0)
- **Avg Wait**: Average wait time in minutes
- **Total Reward**: Cumulative reward for the episode
- **Trips Served**: Number of trips served per day

## Example Output

### PPO Output
```
PPO Selected Stations (with Lat/Long)
============================================================
1. C45: Lat=38.901234, Lon=-77.012345, Population=5678
2. C123: Lat=38.912345, Lon=-77.023456, Population=4321
3. C78: Lat=38.923456, Lon=-77.034567, Population=3456
4. C234: Lat=38.934567, Lon=-77.045678, Population=2345
5. C56: Lat=38.945678, Lon=-77.056789, Population=1234

Total Reward: 2.456
Final Coverage: 0.789
Final Avg Wait: 3.45 minutes
```

### Comparison Output
```
MODEL COMPARISON - Selected Stations with Lat/Long
====================================================================================================
                    PPO (Best Episode)                     |                  DQN (Best Episode)                  
----------------------------------------------------------------------------------------------------
                      C45                                   |                      C67                             
              Lat: 38.901234                                |              Lat: 38.912345                         
              Lon: -77.012345                               |              Lon: -77.023456                       
...
```

## File Structure

After running all scripts, you'll have:

```
RL-WMATA/
├── models/
│   ├── ppo_station_placement.zip    # Trained PPO model
│   └── dqn_station_placement.zip     # Trained DQN model
├── visualizations/
│   └── station_placements.png        # Comparison map
└── logs/
    └── [training logs for TensorBoard]
```

## TensorBoard

View training progress:

```bash
tensorboard --logdir ./logs/
```

Then open http://localhost:6006 in your browser.

## Troubleshooting

### Model Not Found
If you see `❌ Model not found`, train the model first:
```bash
python -m agents.train_ppo --total_timesteps 50000
python -m agents.train_dqn --total_timesteps 50000
```

### Visualization Issues
If the map doesn't show correctly:
- Ensure `geopandas` and `matplotlib` are installed
- Check that `data/dc_open/dc_boundary.geojson` exists (optional, for DC boundary overlay)

### Import Errors
Make sure all dependencies are installed:
```bash
pip install stable-baselines3 geopandas matplotlib pandas numpy
```

## Next Steps

1. **Train for longer**: Increase timesteps to 500k+ for better performance
2. **Hyperparameter tuning**: Adjust learning rates, exploration, etc.
3. **Compare with baselines**: Run `python test_baseline.py` to compare with greedy/kmeans
4. **Analyze results**: Use TensorBoard to understand training dynamics

