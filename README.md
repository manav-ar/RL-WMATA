# Metro Station Placement Optimization Using Reinforcement Learning

## Overview

This project implements a comprehensive **Reinforcement Learning (RL) framework** for sequential placement of new metro stations in urban transit networks. The goal is to **maximize population coverage** and **minimize average wait time** for passengers through intelligent station location optimization. The project integrates real-world data collection, preparation, network modeling, demand sampling, and state-of-the-art RL agents (PPO and DQN) to solve this complex urban planning problem.

The framework is specifically designed for the Washington Metropolitan Area Transit Authority (WMATA) metro system in Washington D.C., but can be adapted to any transit network with appropriate data. The project structure is modular and allows for experimentation with different RL algorithms, baseline heuristics, and network configurations.

**Key Features:**
- Real-world data integration (WMATA, GTFS, LODES, Census)
- Realistic geographic coverage calculations using Haversine distance
- Time-dependent demand modeling with hourly profiles
- Integration with existing metro infrastructure
- GPU-accelerated training with Stable-Baselines3
- Comprehensive visualization and evaluation tools

---

## Project Goals

The primary objectives of this project are:

1. **Optimize Station Placement**: Determine optimal locations for new metro stations that maximize coverage of population centers
2. **Minimize Wait Times**: Reduce average passenger wait time through strategic station placement and connectivity
3. **Account for Existing Infrastructure**: Consider existing WMATA stations when calculating coverage and connectivity
4. **Real-world Applicability**: Use actual transit data and realistic metrics for practical urban planning applications

---

## Directory Structure

```
RL-WMATA/
│
├── simulator/                      # Core simulation environment
│   ├── env.py                     # Gymnasium environment for sequential station placement
│   ├── simulator_core.py          # Core simulator: simulate(day, station_set) -> metrics
│   ├── demand.py                  # Load and sample demand (LODES -> station OD)
│   └── network.py                 # Build network graph from GTFS / other sources
│
├── agents/                        # RL agent training scripts
│   ├── train_ppo.py              # Train Proximal Policy Optimization agent
│   ├── train_dqn.py              # Train Deep Q-Network agent
│   ├── tb_callback.py            # TensorBoard logging callback
│   └── viz_callback.py           # Visualization callback for training progress
│
├── baselines/                     # Baseline heuristic methods
│   ├── greedy.py                 # Greedy placement baseline (population-based)
│   └── kmeans.py                 # K-Means clustering baseline
│
├── scripts/                       # Data preparation and utility scripts
│   ├── build_candidates.py       # Generate candidate station locations
│   ├── build_graph.py            # Build network graph from GTFS
│   ├── build_od.py               # Build origin-destination demand matrix
│   ├── build_time_travel_graph.py # Build time-based travel graph
│   ├── download_census.py        # Download census population data
│   ├── download_gtfs.py          # Download GTFS transit feeds
│   ├── download_wmata_api.py     # Download WMATA API data
│   ├── enrich_candidates_with_existing_stations.py # Enrich candidates with existing station data
│   ├── improve_network_with_existing_stations.py # Enhance network with existing stations
│   ├── merge_gtfs.py             # Merge multiple GTFS feeds
│   ├── sampler.py                # Demand sampling utilities
│   └── sim.py                    # Simulation utilities
│
├── data/                          # All project data
│   ├── census/                   # Census population data
│   │   └── dc_population.csv
│   ├── dc_open/                  # DC open data
│   │   ├── dc_boundary.geojson
│   │   └── Metro_Station_Entrances_Regional.geojson
│   ├── gtfs/                     # GTFS transit feeds
│   │   ├── bus/                  # Bus GTFS files
│   │   └── rail/                 # Rail GTFS files
│   ├── lodes/                    # LODES origin-destination data
│   │   ├── dc_od.csv.gz
│   │   └── dc_od.csv
│   ├── prepared/                 # Preprocessed data files
│   │   ├── all_routes.csv
│   │   ├── all_stops.csv
│   │   ├── candidates.csv
│   │   ├── candidates.geojson
│   │   ├── candidates_enriched.csv
│   │   ├── candidates_enriched.geojson
│   │   ├── candidates_final.csv
│   │   ├── candidates_final.geojson
│   │   ├── enhanced_network.pkl
│   │   ├── multi_modal_graph.pkl
│   │   ├── network.pkl
│   │   └── station_od.csv
│   └── wmata/                    # WMATA API data
│       ├── bus_routes.json
│       ├── bus_stops.json
│       ├── entrances.json
│       ├── lines.json
│       ├── stations.json
│       └── route_details/        # Individual route details
│
├── models/                        # Trained RL models
│   ├── dqn_station_placement.zip
│   └── ppo_station_placement.zip
│
├── logs/                          # Training logs for TensorBoard
│   ├── DQN_1/                   # DQN training runs
│   ├── DQN_2/
│   ├── DQN_3/
│   ├── DQN_4/
│   ├── PPO_1/                   # PPO training runs
│   ├── PPO_2/
│   └── ...
│
├── visualizations/                # Generated visualizations
│   ├── dqn_training.gif          # DQN training progress animation
│   ├── ppo_training.gif          # PPO training progress animation
│   ├── station_placements.png    # Final station placement comparison
│   └── training/                 # Training progress images
│       ├── dqn/
│       └── ppo/
│
├── test_*.py                      # Test scripts for various components
│   ├── test_sampler.py          # Test demand sampling
│   ├── test_gym.py              # Test environment functionality
│   ├── test_baseline.py         # Test baseline methods
│   ├── test_PPO.py              # Test PPO agent
│   └── test_DQN.py              # Test DQN agent
│
├── compare_models.py              # Compare PPO and DQN performance
├── visualize_results.py          # Create visualization maps
├── create_training_gif.py        # Generate training progress GIFs
├── check_candidates_data.py      # Validate candidates data
├── fix_candidates_data.py        # Fix candidates data issues
├── setup_data_improvements.py    # Setup data improvements
│
├── RL_WMATA_Colab.ipynb          # Google Colab setup notebook
├── requirements_colab.txt        # Colab-specific requirements
│
├── README.md                      # This file
└──
```

**Data Directory Note**: The full data directory is available via Google Drive: https://drive.google.com/drive/folders/1GbpfGzb8V6_Uq_csaMjQKoKrV_giwuxY?usp=sharing

---

## Data Collection & Preparation

### Data Sources

The project integrates multiple real-world data sources:

1. **GTFS Transit Feeds**
   - Rail and bus GTFS feeds for WMATA system
   - Contains routes, stops, schedules, and travel times
   - Download using `scripts/download_gtfs.py`
   - Processed into network graphs using NetworkX

2. **WMATA API Data**
   - Station locations, entrances, and metadata
   - Bus routes and stops
   - Download using `scripts/download_wmata_api.py`
   - Provides existing infrastructure information

3. **LODES (LEHD Origin-Destination Employment Statistics)**
   - Worker flow data from U.S. Census Bureau
   - Origin-destination pairs with trip counts
   - Used for demand modeling
   - Stored in `data/lodes/dc_od.csv`

4. **Census Population Data**
   - Population by census block
   - Used for coverage calculations
   - Download using `scripts/download_census.py`
   - Stored in `data/census/dc_population.csv`

5. **DC Open Data**
   - DC boundary shapefile
   - Metro station entrance locations
   - Used for geographic constraints and existing station data

### Data Preparation Pipeline

1. **Candidate Station Generation** (`scripts/build_candidates.py`)
   - Generates candidate locations for new stations
   - Filters by population density and distance to existing stations
   - Creates GeoJSON and CSV formats
   - Attributes include: `candidate_id`, `lat`, `lon`, `population`
   - Output: `data/prepared/candidates_final.geojson` and `candidates_final.csv`

2. **Network Graph Construction** (`scripts/build_graph.py`)
   - Builds multi-modal network graph using NetworkX
   - Integrates GTFS routes, stops, and schedules
   - Includes travel times between nodes
   - Output: `data/prepared/network.pkl` and `multi_modal_graph.pkl`

3. **Origin-Destination Matrix** (`scripts/build_od.py`)
   - Creates OD demand matrix between candidate stations
   - Uses LODES data or synthetic generation
   - Includes hourly demand profiles for time-dependent modeling
   - Output: `data/prepared/station_od.csv`

4. **Candidate Enrichment** (`scripts/enrich_candidates_with_existing_stations.py`)
   - Adds distance to nearest existing station
   - Calculates population density metrics
   - Output: `data/prepared/candidates_enriched.csv`

5. **Network Enhancement** (`scripts/improve_network_with_existing_stations.py`)
   - Integrates existing WMATA stations into network graph
   - Creates connections to existing infrastructure
   - Output: `data/prepared/enhanced_network.pkl`

### Data Validation

Before training, validate your data using:
```bash
python check_candidates_data.py
```

If issues are found (e.g., duplicate coordinates), fix them with:
```bash
python fix_candidates_data.py
```

---

## Simulator Environment

### Station Placement Environment (`simulator/env.py`)

A Gymnasium-compatible environment for sequential station placement with the following characteristics:

**Observation Space:**
- `station_map`: Binary vector (N_candidates) indicating which stations are placed
- `placements_left`: Discrete count of remaining placements
- `action_mask`: Boolean mask indicating valid actions (prevents re-selecting placed stations)
- `distance_to_existing`: Optional normalized distance to nearest existing station (if enriched data available)

**Action Space:**
- Discrete selection from N_candidates (typically 86 candidate locations)

**Reward Function:**
The reward is a weighted combination of:
- **Coverage gain**: Increase in population covered within catchment radius (800m default)
- **Wait time decrease**: Reduction in average passenger wait time
- **Trips served**: Bonus for serving more origin-destination pairs
- **Placement bonus**: Small positive reward for each valid placement

Mathematically: `reward = α·coverage_gain + β·wait_decrease + γ·trips_gain + δ·placement_bonus`

**Key Features:**
- Realistic geographic coverage using Haversine distance calculations
- Integration with existing WMATA stations (102 existing stations)
- Time-dependent wait time calculations using hourly demand profiles
- Vectorized operations for fast computation
- Action masking to prevent invalid selections

**Metrics Calculated:**
- Coverage fraction: Percentage of population within catchment radius of any station
- Average wait time: Multi-factor model considering stations, demand, connectivity, and time of day
- Trips served: Total daily trips between placed stations

### Core Simulation (`simulator_core.py`)

Contains the core simulation logic that can be replaced with more detailed discrete-event simulations using SimPy or similar frameworks.

### Demand Sampling (`demand.py`)

Handles trip sampling from OD demand matrix for simulation. Supports weighted sampling based on trip frequencies.

---

## Baselines

Two baseline heuristic methods are implemented for comparison:

### Greedy Baseline (`baselines/greedy.py`)

Iteratively selects the candidate station with the highest population that hasn't been selected yet. Simple but effective for high-density areas.

**Usage:**
```bash
python test_baseline.py --method greedy
```

### K-Means Baseline (`baselines/kmeans.py`)

Uses K-Means clustering on candidate coordinates to identify M cluster centers, then selects the candidate closest to each center. Good for spatial distribution.

**Usage:**
```bash
python test_baseline.py --method kmeans
```

---

## Reinforcement Learning Agents

### Proximal Policy Optimization (PPO) (`agents/train_ppo.py`)

PPO is an on-policy actor-critic algorithm known for stability and sample efficiency.

**Key Features:**
- Uses `MultiInputPolicy` for dict observations
- Supports GPU acceleration
- TensorBoard logging for training monitoring
- Custom callbacks for metrics and visualization
- Optimized hyperparameters for station placement task

**Training:**
```bash
# Quick test (1000 timesteps, ~1 minute)
python -m agents.train_ppo --total_timesteps 1000

# Short training (100,000 timesteps, ~15-25 minutes on GPU)
python -m agents.train_ppo --total_timesteps 100000

# Full training (1,000,000 timesteps, ~2-3 hours on GPU)
python -m agents.train_ppo --total_timesteps 1000000
```

**Testing:**
```bash
python test_PPO.py
```

**Hyperparameters:**
- Learning rate: 3e-4
- Gamma (discount): 0.99
- Batch size: 128 (GPU) / 64 (CPU)
- N-steps: 128
- N-epochs: 4
- Network architecture: [128, 128] for both policy and value networks

### Deep Q-Network (DQN) (`agents/train_dqn.py`)

DQN is an off-policy value-based algorithm using experience replay.

**Key Features:**
- Uses `MultiInputPolicy` for dict observations
- Experience replay buffer
- Epsilon-greedy exploration
- GPU acceleration support
- TensorBoard logging

**Training:**
```bash
# Quick test (1000 timesteps, ~1 minute)
python -m agents.train_dqn --total_timesteps 1000

# Short training (100,000 timesteps, ~15-25 minutes on GPU)
python -m agents.train_dqn --total_timesteps 100000

# Full training (1,000,000 timesteps, ~2-3 hours on GPU)
python -m agents.train_dqn --total_timesteps 1000000
```

**Testing:**
```bash
python test_DQN.py
```

**Hyperparameters:**
- Learning rate: 1e-4
- Gamma (discount): 0.99
- Buffer size: 20,000 (GPU) / 10,000 (CPU)
- Batch size: 64 (GPU) / 32 (CPU)
- Learning starts: 1000
- Exploration: 1.0 → 0.05 (10% of training)
- Target update interval: 1000
- Network architecture: [128, 128]

---

## Installation & Setup

### Local Installation

1. **Create conda environment:**
```bash
conda create -n metro_rl python=3.12
conda activate metro_rl
```

2. **Install dependencies:**
```bash
conda install -c conda-forge pandas geopandas networkx gymnasium stable-baselines3 matplotlib tqdm tensorboard scikit-learn
pip install torch torchvision torchaudio
```

Or using pip:
```bash
pip install -r requirements_colab.txt
```

3. **Download data:**
   - Download data from Google Drive: https://drive.google.com/drive/folders/1GbpfGzb8V6_Uq_csaMjQKoKrV_giwuxY?usp=sharing
   - Extract to `data/` directory
   - Or run data download scripts (see Data Collection section)

4. **Validate data:**
```bash
python check_candidates_data.py
```

5. **Fix data if needed:**
```bash
python fix_candidates_data.py
python setup_data_improvements.py
```

### Google Colab Setup

For GPU-accelerated training, use the provided Colab notebook:

1. Open `RL_WMATA_Colab.ipynb` in Google Colab
2. Follow the notebook cells to:
   - Mount Google Drive with data
   - Install dependencies
   - Validate and fix data
   - Train models with GPU acceleration
   - Visualize results


---

## Usage Guide

### Quick Start

1. **Test the environment:**
```bash
python test_gym.py
```

2. **Run baseline methods:**
```bash
python test_baseline.py
```

3. **Train a model (choose one):**
```bash
# PPO
python -m agents.train_ppo --total_timesteps 100000

# DQN
python -m agents.train_dqn --total_timesteps 100000
```

4. **Test trained model:**
```bash
python test_PPO.py  # or test_DQN.py
```

5. **Visualize results:**
```bash
python visualize_results.py
python compare_models.py
```

6. **Monitor training (in separate terminal):**
```bash
tensorboard --logdir ./logs/
```
Then open http://localhost:6006

### Training Recommendations

**For Quick Testing:**
- 1,000-10,000 timesteps (~1-10 minutes)
- Good for debugging and verifying setup

**For Development:**
- 100,000 timesteps (~15-25 minutes on GPU)
- Good for algorithm development and hyperparameter tuning

**For Final Results:**
- 500,000-1,000,000 timesteps (~2-5 hours on GPU)
- Recommended for research papers and production use


### Evaluation

Compare models using:
```bash
python compare_models.py --n_episodes 10
```

This runs multiple evaluation episodes and provides:
- Average reward, coverage, wait time
- Best episode results
- Side-by-side comparison table
- Station location coordinates

---

## Visualization

### Station Placement Maps

Generate comparison visualizations:
```bash
python visualize_results.py
```

Outputs: `visualizations/station_placements.png`
- Shows all candidate locations
- Highlights selected stations (PPO in red, DQN in green)
- Displays performance metrics

### Training Progress

Create training progress GIFs:
```bash
python create_training_gif.py --model ppo
python create_training_gif.py --model dqn
python create_training_gif.py --model both
```

Outputs: `visualizations/ppo_training.gif` and `visualizations/dqn_training.gif`

### TensorBoard

Monitor training in real-time:
```bash
tensorboard --logdir ./logs/
```

Key metrics to watch:
- `rollout/ep_rew_mean`: Episode reward (should increase)
- `train/loss`: Training loss (should decrease)
- `env/coverage_fraction`: Population coverage (should increase)
- `env/avg_wait`: Average wait time (should decrease)

---

## Performance Optimizations

The project includes several performance optimizations:

1. **Vectorized Operations**: Geographic calculations use NumPy vectorization
2. **Pre-computed Distance Matrix**: Distances between candidates pre-computed at initialization
3. **GPU Acceleration**: Training automatically uses GPU when available
4. **Parallel Environments**: Uses multiple parallel environments for faster data collection
5. **Efficient Data Structures**: NetworkX graphs optimized for travel time queries


---

## Testing

### Unit Tests

Test individual components:
```bash
python test_sampler.py      # Test demand sampling
python test_gym.py          # Test environment
python test_baseline.py     # Test baseline methods
```

### Model Tests

Test trained models:
```bash
python test_PPO.py          # Test PPO agent
python test_DQN.py          # Test DQN agent
```

### Integration Tests

Run comprehensive comparisons:
```bash
python compare_models.py    # Compare all methods
```

---


## Key Metrics & Results

### Expected Performance

After training for 1,000,000 timesteps:

- **Coverage**: % of population within 800m catchment radius
- **Average Wait Time**: 4-6 minutes (down from 10 minutes baseline)
- **Trips Served**: 500,000+ trips per day
- **Episode Reward**: 500-600 (stable and consistent)

### Convergence Indicators

Good training shows:
-  Stable, increasing rewards
-  Consistent station selections
-  High coverage
-  Low wait times 
-  Decreasing reward variance

---


