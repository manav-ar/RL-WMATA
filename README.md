# Metro Station Placement Optimization Project

## Overview

This project implements a **Reinforcement Learning (RL) framework** for sequential placement of new metro stations in an urban transit network. The goal is to **maximize coverage** and **minimize average wait time** for passengers. The project integrates data collection, preparation, network modeling, demand sampling, and RL agents (PPO and DQN).

The project structure is modular and allows for experimentation with different RL algorithms, baseline heuristics, and network configurations.

---

data dir: https://drive.google.com/drive/folders/1GbpfGzb8V6_Uq_csaMjQKoKrV_giwuxY?usp=sharing

## Directory Structure

```






project/
│
├── simulator/
│   ├── env.py                # Gym Environment for sequential station placement
│   ├── simulator_core.py     # Core simulator: simulate(day, station_set) -> metrics
│   ├── demand.py             # Load and sample demand (LODES -> station OD)
│   ├── network.py            # Build network graph from GTFS / other sources
│
├── data_prep/
│   ├── gtfs_download.py      # Download GTFS transit feeds
│   ├── build_candidates.py   # Generate candidate station locations
│   ├── lodes_map_to_candidates.py # Map LODES demand data to candidate locations
│
├── baselines/
│   ├── greedy.py             # Greedy placement baseline
│   ├── kmeans.py             # KMeans placement baseline
│
├── agents/
│   ├── train_ppo.py          # Train PPO agent
│   ├── train_dqn.py          # Train DQN agent (TBD)
│   ├── gnn_policy.py         # GNN-based policy network (TBD)
│

├── data/
│   ├── raw/                  # Raw downloaded data (GTFS, LODES)
│   ├── prepared/             # Preprocessed data (networks, candidate CSV/GeoJSON, OD)
│
├── logs/                     # Tensorboard / training logs
├── test_*.py                 # Test scripts for sampler, env, baselines, PPO
├── README.md                 # This file

````

---

## Data Collection & Preparation

1. **GTFS feeds**  
   - Download using `data_prep/gtfs_download.py`.  
   - Outputs transit routes, stops, and schedules.

2. **Candidate Station Generation**  
   - `data_prep/build_candidates.py` generates candidate locations for new stations.  
   - Attributes include: `candidate_id`, `lat`, `lon`, `population`.

3. **LODES Demand Mapping**  
   - `data_prep/lodes_map_to_candidates.py` maps OD demand from LODES data to candidate locations.  
   - Generates `station_od.csv` with trip counts between stations.

4. **Network Graph Construction**  
   - `simulator/network.py` builds the network graph using NetworkX.  
   - The graph is saved as a `.pkl` file (`network.pkl` or `multimodal.pkl`).

5. **Candidate GeoJSON Conversion**  
   - If candidates are in GeoJSON, `env.py` automatically converts to CSV.

---

## Simulator (`simulator/`)

- **Env (`env.py`)**: Gym-compatible environment for sequential placement.  
  - Observation space: dict of `station_map` (vector) and `placements_left`.  
  - Action space: discrete selection of candidate stations.  
  - Reward: combination of **coverage gain** and **average wait time penalty**.  
  - Handles both PPO and DQN multi-input observations.

- **Core Simulation (`simulator_core.py`)**:  
  - Contains `simulate()` function to evaluate station placements.  
  - Can be replaced with more realistic simulations using SimPy.

- **Demand (`demand.py`)**:  
  - Samples trips for simulation.  
  - Handles LODES OD data and maps to stations.

---

## Baselines (`baselines/`)

- **Greedy**: Selects stations with highest coverage improvement iteratively.  
- **KMeans**: Uses clustering on demand points to select candidate locations.  

**Scripts updated to handle GeoJSON → CSV conversion automatically.**

---

## RL Agents (`agents/`)

### PPO (`train_ppo.py`)

- Trains PPO agent on `StationPlacementEnv`.
- Uses `MultiInputPolicy` due to dict observations.
- Supports TensorBoard logging:

```bash
tensorboard --logdir ./logs/ppo_tensorboard/
````

* Example training call:

```bash
python -m agents.train_ppo
```

* After training, test with:

```bash
python test_PPO.py
```

### DQN (`train_dqn.py`)

* script to be created for DQN training.

---

## Testing Scripts

* `test_sampler.py`: Verify demand sampling.
* `test_gym.py`: Test environment reset, step, and reward logic.
* `test_baseline.py`: Run greedy/kmeans baselines.
* `test_PPO.py`: Run trained PPO agent.
* `test_DQN.py`: TBD for DQN agent testing.

---

## To-Do / Next Steps

1. **RL Training**

   * Train PPO to **optimality** (increase timesteps from 10k → 500k+).
   * Write and Train DQN agent on the same environment.
   * Experiment with hyperparameters for both agents (`learning_rate`, `gamma`, `batch_size`, etc.).

2. **Evaluation & Visualization**

   * Compare RL agents to baselines (greedy, kmeans).
   * Visualize placements, coverage fraction, and average wait.
   * Create plots of reward progression over episodes using TensorBoard or matplotlib.

3. **Environment & Simulator Improvements**

   * Replace placeholder `_simulate()` with a **realistic simulation** using SimPy or discrete event modeling.
   * Consider multi-day simulation or stochastic demand.

4. **Advanced Agents**

   * Implement GNN-based policy in `gnn_policy.py`.
   * Test on multi-modal network graphs.

5. **Documentation & Reproducibility**

   * Finalize README with clear instructions (this file).
   * Include example `.env` or configuration files for easy reproduction.

---

## Installation & Setup

1. **Create conda environment**:

```bash
conda create -n metro_rl python=3.12
conda activate metro_rl
```

2. **Install dependencies**:

```bash
conda install -c conda-forge pandas geopandas networkx gymnasium stable-baselines3 matplotlib tqdm tensorboard
```

3. **Run tests**:

```bash
python test_sampler.py
python test_gym.py
python test_baseline.py
python test_PPO.py
```

4. **Train agents**:

```bash
python -m agents.train_ppo
python -m agents.train_dqn
```

5. **TensorBoard for monitoring**:

```bash
tensorboard --logdir ./logs/
```

---

## Notes

* All scripts handle GeoJSON → CSV conversion automatically.
* The environment is compatible with **dict observations**, required for `MultiInputPolicy`.
* PPO and DQN currently use simplified simulation; real simulation can improve realism.
* Logging is set up for **TensorBoard** to monitor reward progression, exploration, and convergence.

---
**Project Status:**
✅ Basic environment implemented
✅ Candidate generation & GeoJSON handling
⚠️ PPO agent written to be trained
⚠️ DQN agent to be written and trained
⚠️ Visualization and optimal RL evaluation pending

---
