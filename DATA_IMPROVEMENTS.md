# Data-Driven Improvements Summary

## Overview
This document summarizes comprehensive improvements made to better utilize all available data in the RL-WMATA project.

## Available Data Inventory

### 1. **GTFS Data** ✅
- **Rail GTFS**: `data/gtfs/rail/` - Complete rail transit feed
  - `stops.txt` - Station locations
  - `stop_times.txt` - Scheduled arrival/departure times
  - `trips.txt` - Trip information
  - `routes.txt` - Route definitions
- **Bus GTFS**: `data/gtfs/bus/` - Complete bus transit feed
  - Similar structure to rail GTFS

### 2. **Existing Metro Stations** ✅
- **WMATA API Data**: `data/wmata/stations.json` - All WMATA stations with codes, names, coordinates
- **Station Entrances**: `data/dc_open/Metro_Station_Entrances_Regional.geojson` - Detailed entrance locations

### 3. **Origin-Destination Data** ✅
- **LODES Data**: `data/lodes/dc_od.csv` - Real worker flow data
- **Synthetic OD**: `data/prepared/station_od.csv` - Currently used, has hourly profiles

### 4. **Census/Population Data** ✅
- **Population**: `data/census/dc_population.csv` - Population by census block
- **Candidates**: `data/prepared/candidates_final.geojson` - Candidate locations with population

### 5. **Network Data** ✅
- **Multi-modal Graph**: `data/prepared/multi_modal_graph.pkl` - Current network
- **Travel Time Graph**: `data/prepared/network.pkl` - GTFS-based travel times

## Improvements Implemented

### 1. **Hourly Profile Integration** ✅
**What**: Use time-dependent demand patterns from `hourly_profile` column

**Implementation**:
- Parse hourly_profile data from station_od.csv
- Calculate peak hour demand ratios
- Adjust wait times based on time of day (peak hours have higher wait)

**Location**: `simulator/env.py::_calculate_avg_wait_time()`

**Impact**: More realistic wait time calculations that account for rush hour patterns

### 2. **Existing Station Integration** ✅
**What**: Include existing WMATA metro stations in coverage calculations

**Implementation**:
- Load existing metro station entrances
- Calculate coverage from both existing and new stations
- Avoid double-counting population already covered by existing stations

**Location**: `simulator/env.py::_calculate_coverage()`

**Impact**: More realistic coverage calculations that account for current infrastructure

### 3. **Enhanced Candidate Features** ✅
**What**: Add distance to existing stations as observation feature

**Implementation**:
- Script: `scripts/enrich_candidates_with_existing_stations.py`
- Calculates distance from each candidate to nearest existing metro station
- Adds to observation space for RL agent

**Location**: 
- Data prep: `scripts/enrich_candidates_with_existing_stations.py`
- Environment: `simulator/env.py::_get_obs()`

**Impact**: Agent can learn to place stations considering proximity to existing infrastructure

### 4. **Enhanced Network Graph** ✅
**What**: Connect candidates to existing metro stations in network

**Implementation**:
- Script: `scripts/improve_network_with_existing_stations.py`
- Adds edges between candidates and existing stations within 2km
- Uses realistic travel times based on distance

**Location**: `scripts/improve_network_with_existing_stations.py`

**Impact**: Better connectivity calculations, more realistic travel time estimates

### 5. **Enriched Observation Space** ✅
**What**: Add additional features to help agent make better decisions

**Features Added**:
- `distance_to_existing`: Normalized distance to nearest existing station (0-1)
- Future: population density, demand intensity, etc.

**Location**: `simulator/env.py::observation_space` and `_get_obs()`

**Impact**: Agent has more information to make informed placement decisions

## Scripts Created

### 1. `scripts/enrich_candidates_with_existing_stations.py`
- Enriches candidate data with distance to existing stations
- Outputs: `data/prepared/candidates_enriched.geojson` and `.csv`

**Usage**:
```bash
python scripts/enrich_candidates_with_existing_stations.py
```

### 2. `scripts/improve_network_with_existing_stations.py`
- Enhances network graph by connecting candidates to existing stations
- Outputs: `data/prepared/enhanced_network.pkl`

**Usage**:
```bash
python scripts/improve_network_with_existing_stations.py
```

## Future Improvements (Not Yet Implemented)

### 1. **Real LODES OD Data** ⏳
**Status**: Data available but not integrated

**What to do**:
- Map LODES census block data to candidate locations
- Replace synthetic OD with real worker flow patterns
- More accurate demand modeling

**Files**: `data/lodes/dc_od.csv`

### 2. **GTFS Route-Based Connections** ⏳
**Status**: GTFS data available but not fully utilized

**What to do**:
- Use actual GTFS routes to connect stations
- Calculate real travel times from `stop_times.txt`
- Model service frequency and headways

**Files**: `data/gtfs/rail/stop_times.txt`, `data/gtfs/rail/trips.txt`

### 3. **Service Frequency Calculations** ⏳
**Status**: Can be derived from GTFS but not implemented

**What to do**:
- Calculate headway (time between trains) from GTFS
- Use headway to inform wait time calculations
- Model peak vs. off-peak service frequencies

**Files**: `data/gtfs/rail/stop_times.txt`

### 4. **Multi-Modal Connectivity** ⏳
**Status**: Bus data available but not integrated

**What to do**:
- Include bus routes in network graph
- Model bus-to-rail transfers
- Calculate multi-modal travel times

**Files**: `data/gtfs/bus/`, `data/wmata/bus_stops.json`

## Usage Instructions

### Step 1: Enrich Candidates
```bash
python scripts/enrich_candidates_with_existing_stations.py
```

### Step 2: Enhance Network
```bash
python scripts/improve_network_with_existing_stations.py
```

### Step 3: Update Environment to Use Enriched Data
The environment automatically detects enriched candidates if available. To use:
- Ensure `candidates_enriched.csv` exists
- Environment will automatically load enriched features

### Step 4: Retrain Models
Since observation space changed, retrain models:
```bash
python -m agents.train_ppo --total_timesteps 100000
python -m agents.train_dqn --total_timesteps 100000
```

## Data Flow

```
Raw Data
├── GTFS (rail/bus)
├── WMATA API (stations, bus stops)
├── LODES (OD flows)
└── Census (population)

    ↓

Data Preparation Scripts
├── enrich_candidates_with_existing_stations.py
├── improve_network_with_existing_stations.py
└── build_od.py (synthetic) / [future: use LODES]

    ↓

Prepared Data
├── candidates_enriched.csv (with distance to existing)
├── enhanced_network.pkl (with existing station connections)
└── station_od.csv (with hourly profiles)

    ↓

Environment
├── Uses enriched candidates
├── Uses enhanced network
├── Uses hourly profiles for wait time
└── Includes existing stations in coverage

    ↓

RL Training
└── Agent learns with richer observations
```

## Benefits

1. **More Realistic Coverage**: Accounts for existing infrastructure
2. **Time-Dependent Metrics**: Uses hourly demand patterns
3. **Better Connectivity**: Considers connections to existing network
4. **Richer Observations**: Agent has more information to learn from
5. **Foundation for Future**: Easy to add more data sources

## Testing

After running data preparation scripts, test the environment:
```bash
python test_gym.py
```

Check that:
- Enriched features are loaded (if available)
- Coverage includes existing stations
- Wait times vary with hourly profiles

## Notes

- All improvements are backward compatible
- If enriched data doesn't exist, environment falls back to basic features
- Existing models may need retraining due to observation space changes
- Hourly profile parsing handles both string and list formats

