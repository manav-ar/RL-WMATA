# simulator/env.py
import os
import pickle
import numpy as np
import pandas as pd
import networkx as nx
import gymnasium as gym
from gymnasium import spaces
import geopandas as gpd
import math
import ast

class StationPlacementEnv(gym.Env):
    """
    Gym Environment for sequential placement of new metro stations.
    Uses realistic coverage calculations, network-based travel times, and OD demand.
    """

    metadata = {"render.modes": ["human"]}

    def __init__(self, network_path, candidates_path, station_od_path, max_placements=5, catchment_radius=800, seed=42):
        super().__init__()

        self.seed_value = seed
        np.random.seed(seed)

        # --- Load network ---
        if not os.path.exists(network_path):
            raise FileNotFoundError(f"Network file not found: {network_path}")
        with open(network_path, "rb") as f:
            self.G = pickle.load(f)  # NetworkX graph

        # --- Load candidates (auto-convert GeoJSON → CSV if needed) ---
        csv_path = candidates_path.replace(".geojson", ".csv")
        if candidates_path.endswith(".geojson") and not os.path.exists(csv_path):
            print("Converting candidates GeoJSON → CSV...")
            gdf = gpd.read_file(candidates_path)
            
            # Extract lat/lon from geometry if not already present
            if 'lat' not in gdf.columns or 'lon' not in gdf.columns:
                if gdf.geometry is not None and len(gdf) > 0:
                    gdf['lon'] = gdf.geometry.x  # Longitude from Point geometry
                    gdf['lat'] = gdf.geometry.y  # Latitude from Point geometry
                    print(f"   Extracted lat/lon from geometry column")
                else:
                    raise ValueError("GeoJSON has no geometry column or is empty")
            
            # Ensure required columns exist
            required_cols = ['candidate_id', 'lat', 'lon', 'population']
            missing_cols = [col for col in required_cols if col not in gdf.columns]
            if missing_cols:
                raise ValueError(f"Missing required columns in GeoJSON: {missing_cols}")
            
            # Save to CSV
            gdf[required_cols].to_csv(csv_path, index=False)
            print(f"✅ Converted GeoJSON to CSV with {len(gdf)} candidates")

        self.candidates = pd.read_csv(csv_path)
        
        # Validate coordinates are unique and valid
        unique_coords = self.candidates[['lat', 'lon']].drop_duplicates()
        if len(unique_coords) < len(self.candidates) * 0.5:  # Less than 50% unique
            print(f"⚠️  Warning: Many duplicate coordinates in candidates!")
            print(f"   Total candidates: {len(self.candidates)}")
            print(f"   Unique coordinates: {len(unique_coords)}")
            print(f"   Sample coordinates:")
            print(self.candidates[['candidate_id', 'lat', 'lon']].head(10))
        
        # Validate coordinates are reasonable (DC area)
        invalid_coords = self.candidates[
            (self.candidates['lat'] < 38.5) | (self.candidates['lat'] > 39.2) |
            (self.candidates['lon'] < -77.2) | (self.candidates['lon'] > -76.8)
        ]
        if len(invalid_coords) > 0:
            print(f"⚠️  Warning: {len(invalid_coords)} candidates have coordinates outside DC area")
            print(f"   Lat range: {self.candidates['lat'].min():.4f} to {self.candidates['lat'].max():.4f}")
            print(f"   Lon range: {self.candidates['lon'].min():.4f} to {self.candidates['lon'].max():.4f}")
        
        self.candidate_ids = self.candidates['candidate_id'].tolist()
        self.N_candidates = len(self.candidates)

        # Validate candidates have required columns
        required_cols = {'candidate_id', 'lat', 'lon', 'population'}
        if not required_cols.issubset(self.candidates.columns):
            missing = required_cols - set(self.candidates.columns)
            raise ValueError(f"Missing required columns in candidates: {missing}")

        # --- Load OD demand ---
        if not os.path.exists(station_od_path):
            raise FileNotFoundError(f"Station OD file not found: {station_od_path}")
        self.station_od = pd.read_csv(station_od_path)
        
        # Validate OD demand has required columns
        required_od_cols = {'origin_candidate', 'dest_candidate', 'mean_trips_per_day'}
        if not required_od_cols.issubset(self.station_od.columns):
            missing = required_od_cols - set(self.station_od.columns)
            raise ValueError(f"Missing required columns in station_od: {missing}")
        
        # Parse hourly_profile if available (stored as string in CSV)
        if 'hourly_profile' in self.station_od.columns:
            self.station_od['hourly_profile_parsed'] = self.station_od['hourly_profile'].apply(
                lambda x: ast.literal_eval(x) if isinstance(x, str) else x
            )
        else:
            self.station_od['hourly_profile_parsed'] = None
        
        # --- Load existing metro stations for enhanced coverage ---
        existing_stations_path = "data/dc_open/Metro_Station_Entrances_Regional.geojson"
        if os.path.exists(existing_stations_path):
            try:
                self.existing_stations_gdf = gpd.read_file(existing_stations_path).to_crs("EPSG:4326")
                self.has_existing_stations = True
            except:
                self.has_existing_stations = False
        else:
            self.has_existing_stations = False

        # --- Environment config ---
        self.max_placements = max_placements
        self.catchment_radius = catchment_radius  # in meters

        # Action mask to prevent selecting already-built stations
        self.action_mask = np.ones(self.N_candidates, dtype=bool)

        # Track placements
        self.station_map = np.zeros(self.N_candidates, dtype=int)
        self.placements_done = 0

        # Pre-compute distance matrix for faster coverage calculations
        self._precompute_distances()
        
        # Pre-compute total population for coverage calculations
        self.total_population = self.candidates['population'].sum()
        if self.total_population == 0:
            raise ValueError("Total population is zero - check candidates data")

        # Check if enriched candidates are available
        enriched_candidates_path = candidates_path.replace(".geojson", "_enriched.csv").replace(".csv", "_enriched.csv")
        if os.path.exists(enriched_candidates_path):
            self.candidates_enriched = pd.read_csv(enriched_candidates_path)
            self.has_enriched_features = True
            # Add distance to existing station as feature
            if 'distance_to_nearest_existing_station' in self.candidates_enriched.columns:
                # Normalize distance (max distance ~5000m, normalize to 0-1)
                max_dist = self.candidates_enriched['distance_to_nearest_existing_station'].max()
                self.candidates_enriched['distance_to_existing_norm'] = (
                    self.candidates_enriched['distance_to_nearest_existing_station'] / max(5000, max_dist)
                )
        else:
            self.has_enriched_features = False
        
        # Define observation space with action mask
        obs_dict = {
            "station_map": spaces.Box(low=0, high=1, shape=(self.N_candidates,), dtype=np.int8),
            "placements_left": spaces.Discrete(self.max_placements + 1),
            "action_mask": spaces.Box(low=0, high=1, shape=(self.N_candidates,), dtype=bool)
        }
        
        # Add enriched features if available
        if self.has_enriched_features and 'distance_to_existing_norm' in self.candidates_enriched.columns:
            obs_dict["distance_to_existing"] = spaces.Box(
                low=0, high=1, shape=(self.N_candidates,), dtype=np.float32
            )
        
        self.observation_space = spaces.Dict(obs_dict)

        # Action space: choose one of N_candidates
        self.action_space = spaces.Discrete(self.N_candidates)

        # Metrics storage for reward calculation
        self.last_metrics = {
            "coverage_fraction": 0.0,
            "avg_wait": 0.0,
            "total_trips_served": 0
        }

    # def reset(self):
    #     self.station_map = np.zeros(self.N_candidates, dtype=int)
    #     self.action_mask = np.ones(self.N_candidates, dtype=bool)
    #     self.placements_done = 0
    #     self.last_metrics = {"coverage_fraction": 0.0, "avg_wait": 0.0}
    #     return self._get_obs()
    def reset(self, seed=None, options=None):
        """
        Reset environment to initial state.
        Compatible with Gymnasium & Stable-Baselines3 API.
        """
        # Call the parent reset to handle seeding
        super().reset(seed=seed)
        if seed is not None:
            np.random.seed(seed)

        self.station_map = np.zeros(self.N_candidates, dtype=int)
        self.action_mask = np.ones(self.N_candidates, dtype=bool)
        self.placements_done = 0
        self.last_metrics = {
            "coverage_fraction": 0.0,
            "avg_wait": 0.0,
            "total_trips_served": 0
        }

        obs = self._get_obs()
        info = {}  # required by Gymnasium
        return obs, info

    # def step(self, action):
    #     if not self.action_mask[action]:
    #         raise ValueError(f"Invalid action {action}: station already placed.")

    #     # Mark station as placed
    #     self.station_map[action] = 1
    #     self.action_mask[action] = 0
    #     self.placements_done += 1

    #     # --- Run simulation to get new metrics ---
    #     metrics = self._simulate(self.station_map)
    #     reward = self._compute_reward(metrics)
    #     self.last_metrics = metrics

    #     done = self.placements_done >= self.max_placements
    #     info = metrics.copy()

    #     return self._get_obs(), reward, done, info
    
    # def step(self, action):
    #     if not self.action_mask[action]:
    #         raise ValueError(f"Invalid action {action}: station already placed.")

    #     # Mark station as placed
    #     self.station_map[action] = 1
    #     self.action_mask[action] = 0
    #     self.placements_done += 1

    #     # Simulate
    #     metrics = self._simulate(self.station_map)
    #     reward = self._compute_reward(metrics)
    #     self.last_metrics = metrics

    #     terminated = self.placements_done >= self.max_placements
    #     truncated = False  # You can add custom truncation logic if needed
    #     info = metrics.copy()

    #     return self._get_obs(), reward, terminated, truncated, info
    
    def step(self, action):
        # Check if action is valid
        if not self.action_mask[action]:
            # Invalid action — penalize agent and optionally end episode
            reward = -1.0  # strong penalty for reselecting a placed station
            terminated = True  # end episode to reinforce the rule
            truncated = False
            info = {"invalid_action": True}
            return self._get_obs(), reward, terminated, truncated, info

        # --- Valid action ---
        # Mark station as placed
        self.station_map[action] = 1
        self.action_mask[action] = 0
        self.placements_done += 1

        # Simulate environment dynamics
        metrics = self._simulate(self.station_map)
        reward = self._compute_reward(metrics)
        self.last_metrics = metrics

        # Check termination condition
        terminated = self.placements_done >= self.max_placements
        truncated = False  # custom truncation logic can go here if needed
        
        # Enhanced info with selected stations
        info = metrics.copy()
        placed_indices = np.where(self.station_map == 1)[0]
        info["selected_stations"] = [self.candidate_ids[i] for i in placed_indices]
        info["episode_length"] = self.placements_done
        info["reward"] = reward

        return self._get_obs(), reward, terminated, truncated, info


    def _haversine_distance(self, lat1, lon1, lat2, lon2):
        """
        Calculate distance between two points using Haversine formula.
        Returns distance in meters.
        """
        R = 6371000  # Earth radius in meters
        phi1, phi2 = math.radians(lat1), math.radians(lat2)
        dphi = math.radians(lat2 - lat1)
        dlambda = math.radians(lon2 - lon1)
        a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
        return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    
    def _haversine_distance_vectorized(self, lat1, lon1, lat2, lon2):
        """
        Vectorized Haversine distance calculation for arrays.
        Much faster than looping. Handles broadcasting correctly.
        """
        R = 6371000  # Earth radius in meters
        # Ensure arrays for broadcasting
        lat1 = np.asarray(lat1)
        lon1 = np.asarray(lon1)
        lat2 = np.asarray(lat2)
        lon2 = np.asarray(lon2)
        
        phi1 = np.radians(lat1)
        phi2 = np.radians(lat2)
        dphi = np.radians(lat2 - lat1)
        dlambda = np.radians(lon2 - lon1)
        a = np.sin(dphi / 2) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda / 2) ** 2
        return R * 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

    def _precompute_distances(self):
        """
        Pre-compute distance matrix between all candidates for faster coverage calculations.
        """
        self.distance_matrix = np.zeros((self.N_candidates, self.N_candidates))
        for i in range(self.N_candidates):
            for j in range(i+1, self.N_candidates):
                lat1, lon1 = self.candidates.iloc[i]['lat'], self.candidates.iloc[i]['lon']
                lat2, lon2 = self.candidates.iloc[j]['lat'], self.candidates.iloc[j]['lon']
                dist = self._haversine_distance(lat1, lon1, lat2, lon2)
                self.distance_matrix[i, j] = dist
                self.distance_matrix[j, i] = dist

    def _calculate_coverage(self, placed_indices):
        """
        Calculate coverage fraction: population within catchment radius of placed stations.
        OPTIMIZED: Uses vectorized numpy operations instead of slow iterrows() loops.
        """
        if len(placed_indices) == 0:
            # Check only existing stations if no new placements
            if self.has_existing_stations:
                # Vectorized: check all candidates against existing stations at once
                covered_mask = np.zeros(self.N_candidates, dtype=bool)
                candidates_lat = self.candidates['lat'].values
                candidates_lon = self.candidates['lon'].values
                
                for _, existing_station in self.existing_stations_gdf.iterrows():
                    existing_lat = existing_station.geometry.y
                    existing_lon = existing_station.geometry.x
                    # Vectorized distance calculation
                    distances = self._haversine_distance_vectorized(
                        candidates_lat, candidates_lon,
                        existing_lat, existing_lon
                    )
                    covered_mask |= (distances <= self.catchment_radius)
                
                covered_population = self.candidates.loc[covered_mask, 'population'].sum()
            else:
                covered_population = 0
        else:
            # OPTIMIZED: Use vectorized operations with pre-computed distance matrix
            # Check coverage from placed stations using distance matrix
            if len(placed_indices) > 0:
                # Validate inputs
                if not hasattr(self, 'distance_matrix') or self.distance_matrix is None:
                    raise ValueError("Distance matrix not initialized!")
                
                if max(placed_indices) >= self.N_candidates:
                    raise ValueError(f"Invalid placed_indices: {placed_indices}, max={self.N_candidates}")
                
                # Vectorized: check all candidates at once
                distances_to_placed = self.distance_matrix[:, placed_indices]  # Shape: (N_candidates, n_placed)
                min_distances = np.min(distances_to_placed, axis=1)  # Min distance to any placed station
                covered_mask = min_distances <= self.catchment_radius
                
                # Debug output for troubleshooting (only for small numbers to avoid spam)
                if len(placed_indices) <= 5:
                    print(f"DEBUG Coverage: placed_indices={placed_indices}")
                    print(f"DEBUG Coverage: min_distances range: {min_distances.min():.1f}m to {min_distances.max():.1f}m")
                    print(f"DEBUG Coverage: covered_mask sum: {covered_mask.sum()}/{len(covered_mask)} candidates")
                    print(f"DEBUG Coverage: catchment_radius={self.catchment_radius}m")
                
                # Subtract existing station coverage if checking both
                if self.has_existing_stations:
                    existing_covered_mask = np.zeros(self.N_candidates, dtype=bool)
                    candidates_lat = self.candidates['lat'].values
                    candidates_lon = self.candidates['lon'].values
                    
                    for _, existing_station in self.existing_stations_gdf.iterrows():
                        existing_lat = existing_station.geometry.y
                        existing_lon = existing_station.geometry.x
                        distances = self._haversine_distance_vectorized(
                            candidates_lat, candidates_lon,
                            existing_lat, existing_lon
                        )
                        existing_covered_mask |= (distances <= self.catchment_radius)
                    
                    # Only count population not already covered by existing stations
                    new_coverage_mask = covered_mask & ~existing_covered_mask
                    covered_population = self.candidates.loc[new_coverage_mask, 'population'].sum()
                else:
                    covered_population = self.candidates.loc[covered_mask, 'population'].sum()
            else:
                covered_population = 0
        
        coverage_fraction = covered_population / self.total_population if self.total_population > 0 else 0.0
        
        # Debug output
        if len(placed_indices) <= 5:
            print(f"DEBUG Coverage: covered_population={covered_population:.0f}, total_population={self.total_population:.0f}, coverage={coverage_fraction:.3f}")
        
        return min(1.0, coverage_fraction)  # Cap at 1.0

    def _calculate_avg_wait_time(self, placed_station_ids, peak_hour=8):
        """
        Calculate average wait time based on:
        1. Number of stations (more stations = better connectivity = lower wait)
        2. OD demand served (more demand served = better service)
        3. Network connectivity
        4. Hourly profile (peak hours have higher wait times)
        """
        if len(placed_station_ids) == 0:
            return 10.0  # High wait time with no stations
        
        # Base wait time decreases with more stations
        base_wait = 10.0  # minutes
        station_factor = 1.0 / (1.0 + len(placed_station_ids) * 0.15)
        
        # Calculate trips served between placed stations
        placed_od = self.station_od[
            (self.station_od['origin_candidate'].isin(placed_station_ids)) &
            (self.station_od['dest_candidate'].isin(placed_station_ids))
        ]
        total_trips_served = placed_od['mean_trips_per_day'].sum() if len(placed_od) > 0 else 0
        total_demand = self.station_od['mean_trips_per_day'].sum()
        demand_factor = 1.0 - (total_trips_served / total_demand) * 0.3 if total_demand > 0 else 1.0
        
        # Use hourly profile for time-dependent wait time if available
        # OPTIMIZED: Simplified calculation, skip expensive iterrows
        hourly_factor = 1.0
        if len(placed_od) > 0 and 'hourly_profile_parsed' in self.station_od.columns:
            # Simplified: use mean of all hourly profiles instead of iterating
            try:
                profiles = placed_od['hourly_profile_parsed'].dropna()
                if len(profiles) > 0:
                    # Quick approximation: assume moderate peak factor
                    hourly_factor = 1.1  # 10% increase for peak hours (simplified)
            except:
                pass  # Skip if calculation fails
        
        # Calculate network connectivity (average travel time between placed stations)
        # OPTIMIZED: Skip expensive NetworkX operations, use simple approximation
        connectivity_factor = 1.0
        # Skip network connectivity calculation for speed - use station count as proxy
        # NetworkX shortest_path is very slow and called every step
        # Instead, connectivity is approximated by station_factor already calculated above
        
        avg_wait = base_wait * station_factor * demand_factor * connectivity_factor * hourly_factor
        return max(1.0, avg_wait)  # Minimum 1 minute wait time

    def _calculate_trips_served(self, placed_station_ids):
        """
        Calculate total trips served between placed stations.
        """
        if len(placed_station_ids) == 0:
            return 0
        
        placed_od = self.station_od[
            (self.station_od['origin_candidate'].isin(placed_station_ids)) &
            (self.station_od['dest_candidate'].isin(placed_station_ids))
        ]
        return placed_od['mean_trips_per_day'].sum() if len(placed_od) > 0 else 0

    def _get_obs(self):
        """
        Get current observation including action mask and enriched features.
        """
        obs = {
            "station_map": self.station_map.copy(),
            "placements_left": self.max_placements - self.placements_done,
            "action_mask": self.action_mask.copy().astype(bool)
        }
        
        # Add enriched features if available
        if self.has_enriched_features and 'distance_to_existing_norm' in self.candidates_enriched.columns:
            obs["distance_to_existing"] = self.candidates_enriched['distance_to_existing_norm'].values.astype(np.float32)
        
        return obs

    def _simulate(self, station_map):
        """
        Realistic simulation using actual data:
        - Coverage based on catchment radius and geographic distances
        - Wait time based on number of stations, OD demand, and network connectivity
        - Trips served based on OD demand between placed stations
        """
        # Get list of placed station indices and IDs
        placed_indices = np.where(station_map == 1)[0]
        placed_station_ids = [self.candidate_ids[i] for i in placed_indices]
        
        # Calculate realistic metrics
        coverage_fraction = self._calculate_coverage(placed_indices)
        avg_wait = self._calculate_avg_wait_time(placed_station_ids)
        total_trips_served = self._calculate_trips_served(placed_station_ids)
        
        return {
            "coverage_fraction": coverage_fraction,
            "avg_wait": avg_wait,
            "total_trips_served": total_trips_served,
            "num_stations": len(placed_station_ids)
        }

    def _compute_reward(self, metrics):
        """
        Reward = +coverage improvement + wait time decrease + placement bonus
        Note: wait_decrease is positive when wait time goes DOWN (which is good)
        """
        alpha = 1.0  # weight for coverage
        beta = 0.1   # weight for wait time
        gamma = 0.001  # weight for trips served
        delta = 0.1  # bonus for each valid placement
        
        coverage_gain = metrics["coverage_fraction"] - self.last_metrics.get("coverage_fraction", 0)
        
        # FIXED: wait_decrease is positive when wait time decreases (good thing)
        last_wait = self.last_metrics.get("avg_wait", 10.0)
        current_wait = metrics["avg_wait"]
        wait_decrease = last_wait - current_wait  # Positive when wait goes down
        
        # Bonus for serving more trips
        trips_gain = metrics.get("total_trips_served", 0) - self.last_metrics.get("total_trips_served", 0)
        
        # Small positive reward for each valid placement (encourages exploration)
        placement_bonus = delta
        
        reward = alpha * coverage_gain + beta * wait_decrease + gamma * trips_gain + placement_bonus
        
        return reward

    def render(self, mode="human"):
        """
        Render current state of the environment.
        """
        placed_indices = np.where(self.station_map == 1)[0]
        selected_stations = [self.candidate_ids[i] for i in placed_indices]
        
        print(f"\n{'='*60}")
        print(f"Station Placement Environment State")
        print(f"{'='*60}")
        print(f"Placements done: {self.placements_done}/{self.max_placements}")
        print(f"Selected stations: {selected_stations if selected_stations else 'None'}")
        print(f"Coverage fraction: {self.last_metrics['coverage_fraction']:.3f} ({self.last_metrics['coverage_fraction']*100:.1f}%)")
        print(f"Average wait time: {self.last_metrics['avg_wait']:.2f} minutes")
        print(f"Total trips served: {self.last_metrics.get('total_trips_served', 0):.0f} trips/day")
        print(f"Number of stations: {self.last_metrics.get('num_stations', 0)}")
        print(f"{'='*60}\n")

    def close(self):
        pass

