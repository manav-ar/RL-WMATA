# simulator/env.py
import os
import pickle
import numpy as np
import pandas as pd
import networkx as nx
import gymnasium as gym
from gymnasium import spaces
import geopandas as gpd

class StationPlacementEnv(gym.Env):
    """
    Gym Environment for sequential placement of new metro stations.
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
        if candidates_path.endswith(".geojson") and not os.path.exists(candidates_path.replace(".geojson", ".csv")):
            print("Converting candidates GeoJSON → CSV...")
            gdf = gpd.read_file(candidates_path)
            gdf[['candidate_id', 'lat', 'lon', 'population']].to_csv(
                candidates_path.replace(".geojson", ".csv"), index=False
            )

        self.candidates = pd.read_csv(candidates_path.replace(".geojson", ".csv"))
        self.candidate_ids = self.candidates['candidate_id'].tolist()
        self.N_candidates = len(self.candidates)

        # --- Load OD demand ---
        if not os.path.exists(station_od_path):
            raise FileNotFoundError(f"Station OD file not found: {station_od_path}")
        self.station_od = pd.read_csv(station_od_path)

        # --- Environment config ---
        self.max_placements = max_placements
        self.catchment_radius = catchment_radius

        # Action mask to prevent selecting already-built stations
        self.action_mask = np.ones(self.N_candidates, dtype=bool)

        # Track placements
        self.station_map = np.zeros(self.N_candidates, dtype=int)
        self.placements_done = 0

        # Define observation space: vector of length N_candidates + 1 (placements left)
        self.observation_space = spaces.Dict({
            "station_map": spaces.Box(low=0, high=1, shape=(self.N_candidates,), dtype=np.int8),
            "placements_left": spaces.Discrete(self.max_placements + 1)
        })

        # Action space: choose one of N_candidates
        self.action_space = spaces.Discrete(self.N_candidates)

        # Metrics storage for reward calculation
        self.last_metrics = {
            "coverage_fraction": 0.0,
            "avg_wait": 0.0
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
        self.last_metrics = {"coverage_fraction": 0.0, "avg_wait": 0.0}

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
        info = metrics.copy()

        return self._get_obs(), reward, terminated, truncated, info


    def _get_obs(self):
        return {
            "station_map": self.station_map.copy(),
            "placements_left": self.max_placements - self.placements_done
        }

    def _simulate(self, station_map):
        """
        Simplified simulator stub.
        Replace with SimPy or real simulation for detailed flow.
        """
        # Coverage fraction = fraction of population within catchment radius of any station
        coverage_fraction = min(1.0, 0.1 + 0.15 * station_map.sum())  # placeholder
        avg_wait = max(1.0, 10 - station_map.sum())  # placeholder
        return {"coverage_fraction": coverage_fraction, "avg_wait": avg_wait}

    def _compute_reward(self, metrics):
        """
        Reward = +coverage improvement - avg_wait penalty
        """
        alpha = 1.0  # weight for coverage
        beta = 0.1   # weight for wait time
        coverage_gain = metrics["coverage_fraction"] - self.last_metrics.get("coverage_fraction", 0)
        wait_increase = metrics["avg_wait"] - self.last_metrics.get("avg_wait", 0)
        reward = alpha * coverage_gain - beta * wait_increase
        return reward

    def render(self, mode="human"):
        print(f"Placements done: {self.placements_done}/{self.max_placements}")
        print(f"Coverage: {self.last_metrics['coverage_fraction']:.3f}, Avg Wait: {self.last_metrics['avg_wait']:.2f}")

    def close(self):
        pass

