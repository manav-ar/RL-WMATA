# simulator/simulator_core.py
import pandas as pd
import networkx as nx
import numpy as np
import pickle
import geopandas as gpd
import os

class Simulator:
    def __init__(self, network_path, od_path, candidates_path):
        # Load network graph
        if not os.path.exists(network_path):
            raise FileNotFoundError(f"Network file not found: {network_path}")
        with open(network_path, "rb") as f:
            self.G = pickle.load(f)

        # Load OD demand
        if not os.path.exists(od_path):
            raise FileNotFoundError(f"OD file not found: {od_path}")
        self.station_od = pd.read_csv(od_path)

        # Load candidate stations (CSV or GeoJSON)
        if not os.path.exists(candidates_path):
            raise FileNotFoundError(f"Candidates file not found: {candidates_path}")
        if candidates_path.endswith(".geojson") or candidates_path.endswith(".json"):
            gdf = gpd.read_file(candidates_path)
            self.candidates = pd.DataFrame(gdf.drop(columns='geometry'))
        elif candidates_path.endswith(".csv"):
            self.candidates = pd.read_csv(candidates_path)
        else:
            raise ValueError("Candidates file must be CSV or GeoJSON")

        # Basic sanity check
        required_cols = {"candidate_id", "lat", "lon", "population"}
        missing_cols = required_cols - set(self.candidates.columns)
        if missing_cols:
            raise ValueError(f"Candidates file missing required columns: {missing_cols}")

    def simulate_day(self, station_set):
        """
        Simulate a single day for the given set of station placements.
        Returns a dictionary of metrics.
        """
        if not station_set:
            coverage_fraction = 0.0
        else:
            # Simple coverage: fraction of total population covered by stations
            total_pop = self.candidates['population'].sum()
            covered_pop = self.candidates.loc[
                self.candidates['candidate_id'].isin(station_set), 'population'
            ].sum()
            coverage_fraction = covered_pop / total_pop

        # Example simulated metrics
        avg_wait = np.random.uniform(5, 15) / (1 + len(station_set)/10)
        congestion_index = np.random.uniform(0.5, 1.5) / (1 + len(station_set)/10)
        operational_cost = 5e6 * len(station_set)

        metrics = {
            'coverage_fraction': coverage_fraction,
            'avg_wait': avg_wait,
            'congestion_index': congestion_index,
            'operational_cost': operational_cost
        }
        return metrics
