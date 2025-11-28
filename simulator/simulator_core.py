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

    def simulate_day(self, station_set, catchment_radius=800):
        """
        Simulate a single day for the given set of station placements.
        Uses realistic calculations based on catchment radius and OD demand.
        Returns a dictionary of metrics.
        """
        import math
        
        def haversine_distance(lat1, lon1, lat2, lon2):
            """Calculate distance in meters using Haversine formula."""
            R = 6371000  # Earth radius in meters
            phi1, phi2 = math.radians(lat1), math.radians(lat2)
            dphi = math.radians(lat2 - lat1)
            dlambda = math.radians(lon2 - lon1)
            a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
            return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
        
        if not station_set:
            coverage_fraction = 0.0
            avg_wait = 10.0
            total_trips_served = 0
        else:
            # Calculate coverage using catchment radius
            total_pop = self.candidates['population'].sum()
            covered_pop = 0
            
            placed_candidates = self.candidates[self.candidates['candidate_id'].isin(station_set)]
            
            for idx, candidate in self.candidates.iterrows():
                is_covered = False
                for _, placed in placed_candidates.iterrows():
                    distance = haversine_distance(
                        candidate['lat'], candidate['lon'],
                        placed['lat'], placed['lon']
                    )
                    if distance <= catchment_radius:
                        is_covered = True
                        break
                
                if is_covered:
                    covered_pop += candidate['population']
            
            coverage_fraction = covered_pop / total_pop if total_pop > 0 else 0.0
            
            # Calculate trips served between placed stations
            placed_od = self.station_od[
                (self.station_od['origin_candidate'].isin(station_set)) &
                (self.station_od['dest_candidate'].isin(station_set))
            ]
            total_trips_served = placed_od['mean_trips_per_day'].sum() if len(placed_od) > 0 else 0
            
            # Calculate average wait time based on stations and demand
            base_wait = 10.0
            station_factor = 1.0 / (1.0 + len(station_set) * 0.15)
            total_demand = self.station_od['mean_trips_per_day'].sum()
            demand_factor = 1.0 - (total_trips_served / total_demand) * 0.3 if total_demand > 0 else 1.0
            avg_wait = base_wait * station_factor * demand_factor
            avg_wait = max(1.0, avg_wait)

        # Calculate operational cost (realistic: $50M per station)
        operational_cost = 50e6 * len(station_set)
        
        # Congestion index: lower with more stations
        congestion_index = 1.5 / (1 + len(station_set) * 0.2) if station_set else 1.5

        metrics = {
            'coverage_fraction': coverage_fraction,
            'avg_wait': avg_wait,
            'congestion_index': congestion_index,
            'operational_cost': operational_cost,
            'total_trips_served': total_trips_served,
            'num_stations': len(station_set)
        }
        return metrics
