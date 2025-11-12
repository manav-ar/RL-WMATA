# scripts/simulator.py
import pickle
import networkx as nx
from sampler import sample_trips

with open("data/prepared/network.pkl","rb") as f:
    G = pickle.load(f)

def simulate(station_set, random_seed=42):
    """
    Input: station_set = list of candidate_ids (new stations)
    Output: metrics dict
    """
    np.random.seed(random_seed)
    
    # Placeholder for passenger flows
    # Compute coverage: fraction of population within 1km of a station
    # For now, use dummy metrics
    coverage_fraction = 0.7 + 0.05*len(station_set)  # simplistic
    avg_wait_time = 5.0 / (1 + 0.2*len(station_set))
    total_walk_time = 10.0 / (1 + 0.1*len(station_set))
    operational_cost = 100e6 * len(station_set)
    
    metrics = {
        "coverage_fraction": coverage_fraction,
        "avg_wait_time": avg_wait_time,
        "total_walk_time": total_walk_time,
        "operational_cost": operational_cost
    }
    return metrics
