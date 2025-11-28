# test_gym.py
from simulator.env import StationPlacementEnv
import os
import numpy as np

# Paths to prepared data
network_path = "data/prepared/multi_modal_graph.pkl"
candidates_path = "data/prepared/candidates.geojson"  # GeoJSON will auto-convert
station_od_path = "data/prepared/station_od.csv"

# Check files exist
for path in [network_path, candidates_path, station_od_path]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Required file not found: {path}")

# Create environment
env = StationPlacementEnv(
    network_path=network_path,
    candidates_path=candidates_path,
    station_od_path=station_od_path,
    max_placements=5,
    seed=42
)

# Reset env
obs, info = env.reset()
print("Initial observation keys:", obs.keys())
print("Station map shape:", obs["station_map"].shape)
print("Placements left:", obs["placements_left"])
print("Action mask shape:", obs["action_mask"].shape)

done = False
total_reward = 0.0

while not done:
    # Select the first available action (just for testing)
    valid_actions = np.where(env.action_mask)[0]
    if len(valid_actions) == 0:
        break
    action = valid_actions[0]
    obs, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated
    total_reward += reward
    env.render()

print("\nEpisode finished. Total reward:", total_reward)
