# test_gym.py
from simulator.env import StationPlacementEnv
import os

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
obs = env.reset()
print("Initial observation:", obs)

done = False
total_reward = 0.0

while not done:
    # Select the first available action (just for testing)
    action = int((env.action_mask * range(env.N_candidates)).nonzero()[0][0])
    obs, reward, done, info = env.step(action)
    total_reward += reward
    env.render()

print("Episode finished. Total reward:", total_reward)
