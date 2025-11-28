from stable_baselines3 import PPO
from simulator.env import StationPlacementEnv
import pandas as pd
import numpy as np

NETWORK_PATH = "data/prepared/network.pkl"
CANDIDATES_PATH = "data/prepared/candidates_final.geojson"
STATION_OD_PATH = "data/prepared/station_od.csv"

env = StationPlacementEnv(
    network_path=NETWORK_PATH,
    candidates_path=CANDIDATES_PATH,
    station_od_path=STATION_OD_PATH,
    max_placements=5,
    catchment_radius=800,
    seed=42
)

# Load candidates to get lat/long
candidates_df = pd.read_csv(CANDIDATES_PATH.replace(".geojson", ".csv"))

# Load trained PPO model
try:
    model = PPO.load("models/ppo_station_placement.zip", env=env)
    print("✅ Loaded trained PPO model")
except FileNotFoundError:
    print("❌ PPO model not found. Please train first with: python -m agents.train_ppo")
    exit(1)

# Run evaluation
obs, _ = env.reset()
total_reward = 0.0
selected_stations = []

print("\n" + "="*60)
print("PPO Agent Evaluation")
print("="*60)

for step in range(5):
    # Get valid actions from mask
    valid_actions = np.where(obs['action_mask'])[0]
    if len(valid_actions) == 0:
        print("⚠️  No valid actions remaining!")
        break
    
    # Predict with action mask - MultiInputPolicy should respect mask automatically
    action, _ = model.predict(obs, deterministic=True)
    
    # Extract scalar action - handle different types (scalar, 0-d array, 1-d array, list)
    if isinstance(action, np.ndarray):
        if action.ndim == 0:
            # 0-dimensional array (scalar array)
            action = int(action.item())
        else:
            # Multi-dimensional array, take first element
            action = int(action.flat[0])
    elif isinstance(action, (list, tuple)):
        action = int(action[0])
    else:
        action = int(action)
    
    # Ensure action is valid (enforce mask if model doesn't respect it)
    if action >= len(obs['action_mask']) or not obs['action_mask'][action]:
        # If invalid, choose random valid action
        action = int(np.random.choice(valid_actions))
        print(f"⚠️  Step {step+1}: Model selected invalid action {action}, using random valid: {action}")
    else:
        print(f"✅ Step {step+1}: Model selected valid action: {action}")
    
    obs, reward, terminated, truncated, info = env.step(action)
    total_reward += reward
    
    # Get selected station info
    selected_id = env.candidate_ids[action]
    station_info = candidates_df[candidates_df['candidate_id'] == selected_id].iloc[0]
    
    # Check for duplicates before adding
    if any(s['candidate_id'] == selected_id for s in selected_stations):
        print(f"⚠️  WARNING: {selected_id} was already selected! Skipping duplicate.")
    else:
        selected_stations.append({
            'candidate_id': selected_id,
            'lat': station_info['lat'],
            'lon': station_info['lon'],
            'population': station_info['population']
        })
    
    print(f"   Selected {selected_id}, Reward: {reward:.3f}, Coverage: {info.get('coverage_fraction', 0):.3f}")
    
    env.render()
    
    if terminated or truncated:
        break

# Get final metrics from environment state (not just last info)
# Use unique placed stations only
placed_indices = np.where(env.station_map == 1)[0]
print(f"\n🔍 Debug: Final station_map has {len(placed_indices)} placed stations at indices: {placed_indices}")
final_metrics = env._simulate(env.station_map)

print("\n" + "="*60)
print("PPO Selected Stations (with Lat/Long)")
print("="*60)
for i, station in enumerate(selected_stations, 1):
    print(f"{i}. {station['candidate_id']}: "
          f"Lat={station['lat']:.6f}, Lon={station['lon']:.6f}, "
          f"Population={station['population']:.0f}")

print(f"\nMetrics:")
print(f"  Total Reward: {total_reward:.3f}")
print(f"  Coverage: {final_metrics.get('coverage_fraction', 0):.3f} ({final_metrics.get('coverage_fraction', 0)*100:.1f}%)")
print(f"  Avg Wait Time: {final_metrics.get('avg_wait', 0):.2f} minutes")
print(f"  Trips Served: {final_metrics.get('total_trips_served', 0):.0f} trips/day")
