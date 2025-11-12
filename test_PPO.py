from stable_baselines3 import PPO
from simulator.env import StationPlacementEnv

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
model = PPO.load("models/ppo_station_placement.zip", env=env)

obs, _ = env.reset()
for _ in range(5):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)
    env.render()
