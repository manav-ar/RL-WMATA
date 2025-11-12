# agents/train_ppo.py
import os
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from simulator.env import StationPlacementEnv
from agents.tb_callback import TBMetricsCallback

# Define data paths relative to project root
NETWORK_PATH = "data/prepared/network.pkl"
CANDIDATES_PATH = "data/prepared/candidates_final.geojson"
STATION_OD_PATH = "data/prepared/station_od.csv"

def make_env():
    return StationPlacementEnv(
        network_path=NETWORK_PATH,
        candidates_path=CANDIDATES_PATH,
        station_od_path=STATION_OD_PATH,
        max_placements=5,
        catchment_radius=800,
        seed=42
    )

if __name__ == "__main__":
    # Create vectorized environment (1 parallel env)
    env = make_vec_env(make_env, n_envs=1)

    # Define PPO model
    model = PPO(
        "MultiInputPolicy",
        env,
        verbose=1,
        n_steps=256,
        batch_size=64,
        learning_rate=3e-4,
        gamma=0.99,
        tensorboard_log="./logs/"
    )
    tb_callback = TBMetricsCallback()
    
    # Train
    total_timesteps = 10000 
    model.learn(total_timesteps=total_timesteps, callback=tb_callback)

    # Save trained model
    os.makedirs("models", exist_ok=True)
    model.save("models/ppo_station_placement")

    print(f"✅ Training complete. Model saved to models/ppo_station_placement.zip")

