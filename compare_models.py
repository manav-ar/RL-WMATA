# compare_models.py
"""
Compare PPO and DQN models side by side.
Shows selected stations with lat/long and performance metrics.
"""
import pandas as pd
from stable_baselines3 import PPO, DQN
from simulator.env import StationPlacementEnv
import os

NETWORK_PATH = "data/prepared/network.pkl"
CANDIDATES_PATH = "data/prepared/candidates_final.geojson"
STATION_OD_PATH = "data/prepared/station_od.csv"

def run_model_evaluation(model, model_name, env, candidates_df, n_episodes=5):
    """Run multiple episodes and aggregate results."""
    all_results = []
    
    for episode in range(n_episodes):
        obs, _ = env.reset(seed=42 + episode)
        selected_stations = []
        total_reward = 0.0
        
        for step in range(5):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            
            selected_id = env.candidate_ids[action]
            station_info = candidates_df[candidates_df['candidate_id'] == selected_id].iloc[0]
            selected_stations.append({
                'candidate_id': selected_id,
                'lat': station_info['lat'],
                'lon': station_info['lon'],
                'population': station_info['population']
            })
            
            if terminated or truncated:
                break
        
        all_results.append({
            'episode': episode + 1,
            'stations': selected_stations,
            'total_reward': total_reward,
            'coverage': info.get('coverage_fraction', 0),
            'avg_wait': info.get('avg_wait', 0),
            'trips_served': info.get('total_trips_served', 0)
        })
    
    return all_results

def print_comparison_table(ppo_results, dqn_results):
    """Print a formatted comparison table."""
    print("\n" + "="*100)
    print("MODEL COMPARISON - Selected Stations with Lat/Long")
    print("="*100)
    
    # Find the best episode for each model
    ppo_best = max(ppo_results, key=lambda x: x['total_reward'])
    dqn_best = max(dqn_results, key=lambda x: x['total_reward'])
    
    print(f"\n{'PPO (Best Episode)':^50} | {'DQN (Best Episode)':^50}")
    print("-" * 100)
    
    max_stations = max(len(ppo_best['stations']), len(dqn_best['stations']))
    
    for i in range(max_stations):
        ppo_station = ppo_best['stations'][i] if i < len(ppo_best['stations']) else None
        dqn_station = dqn_best['stations'][i] if i < len(dqn_best['stations']) else None
        
        if ppo_station:
            ppo_str = f"{ppo_station['candidate_id']}\nLat: {ppo_station['lat']:.6f}\nLon: {ppo_station['lon']:.6f}"
        else:
            ppo_str = "-"
        
        if dqn_station:
            dqn_str = f"{dqn_station['candidate_id']}\nLat: {dqn_station['lat']:.6f}\nLon: {dqn_station['lon']:.6f}"
        else:
            dqn_str = "-"
        
        print(f"{ppo_str:^50} | {dqn_str:^50}")
    
    print("\n" + "="*100)
    print("PERFORMANCE METRICS COMPARISON")
    print("="*100)
    
    # Calculate averages
    ppo_avg_reward = sum(r['total_reward'] for r in ppo_results) / len(ppo_results)
    dqn_avg_reward = sum(r['total_reward'] for r in dqn_results) / len(dqn_results)
    ppo_avg_coverage = sum(r['coverage'] for r in ppo_results) / len(ppo_results)
    dqn_avg_coverage = sum(r['coverage'] for r in dqn_results) / len(dqn_results)
    ppo_avg_wait = sum(r['avg_wait'] for r in ppo_results) / len(ppo_results)
    dqn_avg_wait = sum(r['avg_wait'] for r in dqn_results) / len(dqn_results)
    
    print(f"\n{'Metric':<20} | {'PPO':<20} | {'DQN':<20} | {'Winner':<20}")
    print("-" * 80)
    print(f"{'Avg Reward':<20} | {ppo_avg_reward:<20.3f} | {dqn_avg_reward:<20.3f} | {'PPO' if ppo_avg_reward > dqn_avg_reward else 'DQN':<20}")
    print(f"{'Avg Coverage':<20} | {ppo_avg_coverage:<20.3f} | {dqn_avg_coverage:<20.3f} | {'PPO' if ppo_avg_coverage > dqn_avg_coverage else 'DQN':<20}")
    print(f"{'Avg Wait Time':<20} | {ppo_avg_wait:<20.2f} | {dqn_avg_wait:<20.2f} | {'PPO' if ppo_avg_wait < dqn_avg_wait else 'DQN':<20}")
    
    print(f"\n{'Best Episode Reward':<20} | {ppo_best['total_reward']:<20.3f} | {dqn_best['total_reward']:<20.3f} | {'PPO' if ppo_best['total_reward'] > dqn_best['total_reward'] else 'DQN':<20}")
    print(f"{'Best Coverage':<20} | {ppo_best['coverage']:<20.3f} | {dqn_best['coverage']:<20.3f} | {'PPO' if ppo_best['coverage'] > dqn_best['coverage'] else 'DQN':<20}")
    print(f"{'Best Wait Time':<20} | {ppo_best['avg_wait']:<20.2f} | {dqn_best['avg_wait']:<20.2f} | {'PPO' if ppo_best['avg_wait'] < dqn_best['avg_wait'] else 'DQN':<20}")
    
    print("\n" + "="*100)
    print("DETAILED STATION COORDINATES")
    print("="*100)
    
    print(f"\n{'PPO Best Episode Stations':^50}")
    print("-" * 50)
    for i, station in enumerate(ppo_best['stations'], 1):
        print(f"{i}. {station['candidate_id']}")
        print(f"   Latitude:  {station['lat']:.6f}")
        print(f"   Longitude: {station['lon']:.6f}")
        print(f"   Population: {station['population']:.0f}")
    
    print(f"\n{'DQN Best Episode Stations':^50}")
    print("-" * 50)
    for i, station in enumerate(dqn_best['stations'], 1):
        print(f"{i}. {station['candidate_id']}")
        print(f"   Latitude:  {station['lat']:.6f}")
        print(f"   Longitude: {station['lon']:.6f}")
        print(f"   Population: {station['population']:.0f}")

if __name__ == "__main__":
    # Load candidates
    candidates_df = pd.read_csv(CANDIDATES_PATH.replace(".geojson", ".csv"))
    
    # Create environments
    env_ppo = StationPlacementEnv(
        network_path=NETWORK_PATH,
        candidates_path=CANDIDATES_PATH,
        station_od_path=STATION_OD_PATH,
        max_placements=5,
        catchment_radius=800,
        seed=42
    )
    
    env_dqn = StationPlacementEnv(
        network_path=NETWORK_PATH,
        candidates_path=CANDIDATES_PATH,
        station_od_path=STATION_OD_PATH,
        max_placements=5,
        catchment_radius=800,
        seed=42
    )
    
    # Load and evaluate PPO
    ppo_results = None
    try:
        ppo_model = PPO.load("models/ppo_station_placement.zip", env=env_ppo)
        print("✅ Loaded PPO model")
        print("Running PPO evaluation...")
        ppo_results = run_model_evaluation(ppo_model, "PPO", env_ppo, candidates_df, n_episodes=5)
    except FileNotFoundError:
        print("❌ PPO model not found. Please train first: python -m agents.train_ppo")
    
    # Load and evaluate DQN
    dqn_results = None
    try:
        dqn_model = DQN.load("models/dqn_station_placement.zip", env=env_dqn)
        print("✅ Loaded DQN model")
        print("Running DQN evaluation...")
        dqn_results = run_model_evaluation(dqn_model, "DQN", env_dqn, candidates_df, n_episodes=5)
    except FileNotFoundError:
        print("❌ DQN model not found. Please train first: python -m agents.train_dqn")
    
    # Compare results
    if ppo_results and dqn_results:
        print_comparison_table(ppo_results, dqn_results)
    else:
        print("\n⚠️  Both models are required for comparison.")
        if ppo_results:
            print("PPO results available but DQN model missing.")
        if dqn_results:
            print("DQN results available but PPO model missing.")

