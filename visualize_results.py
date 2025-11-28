# visualize_results.py
"""
Visualize station placements from PPO and DQN models on a map.
Shows selected stations with lat/long coordinates.
"""
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from stable_baselines3 import PPO, DQN
from simulator.env import StationPlacementEnv
import os

NETWORK_PATH = "data/prepared/network.pkl"
CANDIDATES_PATH = "data/prepared/candidates_final.geojson"
STATION_OD_PATH = "data/prepared/station_od.csv"

def evaluate_model(model, model_name, env, candidates_df):
    """Evaluate a model and return selected stations with metrics."""
    obs, _ = env.reset()
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
            'population': station_info['population'],
            'step': step + 1
        })
        
        if terminated or truncated:
            break
    
    return {
        'stations': selected_stations,
        'metrics': {
            'total_reward': total_reward,
            'coverage': info.get('coverage_fraction', 0),
            'avg_wait': info.get('avg_wait', 0),
            'trips_served': info.get('total_trips_served', 0)
        }
    }

def create_visualization(ppo_results, dqn_results, candidates_df, output_path="visualizations/station_placements.png"):
    """Create a map visualization comparing PPO and DQN placements."""
    os.makedirs("visualizations", exist_ok=True)
    
    # Load DC boundary if available
    try:
        dc_boundary = gpd.read_file("data/dc_open/dc_boundary.geojson")
        has_boundary = True
    except:
        has_boundary = False
    
    # Create figure with subplots
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    
    # Plot PPO results
    ax1 = axes[0]
    if has_boundary:
        dc_boundary.plot(ax=ax1, color='lightgray', edgecolor='black', alpha=0.3)
    
    # Plot all candidates
    candidates_gdf = gpd.GeoDataFrame(
        candidates_df,
        geometry=gpd.points_from_xy(candidates_df['lon'], candidates_df['lat']),
        crs="EPSG:4326"
    )
    candidates_gdf.plot(ax=ax1, color='lightblue', markersize=5, alpha=0.3, label='All Candidates')
    
    # Plot PPO selected stations
    ppo_stations = pd.DataFrame(ppo_results['stations'])
    ppo_gdf = gpd.GeoDataFrame(
        ppo_stations,
        geometry=gpd.points_from_xy(ppo_stations['lon'], ppo_stations['lat']),
        crs="EPSG:4326"
    )
    ppo_gdf.plot(ax=ax1, color='red', markersize=100, marker='*', edgecolor='darkred', linewidth=2, label='PPO Selected')
    
    # Annotate PPO stations
    for idx, row in ppo_stations.iterrows():
        ax1.annotate(
            f"{row['step']}\n{row['candidate_id']}",
            (row['lon'], row['lat']),
            xytext=(5, 5),
            textcoords='offset points',
            fontsize=8,
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7)
        )
    
    ax1.set_title(f"PPO Agent\nCoverage: {ppo_results['metrics']['coverage']:.2%}, "
                  f"Wait: {ppo_results['metrics']['avg_wait']:.1f}min, "
                  f"Reward: {ppo_results['metrics']['total_reward']:.2f}",
                  fontsize=12, fontweight='bold')
    ax1.set_xlabel("Longitude", fontsize=10)
    ax1.set_ylabel("Latitude", fontsize=10)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot DQN results
    ax2 = axes[1]
    if has_boundary:
        dc_boundary.plot(ax=ax2, color='lightgray', edgecolor='black', alpha=0.3)
    
    # Plot all candidates
    candidates_gdf.plot(ax=ax2, color='lightblue', markersize=5, alpha=0.3, label='All Candidates')
    
    # Plot DQN selected stations
    dqn_stations = pd.DataFrame(dqn_results['stations'])
    dqn_gdf = gpd.GeoDataFrame(
        dqn_stations,
        geometry=gpd.points_from_xy(dqn_stations['lon'], dqn_stations['lat']),
        crs="EPSG:4326"
    )
    dqn_gdf.plot(ax=ax2, color='green', markersize=100, marker='*', edgecolor='darkgreen', linewidth=2, label='DQN Selected')
    
    # Annotate DQN stations
    for idx, row in dqn_stations.iterrows():
        ax2.annotate(
            f"{row['step']}\n{row['candidate_id']}",
            (row['lon'], row['lat']),
            xytext=(5, 5),
            textcoords='offset points',
            fontsize=8,
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7)
        )
    
    ax2.set_title(f"DQN Agent\nCoverage: {dqn_results['metrics']['coverage']:.2%}, "
                  f"Wait: {dqn_results['metrics']['avg_wait']:.1f}min, "
                  f"Reward: {dqn_results['metrics']['total_reward']:.2f}",
                  fontsize=12, fontweight='bold')
    ax2.set_xlabel("Longitude", fontsize=10)
    ax2.set_ylabel("Latitude", fontsize=10)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ Visualization saved to {output_path}")
    plt.close()

def print_station_details(results, model_name):
    """Print detailed station information."""
    print(f"\n{'='*60}")
    print(f"{model_name} Selected Stations (with Lat/Long)")
    print(f"{'='*60}")
    for i, station in enumerate(results['stations'], 1):
        print(f"{i}. {station['candidate_id']}: "
              f"Lat={station['lat']:.6f}, Lon={station['lon']:.6f}, "
              f"Population={station['population']:.0f}")
    print(f"\nMetrics:")
    print(f"  Total Reward: {results['metrics']['total_reward']:.3f}")
    print(f"  Coverage: {results['metrics']['coverage']:.3f} ({results['metrics']['coverage']*100:.1f}%)")
    print(f"  Avg Wait Time: {results['metrics']['avg_wait']:.2f} minutes")
    print(f"  Trips Served: {results['metrics']['trips_served']:.0f} trips/day")

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
        ppo_results = evaluate_model(ppo_model, "PPO", env_ppo, candidates_df)
        print_station_details(ppo_results, "PPO")
    except FileNotFoundError:
        print("❌ PPO model not found. Skipping PPO evaluation.")
    
    # Load and evaluate DQN
    dqn_results = None
    try:
        dqn_model = DQN.load("models/dqn_station_placement.zip", env=env_dqn)
        print("✅ Loaded DQN model")
        dqn_results = evaluate_model(dqn_model, "DQN", env_dqn, candidates_df)
        print_station_details(dqn_results, "DQN")
    except FileNotFoundError:
        print("❌ DQN model not found. Skipping DQN evaluation.")
    
    # Create visualization
    if ppo_results and dqn_results:
        create_visualization(ppo_results, dqn_results, candidates_df)
    elif ppo_results:
        print("\n⚠️  Only PPO results available. Creating single-model visualization...")
        # Create single model visualization
        fig, ax = plt.subplots(1, 1, figsize=(10, 8))
        try:
            dc_boundary = gpd.read_file("data/dc_open/dc_boundary.geojson")
            dc_boundary.plot(ax=ax, color='lightgray', edgecolor='black', alpha=0.3)
        except:
            pass
        
        candidates_gdf = gpd.GeoDataFrame(
            candidates_df,
            geometry=gpd.points_from_xy(candidates_df['lon'], candidates_df['lat']),
            crs="EPSG:4326"
        )
        candidates_gdf.plot(ax=ax, color='lightblue', markersize=5, alpha=0.3, label='All Candidates')
        
        ppo_stations = pd.DataFrame(ppo_results['stations'])
        ppo_gdf = gpd.GeoDataFrame(
            ppo_stations,
            geometry=gpd.points_from_xy(ppo_stations['lon'], ppo_stations['lat']),
            crs="EPSG:4326"
        )
        ppo_gdf.plot(ax=ax, color='red', markersize=100, marker='*', edgecolor='darkred', linewidth=2, label='PPO Selected')
        
        for idx, row in ppo_stations.iterrows():
            ax.annotate(
                f"{row['step']}\n{row['candidate_id']}",
                (row['lon'], row['lat']),
                xytext=(5, 5),
                textcoords='offset points',
                fontsize=8,
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7)
            )
        
        ax.set_title(f"PPO Agent - Selected Stations\nCoverage: {ppo_results['metrics']['coverage']:.2%}, "
                     f"Wait: {ppo_results['metrics']['avg_wait']:.1f}min", fontsize=12, fontweight='bold')
        ax.set_xlabel("Longitude", fontsize=10)
        ax.set_ylabel("Latitude", fontsize=10)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        os.makedirs("visualizations", exist_ok=True)
        plt.savefig("visualizations/ppo_stations.png", dpi=300, bbox_inches='tight')
        print("✅ Visualization saved to visualizations/ppo_stations.png")
        plt.close()
    elif dqn_results:
        print("\n⚠️  Only DQN results available. Creating single-model visualization...")
        # Similar single visualization for DQN
        fig, ax = plt.subplots(1, 1, figsize=(10, 8))
        try:
            dc_boundary = gpd.read_file("data/dc_open/dc_boundary.geojson")
            dc_boundary.plot(ax=ax, color='lightgray', edgecolor='black', alpha=0.3)
        except:
            pass
        
        candidates_gdf = gpd.GeoDataFrame(
            candidates_df,
            geometry=gpd.points_from_xy(candidates_df['lon'], candidates_df['lat']),
            crs="EPSG:4326"
        )
        candidates_gdf.plot(ax=ax, color='lightblue', markersize=5, alpha=0.3, label='All Candidates')
        
        dqn_stations = pd.DataFrame(dqn_results['stations'])
        dqn_gdf = gpd.GeoDataFrame(
            dqn_stations,
            geometry=gpd.points_from_xy(dqn_stations['lon'], dqn_stations['lat']),
            crs="EPSG:4326"
        )
        dqn_gdf.plot(ax=ax, color='green', markersize=100, marker='*', edgecolor='darkgreen', linewidth=2, label='DQN Selected')
        
        for idx, row in dqn_stations.iterrows():
            ax.annotate(
                f"{row['step']}\n{row['candidate_id']}",
                (row['lon'], row['lat']),
                xytext=(5, 5),
                textcoords='offset points',
                fontsize=8,
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7)
            )
        
        ax.set_title(f"DQN Agent - Selected Stations\nCoverage: {dqn_results['metrics']['coverage']:.2%}, "
                     f"Wait: {dqn_results['metrics']['avg_wait']:.1f}min", fontsize=12, fontweight='bold')
        ax.set_xlabel("Longitude", fontsize=10)
        ax.set_ylabel("Latitude", fontsize=10)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        os.makedirs("visualizations", exist_ok=True)
        plt.savefig("visualizations/dqn_stations.png", dpi=300, bbox_inches='tight')
        print("✅ Visualization saved to visualizations/dqn_stations.png")
        plt.close()
    else:
        print("\n❌ No trained models found. Please train models first.")

