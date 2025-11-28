# agents/viz_callback.py
"""
Callback to save visualization images during training.
Creates images every N episodes for GIF generation.
"""
import os
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from stable_baselines3.common.callbacks import BaseCallback
from simulator.env import StationPlacementEnv

class VisualizationCallback(BaseCallback):
    """
    Saves visualization images every N episodes during training.
    """
    def __init__(self, 
                 candidates_path,
                 network_path,
                 station_od_path,
                 save_dir="visualizations/training",
                 episodes_per_image=500,
                 verbose=0):
        super().__init__(verbose)
        self.candidates_path = candidates_path
        self.network_path = network_path
        self.station_od_path = station_od_path
        self.save_dir = save_dir
        self.episodes_per_image = episodes_per_image
        self.episode_count = 0
        self.last_episode_saved = -1
        
        # Create save directory
        os.makedirs(self.save_dir, exist_ok=True)
        
        # Load candidates for visualization
        if candidates_path.endswith(".geojson"):
            csv_path = candidates_path.replace(".geojson", ".csv")
            if os.path.exists(csv_path):
                self.candidates_df = pd.read_csv(csv_path)
            else:
                gdf = gpd.read_file(candidates_path)
                self.candidates_df = pd.DataFrame(gdf.drop(columns='geometry'))
        else:
            self.candidates_df = pd.read_csv(candidates_path)
    
    def _on_step(self) -> bool:
        # Check if episode ended
        # In VecEnv, 'dones' can be a numpy array, list, or boolean
        dones = self.locals.get('dones', [False])
        
        # Handle different types of dones safely
        episode_ended = False
        
        # Check if dones is a numpy array (has dtype attribute)
        if hasattr(dones, 'dtype'):  # NumPy array
            if dones.size > 0:
                episode_ended = bool(dones.flat[0])  # Use flat[0] to safely get first element
        elif isinstance(dones, (list, tuple)):
            if len(dones) > 0:
                episode_ended = bool(dones[0])
        elif isinstance(dones, bool):
            episode_ended = dones
        else:
            # Try to convert (handles numpy scalars, etc.)
            try:
                episode_ended = bool(dones)
            except (ValueError, TypeError):
                episode_ended = False
        
        if episode_ended:
            self.episode_count += 1
            
            # Save image every N episodes
            if self.episode_count % self.episodes_per_image == 0:
                self._save_training_image()
        
        return True
    
    def _save_training_image(self):
        """Save a visualization image of current model performance."""
        try:
            # Get current model
            model = self.model
            
            # Create a fresh environment for evaluation (not the training env)
            # We need to create it directly, not through make_vec_env
            env = StationPlacementEnv(
                network_path=self.network_path,
                candidates_path=self.candidates_path,
                station_od_path=self.station_od_path,
                max_placements=5,
                catchment_radius=800,
                seed=42 + self.episode_count  # Different seed for variety
            )
            
            # Run one episode
            obs, _ = env.reset()
            selected_stations = []
            total_reward = 0.0
            
            for step in range(5):
                action, _ = model.predict(obs, deterministic=False)  # Use stochastic for variety
                obs, reward, terminated, truncated, info = env.step(action)
                total_reward += reward
                
                selected_id = env.candidate_ids[action]
                station_info = self.candidates_df[
                    self.candidates_df['candidate_id'] == selected_id
                ]
                if len(station_info) > 0:
                    selected_stations.append({
                        'candidate_id': selected_id,
                        'lat': station_info.iloc[0]['lat'],
                        'lon': station_info.iloc[0]['lon'],
                        'population': station_info.iloc[0]['population'],
                        'step': step + 1
                    })
                
                if terminated or truncated:
                    break
            
            # Create visualization
            self._create_image(selected_stations, total_reward, info, self.episode_count)
            
        except Exception as e:
            if self.verbose > 0:
                print(f"Warning: Could not save training image: {e}")
    
    def _create_image(self, selected_stations, total_reward, info, episode):
        """Create and save a single training visualization image."""
        fig, ax = plt.subplots(1, 1, figsize=(12, 10))
        
        # Load DC boundary if available
        try:
            dc_boundary = gpd.read_file("data/dc_open/dc_boundary.geojson")
            dc_boundary.plot(ax=ax, color='lightgray', edgecolor='black', alpha=0.3)
        except:
            pass
        
        # Plot all candidates
        candidates_gdf = gpd.GeoDataFrame(
            self.candidates_df,
            geometry=gpd.points_from_xy(self.candidates_df['lon'], self.candidates_df['lat']),
            crs="EPSG:4326"
        )
        candidates_gdf.plot(ax=ax, color='lightblue', markersize=3, alpha=0.2, label='All Candidates')
        
        # Plot selected stations
        if selected_stations:
            stations_df = pd.DataFrame(selected_stations)
            stations_gdf = gpd.GeoDataFrame(
                stations_df,
                geometry=gpd.points_from_xy(stations_df['lon'], stations_df['lat']),
                crs="EPSG:4326"
            )
            stations_gdf.plot(ax=ax, color='red', markersize=150, marker='*', 
                            edgecolor='darkred', linewidth=2, label='Selected Stations')
            
            # Annotate stations
            for idx, row in stations_df.iterrows():
                ax.annotate(
                    f"{row['step']}\n{row['candidate_id']}",
                    (row['lon'], row['lat']),
                    xytext=(5, 5),
                    textcoords='offset points',
                    fontsize=9,
                    bbox=dict(boxstyle='round,pad=0.4', facecolor='yellow', alpha=0.8)
                )
        
        # Set title with metrics
        coverage = info.get('coverage_fraction', 0)
        avg_wait = info.get('avg_wait', 0)
        model_name = "PPO" if "PPO" in str(type(self.model)) else "DQN"
        
        ax.set_title(
            f"{model_name} Training - Episode {episode}\n"
            f"Reward: {total_reward:.2f} | Coverage: {coverage:.1%} | Wait: {avg_wait:.1f}min",
            fontsize=14, fontweight='bold'
        )
        ax.set_xlabel("Longitude", fontsize=11)
        ax.set_ylabel("Latitude", fontsize=11)
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)
        
        # Save image
        filename = f"{self.save_dir}/episode_{episode:06d}.png"
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        plt.close()
        
        if self.verbose > 0:
            print(f"  💾 Saved training image: {filename}")

