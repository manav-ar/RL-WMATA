# agents/train_ppo.py
import os
import argparse
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env, SubprocVecEnv, DummyVecEnv
from stable_baselines3.common.callbacks import CallbackList
from simulator.env import StationPlacementEnv
from agents.tb_callback import TBMetricsCallback
from agents.viz_callback import VisualizationCallback

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
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Train PPO agent for station placement")
    parser.add_argument(
        "--total_timesteps",
        type=int,
        default=10000,
        help="Total number of training timesteps (default: 10000)"
    )
    args = parser.parse_args()
    total_timesteps = args.total_timesteps
    
    # Check for GPU and set up environment
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # GPU-optimized environment setup
    # For very small test runs, use 1 env to avoid complexity
    # For GPU: use more parallel envs and SubprocVecEnv for true parallelism
    # For CPU: use fewer envs and DummyVecEnv (sequential but lower overhead)
    if total_timesteps < 1000:
        n_envs = 1  # Single env for small test runs
        vec_env_class = DummyVecEnv  # Sequential for single env
        print(f"📦 Using 1 environment for small test run")
    elif torch.cuda.is_available():
        # GPU: Use fewer envs with DummyVecEnv for faster env steps
        # SubprocVecEnv has high overhead when env steps are slow
        n_envs = 4  # Reduced from 8 - DummyVecEnv is faster for CPU-bound envs
        vec_env_class = DummyVecEnv  # Use DummyVecEnv - env steps are CPU-bound anyway
        print(f"📦 Creating {n_envs} parallel environments (DummyVecEnv) for GPU training...")
    else:
        n_envs = 1  # CPU: single env to avoid overhead
        vec_env_class = DummyVecEnv
        print(f"📦 Creating {n_envs} environment for CPU training...")
    
    # Create vectorized environment with appropriate class
    # Try SubprocVecEnv first, fallback to DummyVecEnv if it fails (e.g., in Colab)
    try:
        env = make_vec_env(make_env, n_envs=n_envs, vec_env_cls=vec_env_class)
        if n_envs > 1:
            print(f"🚀 Using {n_envs} parallel environments ({vec_env_class.__name__}) for faster training")
    except Exception as e:
        print(f"⚠️  Warning: {vec_env_class.__name__} failed ({e}), falling back to DummyVecEnv")
        env = make_vec_env(make_env, n_envs=n_envs, vec_env_cls=DummyVecEnv)
        if n_envs > 1:
            print(f"🚀 Using {n_envs} parallel environments (DummyVecEnv) for training")
    
    # Test environment before training
    print("🧪 Testing environment...")
    try:
        test_obs = env.reset()
        print(f"   ✅ Environment reset successful")
        print(f"   ✅ Observation keys: {list(test_obs.keys()) if isinstance(test_obs, dict) else 'vector'}")
        
        # Test a few steps to ensure environment works
        print("   🧪 Testing environment step...")
        import numpy as np
        # Get a valid action from action mask
        if isinstance(test_obs, dict) and 'action_mask' in test_obs:
            # VecEnv always returns dict of arrays, even with n_envs=1
            # Access first environment's action mask
            action_mask = test_obs['action_mask'][0]
            valid_actions = np.where(action_mask)[0]
            if len(valid_actions) > 0:
                test_action = valid_actions[0]
                # VecEnv expects array of actions with shape (n_envs,)
                # Ensure we create array with exactly n_envs elements
                actions = np.array([test_action] * n_envs, dtype=np.int32)
                test_obs, test_reward, test_done, test_info = env.step(actions)
                print(f"   ✅ Environment step successful")
                # VecEnv always returns arrays
                print(f"      Reward: {test_reward[0]:.3f}, Done: {test_done[0]}")
            else:
                print(f"   ⚠️  No valid actions found in action mask")
        else:
            # Fallback: try action 0
            # Ensure array matches n_envs
            actions = np.array([0] * n_envs, dtype=np.int32)
            test_obs, test_reward, test_done, test_info = env.step(actions)
            print(f"   ✅ Environment step successful")
            print(f"      Reward: {test_reward[0]:.3f}, Done: {test_done[0]}")
    except Exception as e:
        print(f"   ❌ Environment test failed: {e}")
        import traceback
        traceback.print_exc()
        raise
    
    if device == "cuda":
        print(f"🚀 Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("💻 Using CPU")
    
    # PPO requires n_steps timesteps to collect a rollout before learning
    # PPO also requires n_steps * n_envs > 1 (minimum 2)
    # GPU-optimized: Use moderate n_steps (smaller for faster collection)
    if torch.cuda.is_available() and total_timesteps >= 1000:
        default_n_steps = 128  # Reduced from 512 - faster rollout collection
    else:
        default_n_steps = 128  # Standard for CPU or small runs
    
    if total_timesteps < default_n_steps * n_envs:
        # Adjust n_steps to be at most total_timesteps / n_envs
        # But ensure n_steps * n_envs >= 2 (PPO requirement)
        min_required = max(2, (2 + n_envs - 1) // n_envs)  # Ceiling division to ensure >= 2 total
        adjusted_n_steps = max(min_required, total_timesteps // n_envs)
        print(f"⚠️  Warning: total_timesteps ({total_timesteps}) is less than required n_steps ({default_n_steps * n_envs})")
        print(f"   Adjusting n_steps to {adjusted_n_steps} for this test run (n_steps * n_envs = {adjusted_n_steps * n_envs})")
        n_steps = adjusted_n_steps
        # Also disable visualization for very short runs to avoid overhead
        use_viz = False
    else:
        n_steps = default_n_steps
        use_viz = True
    
    # GPU-optimized hyperparameters
    if torch.cuda.is_available():
        batch_size = 128  # Reduced from 256 - faster training with smaller batches
        n_epochs = 4  # Reduced from 10 - faster updates
        policy_kwargs = dict(
            net_arch=[dict(pi=[128, 128], vf=[128, 128])]  # Smaller networks for faster training
        )
    else:
        batch_size = 64
        n_epochs = 4
        policy_kwargs = dict(
            net_arch=[dict(pi=[128, 128], vf=[128, 128])]  # Smaller networks for CPU
        )
    
    # Define PPO model with GPU-optimized settings
    # MultiInputPolicy automatically handles action masks when action_mask is in observation
    model = PPO(
        "MultiInputPolicy",
        env,
        verbose=1,
        n_steps=n_steps,
        batch_size=batch_size,
        n_epochs=n_epochs,
        learning_rate=3e-4,
        gamma=0.99,
        tensorboard_log="./logs/",
        device=device,  # Explicitly set device for GPU
        policy_kwargs=policy_kwargs,
        # Ensure action mask is properly handled
        # MultiInputPolicy will automatically mask invalid actions if action_mask is in obs
    )
    
    # Create callbacks
    callbacks = [TBMetricsCallback()]
    
    # Disable visualization during training for speed (can enable after training)
    # Visualization callback runs evaluation episodes which are slow
    if use_viz and total_timesteps < 50000:
        # Only enable visualization for very long training runs
        print(f"ℹ️  Visualization disabled for faster training (enable for runs > 50k timesteps)")
    elif use_viz:
        viz_callback = VisualizationCallback(
            candidates_path=CANDIDATES_PATH,
            network_path=NETWORK_PATH,
            station_od_path=STATION_OD_PATH,
            save_dir="visualizations/training/ppo",
            episodes_per_image=1000,  # Less frequent to reduce overhead
            verbose=0  # Reduce output
        )
        callbacks.append(viz_callback)
        print(f"📸 Training with visualization: saving images every 1000 episodes")
    else:
        print(f"ℹ️  Visualization disabled for short test run")
    
    # Combine callbacks
    callback_list = CallbackList(callbacks)
    
    # Train
    print(f"🚀 Starting training for {total_timesteps} timesteps...")
    print(f"   n_steps={n_steps}, batch_size={model.batch_size}, n_epochs={model.n_epochs}, n_envs={n_envs}")
    print(f"   VecEnv type: {type(env).__name__}")
    if torch.cuda.is_available():
        print(f"   GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    print(f"   This should take a few seconds for small runs...")
    try:
        # progress_bar might not be available in all SB3 versions
        try:
            model.learn(total_timesteps=total_timesteps, callback=callback_list, progress_bar=True)
        except TypeError:
            # Fallback if progress_bar not supported
            model.learn(total_timesteps=total_timesteps, callback=callback_list)
    except KeyboardInterrupt:
        print("\n⚠️  Training interrupted by user")
        raise
    except Exception as e:
        print(f"\n❌ Training error: {e}")
        import traceback
        traceback.print_exc()
        raise

    # Save trained model
    os.makedirs("models", exist_ok=True)
    model.save("models/ppo_station_placement")

    print(f"✅ Training complete. Model saved to models/ppo_station_placement.zip")

