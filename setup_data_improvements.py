#!/usr/bin/env python3
"""
Setup script to run all data improvement scripts.
This prepares enriched data for better RL training.
"""
import os
import sys
import subprocess

def run_script(script_path, description):
    """Run a Python script and handle errors."""
    print(f"\n{'='*60}")
    print(f"Running: {description}")
    print(f"Script: {script_path}")
    print(f"{'='*60}")
    
    if not os.path.exists(script_path):
        print(f"⚠️  Script not found: {script_path}")
        return False
    
    try:
        result = subprocess.run(
            [sys.executable, script_path],
            check=True,
            capture_output=True,
            text=True
        )
        print(result.stdout)
        if result.stderr:
            print("Warnings:", result.stderr)
        print(f"✅ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error running {script_path}:")
        print(e.stdout)
        print(e.stderr)
        return False

def main():
    """Run all data improvement scripts."""
    print("="*60)
    print("RL-WMATA Data Improvements Setup")
    print("="*60)
    print("\nThis script will:")
    print("1. Enrich candidates with existing station distances")
    print("2. Enhance network graph with existing station connections")
    print("\nStarting setup...")
    
    scripts = [
        ("scripts/enrich_candidates_with_existing_stations.py", 
         "Enriching candidates with existing station data"),
        ("scripts/improve_network_with_existing_stations.py",
         "Enhancing network graph with existing stations")
    ]
    
    results = []
    for script_path, description in scripts:
        success = run_script(script_path, description)
        results.append((description, success))
    
    print("\n" + "="*60)
    print("Setup Summary")
    print("="*60)
    
    for description, success in results:
        status = "✅ Success" if success else "❌ Failed"
        print(f"{status}: {description}")
    
    all_success = all(success for _, success in results)
    
    if all_success:
        print("\n✅ All data improvements set up successfully!")
        print("\nNext steps:")
        print("1. Test the environment: python test_gym.py")
        print("2. Retrain models with enriched data:")
        print("   python -m agents.train_ppo --total_timesteps 100000")
        print("   python -m agents.train_dqn --total_timesteps 100000")
    else:
        print("\n⚠️  Some scripts failed. Check errors above.")
        print("You can still use the project, but some features may not be available.")
    
    return 0 if all_success else 1

if __name__ == "__main__":
    sys.exit(main())

