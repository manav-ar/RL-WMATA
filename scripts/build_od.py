# scripts/build_station_od_synthetic.py
import geopandas as gpd
import pandas as pd
import numpy as np

# Load candidate stations
candidates = gpd.read_file("data/prepared/candidates_final.geojson")

# Create synthetic OD demand
# For simplicity: trips proportional to population of origin and destination
od_list = []
for i, origin in candidates.iterrows():
    for j, dest in candidates.iterrows():
        if origin["candidate_id"] == dest["candidate_id"]:
            continue
        # Synthetic trips: proportional to origin population * dest population
        trips = int(0.001 * origin["population"] * dest["population"])
        if trips > 0:
            # Hourly distribution: simple rush hour peaks
            hourly_profile = [0.02]*7 + [0.1,0.1] + [0.05]*6 + [0.1,0.1] + [0.02]*3
            # scale hourly profile to total trips
            total_profile = np.array(hourly_profile)
            total_profile = total_profile / total_profile.sum() * trips
            od_list.append({
                "origin_candidate": origin["candidate_id"],
                "dest_candidate": dest["candidate_id"],
                "mean_trips_per_day": trips,
                "hourly_profile": total_profile.tolist()
            })

station_od = pd.DataFrame(od_list)
station_od.to_csv("data/prepared/station_od.csv", index=False)
print(f"✅ Station OD demand created for {len(station_od)} OD pairs")
