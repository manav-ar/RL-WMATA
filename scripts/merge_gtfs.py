# scripts/merge_gtfs_data.py
import pandas as pd
import os

def load_gtfs(mode):
    folder = f"data/gtfs/{mode}"
    stops = pd.read_csv(os.path.join(folder, "stops.txt"))
    routes = pd.read_csv(os.path.join(folder, "routes.txt"))
    stops["mode"] = mode
    routes["mode"] = mode
    return stops, routes

bus_stops, bus_routes = load_gtfs("bus")
rail_stops, rail_routes = load_gtfs("rail")

all_stops = pd.concat([bus_stops, rail_stops])
all_routes = pd.concat([bus_routes, rail_routes])

os.makedirs("data/prepared", exist_ok=True)
all_stops.to_csv("data/prepared/all_stops.csv", index=False)
all_routes.to_csv("data/prepared/all_routes.csv", index=False)

print("✅ Combined stops and routes saved to data/prepared/")


