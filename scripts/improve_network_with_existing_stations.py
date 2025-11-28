# scripts/improve_network_with_existing_stations.py
"""
Enhance network graph by integrating existing metro stations.
This creates a more realistic network that includes current WMATA infrastructure.
"""
import os
import json
import pickle
import math
import geopandas as gpd
import networkx as nx
import pandas as pd

def haversine(lon1, lat1, lon2, lat2):
    """Calculate distance in kilometers."""
    R = 6371  # Earth radius in km
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

# Load existing network
network_path = "data/prepared/network.pkl"
if os.path.exists(network_path):
    with open(network_path, "rb") as f:
        G = pickle.load(f)
else:
    G = nx.Graph()

# Load existing metro stations
stations_file = "data/wmata/stations.json"
with open(stations_file) as f:
    stations_data = json.load(f)["Stations"]

# Load existing metro station entrances (more detailed)
entrances_file = "data/dc_open/Metro_Station_Entrances_Regional.geojson"
existing_stations_gdf = gpd.read_file(entrances_file).to_crs("EPSG:4326")

# Add existing stations to graph if not already present
for s in stations_data:
    node_id = f"ST_{s['Code']}"
    if node_id not in G.nodes:
        G.add_node(
            node_id,
            type="existing_station",
            name=s["Name"],
            code=s["Code"],
            pos=(s["Lon"], s["Lat"]),
            lat=s["Lat"],
            lon=s["Lon"]
        )

# Load candidates
candidates_file = "data/prepared/candidates_final.geojson"
candidates = gpd.read_file(candidates_file)

# Connect candidates to existing stations within reasonable distance
connection_threshold_km = 2.0  # 2km connection threshold
existing_station_nodes = [n for n, d in G.nodes(data=True) if d.get("type") == "existing_station"]

for _, candidate in candidates.iterrows():
    candidate_id = candidate['candidate_id']
    candidate_lat = candidate['lat']
    candidate_lon = candidate['lon']
    
    # Find nearest existing station
    min_dist = float('inf')
    nearest_station = None
    
    for station_node in existing_station_nodes:
        station_data = G.nodes[station_node]
        dist = haversine(candidate_lon, candidate_lat, station_data['lon'], station_data['lat'])
        if dist < min_dist:
            min_dist = dist
            nearest_station = station_node
    
    # Connect if within threshold
    if nearest_station and min_dist <= connection_threshold_km:
        if not G.has_edge(candidate_id, nearest_station):
            # Weight by distance (in minutes, assuming 30 km/h average speed)
            travel_time = (min_dist / 30.0) * 60  # minutes
            G.add_edge(candidate_id, nearest_station, weight=travel_time, distance_km=min_dist)

# Save enhanced network
enhanced_network_path = "data/prepared/enhanced_network.pkl"
with open(enhanced_network_path, "wb") as f:
    pickle.dump(G, f)

print(f"✅ Enhanced network saved to {enhanced_network_path}")
print(f"   Total nodes: {G.number_of_nodes()}")
print(f"   Total edges: {G.number_of_edges()}")
print(f"   Existing stations: {len(existing_station_nodes)}")
print(f"   Candidates connected to existing stations: {sum(1 for n in G.nodes() if G.nodes[n].get('type') == 'candidate' and any(G.has_edge(n, s) for s in existing_station_nodes))}")

