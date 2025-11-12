# scripts/build_graph.py
import os
import json
import pickle
import math
import geopandas as gpd
from shapely.geometry import Point
import networkx as nx

# -----------------------------
# Helper: haversine distance
# -----------------------------
def haversine(lon1, lat1, lon2, lat2):
    R = 6371  # Earth radius in km
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

# -----------------------------
# Create directories
# -----------------------------
os.makedirs("data/prepared", exist_ok=True)

# -----------------------------
# Load WMATA data
# -----------------------------
stations_file = "data/wmata/stations.json"
bus_stops_file = "data/wmata/bus_stops.json"
candidates_file = "data/prepared/candidates.geojson"

with open(stations_file) as f:
    stations = json.load(f)["Stations"]

with open(bus_stops_file) as f:
    bus_stops = json.load(f)["Stops"]

candidates = gpd.read_file(candidates_file)

# -----------------------------
# Build graph
# -----------------------------
G = nx.Graph()

# Add metro stations
for s in stations:
    G.add_node(
        f"ST_{s['Code']}",
        type="station",
        name=s["Name"],
        pos=(s["Lon"], s["Lat"])
    )

# Add bus stops
for b in bus_stops:
    G.add_node(
        f"BS_{b['StopID']}",
        type="bus",
        name=b["Name"],
        pos=(b["Lon"], b["Lat"])
    )

# Add candidate sites
for _, r in candidates.iterrows():
    G.add_node(
        r["candidate_id"],
        type="candidate",
        pos=(r.geometry.x, r.geometry.y)
    )

# -----------------------------
# Connect nodes within threshold
# -----------------------------
nodes = list(G.nodes(data=True))
distance_threshold_km = 0.5  # connect nodes within 500 meters

for i, (n1, d1) in enumerate(nodes):
    for n2, d2 in nodes[i + 1:]:
        # Optionally: avoid connecting candidate->candidate
        if d1["type"] == "candidate" and d2["type"] == "candidate":
            continue
        lon1, lat1 = d1["pos"]
        lon2, lat2 = d2["pos"]
        dist = haversine(lon1, lat1, lon2, lat2)
        if dist <= distance_threshold_km:
            G.add_edge(n1, n2, weight=dist)

# -----------------------------
# Save graph using pickle
# -----------------------------
graph_path = "data/prepared/multi_modal_graph.pkl"
with open(graph_path, "wb") as f:
    pickle.dump(G, f)

print(f"✅ Multi-modal graph saved to {graph_path}")
