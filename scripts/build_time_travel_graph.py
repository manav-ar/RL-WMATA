# scripts/build_travel_time_graph.py
import pandas as pd
import networkx as nx
import pickle
import zipfile

# Load GTFS rail static feed
gtfs_zip = "data/gtfs/rail/rail_gtfs_static.zip"
with zipfile.ZipFile(gtfs_zip) as z:
    stops = pd.read_csv(z.open("stops.txt"))
    stop_times = pd.read_csv(z.open("stop_times.txt"))
    trips = pd.read_csv(z.open("trips.txt"))

# Merge stop_times with trips
stop_times = stop_times.merge(trips[["trip_id","route_id"]], on="trip_id", how="left")

# Build DiGraph
G = nx.DiGraph()

# Add nodes
for _, row in stops.iterrows():
    G.add_node(row["stop_id"], name=row["stop_name"], pos=(row["stop_lon"], row["stop_lat"]))

# Add edges based on stop sequence and scheduled times
for trip_id, group in stop_times.groupby("trip_id"):
    group = group.sort_values("stop_sequence")
    for i in range(len(group)-1):
        u = group.iloc[i]["stop_id"]
        v = group.iloc[i+1]["stop_id"]
        t1 = group.iloc[i]["arrival_time"]
        t2 = group.iloc[i+1]["arrival_time"]
        # convert HH:MM:SS to minutes
        h1,m1,s1 = map(int,t1.split(":"))
        h2,m2,s2 = map(int,t2.split(":"))
        travel_time = (h2*60+m2+s2/60) - (h1*60+m1+s1/60)
        if travel_time < 0:  # handle midnight crossing
            travel_time += 24*60
        if not G.has_edge(u,v):
            G.add_edge(u,v,weight=travel_time)
        else:
            # keep minimum travel time
            G[u][v]["weight"] = min(G[u][v]["weight"], travel_time)

# Save
with open("data/prepared/network.pkl", "wb") as f:
    pickle.dump(G, f)
print("✅ Travel-time graph saved")
