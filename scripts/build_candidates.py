# scripts/build_candidates_final.py
import osmnx as ox
import geopandas as gpd
from shapely.geometry import Point
import pandas as pd
import numpy as np

# -----------------------------
# Load DC boundary
# -----------------------------
city = "Washington, D.C., USA"
boundary = ox.geocode_to_gdf(city)
boundary.to_file("data/dc_open/dc_boundary.geojson", driver="GeoJSON")

# -----------------------------
# Load population centroids
# -----------------------------
pop = pd.read_csv("data/census/dc_population.csv")
# For now, randomize points inside blocks (replace with true lat/lon if available)
pop["geometry"] = pop.apply(lambda r: Point(np.random.uniform(-77.12, -76.90),
                                            np.random.uniform(38.80, 38.98)), axis=1)
gdf = gpd.GeoDataFrame(pop, geometry="geometry", crs="EPSG:4326")

# Filter points inside DC
gdf = gpd.sjoin(gdf, boundary, predicate="within")

# -----------------------------
# Load existing metro stations
# -----------------------------
stations = gpd.read_file("data/dc_open/Metro_Station_Entrances_Regional.geojson").to_crs("EPSG:4326")

# Compute distance to nearest station (in meters)
def nearest_station_distance(candidate, stations):
    nearest_geom = stations.geometry.iloc[stations.geometry.distance(candidate.geometry).argmin()]
    return candidate.geometry.distance(nearest_geom) * 111000  # approx meters

gdf["distance_to_nearest_station"] = gdf.apply(
    lambda row: nearest_station_distance(row, stations), axis=1
)

# Filter out candidates too close (<400 m)
gdf = gdf[gdf["distance_to_nearest_station"] > 400]

# -----------------------------
# Keep top population candidates
# -----------------------------
top_candidates = gdf.nlargest(500, "population").copy()
top_candidates["candidate_id"] = [f"C{i}" for i in range(len(top_candidates))]

# Save
top_candidates.to_file("data/prepared/candidates_final.geojson", driver="GeoJSON")
print(f"✅ {len(top_candidates)} candidate station sites saved to candidates_final.geojson")
