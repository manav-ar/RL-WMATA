# scripts/enrich_candidates_with_existing_stations.py
"""
Enrich candidate data with information about existing metro stations.
Adds distance to nearest existing station, which can be used as a feature.
"""
import geopandas as gpd
import pandas as pd
import numpy as np
import warnings

# Suppress the CRS warning since we're handling it properly
warnings.filterwarnings('ignore', category=UserWarning, message='.*Geometry is in a geographic CRS.*')

# Load candidates
candidates_file = "data/prepared/candidates_final.geojson"
candidates = gpd.read_file(candidates_file)

# Load existing metro stations
existing_stations_file = "data/dc_open/Metro_Station_Entrances_Regional.geojson"
existing_stations = gpd.read_file(existing_stations_file)

# Ensure both are in EPSG:4326 first
if candidates.crs is None:
    candidates.set_crs("EPSG:4326", inplace=True)
if existing_stations.crs is None:
    existing_stations.set_crs("EPSG:4326", inplace=True)

# Project to UTM Zone 18N (covers DC area) for accurate distance calculations
# EPSG:32618 is UTM Zone 18N (meters) - this eliminates the CRS warning
UTM_CRS = "EPSG:32618"
candidates_proj = candidates.to_crs(UTM_CRS)
existing_stations_proj = existing_stations.to_crs(UTM_CRS)

# Calculate distance to nearest existing station for each candidate
def distance_to_nearest_station(candidate_geom, stations_gdf):
    """Calculate distance in meters to nearest existing station."""
    # Distance is now in meters directly (no conversion needed) - no CRS warning!
    distances = stations_gdf.geometry.distance(candidate_geom)
    nearest_distance_m = distances.min()
    return nearest_distance_m

# Apply distance calculation in projected CRS (no warnings now)
candidates_proj['distance_to_nearest_existing_station'] = candidates_proj.geometry.apply(
    lambda geom: distance_to_nearest_station(geom, existing_stations_proj)
)

# Convert back to EPSG:4326 for saving
candidates_enriched = candidates_proj.to_crs("EPSG:4326")

# Add population density (if we have area data, otherwise use a proxy)
candidates_enriched['population_density'] = candidates_enriched['population']  # Can be improved with actual area data

# Ensure lat/lon are extracted from geometry for CSV export
if 'lat' not in candidates_enriched.columns or 'lon' not in candidates_enriched.columns:
    candidates_enriched['lon'] = candidates_enriched.geometry.x
    candidates_enriched['lat'] = candidates_enriched.geometry.y

# Save enriched candidates
output_file = "data/prepared/candidates_enriched.geojson"
candidates_enriched.to_file(output_file, driver="GeoJSON")

# Also save as CSV (ensure lat/lon are included)
csv_columns = ['candidate_id', 'lat', 'lon', 'population', 'distance_to_nearest_existing_station', 'population_density']
if all(col in candidates_enriched.columns for col in csv_columns):
    candidates_csv = candidates_enriched[csv_columns].copy()
else:
    # Fallback: drop geometry and keep all other columns
    candidates_csv = candidates_enriched.drop(columns=['geometry'] if 'geometry' in candidates_enriched.columns else [])
candidates_csv.to_csv("data/prepared/candidates_enriched.csv", index=False)

print(f"✅ Enriched candidates saved to {output_file}")
print(f"   Added columns: distance_to_nearest_existing_station, population_density")
print(f"   Min distance to existing station: {candidates_enriched['distance_to_nearest_existing_station'].min():.0f}m")
print(f"   Max distance to existing station: {candidates_enriched['distance_to_nearest_existing_station'].max():.0f}m")
print(f"   Mean distance to existing station: {candidates_enriched['distance_to_nearest_existing_station'].mean():.0f}m")