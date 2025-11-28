# check_candidates_data.py
"""
Diagnostic script to check candidates data for issues.
"""
import pandas as pd
import geopandas as gpd
import numpy as np
import os

CANDIDATES_PATH = "data/prepared/candidates_final.geojson"
CSV_PATH = CANDIDATES_PATH.replace(".geojson", ".csv")

print("="*60)
print("Candidates Data Diagnostic")
print("="*60)

# Check GeoJSON
if os.path.exists(CANDIDATES_PATH):
    print(f"\n📄 Checking GeoJSON: {CANDIDATES_PATH}")
    gdf = gpd.read_file(CANDIDATES_PATH)
    print(f"   Total candidates: {len(gdf)}")
    print(f"   Columns: {list(gdf.columns)}")
    
    # Check if lat/lon exist as columns
    if 'lat' in gdf.columns and 'lon' in gdf.columns:
        print(f"   ✅ Has lat/lon columns")
        print(f"   Lat range: {gdf['lat'].min():.6f} to {gdf['lat'].max():.6f}")
        print(f"   Lon range: {gdf['lon'].min():.6f} to {gdf['lon'].max():.6f}")
        unique_coords = gdf[['lat', 'lon']].drop_duplicates()
        print(f"   Unique coordinates: {len(unique_coords)}/{len(gdf)}")
        
        if len(unique_coords) < len(gdf) * 0.5:
            print(f"   ⚠️  WARNING: Many duplicate coordinates!")
            print(f"   Sample duplicates:")
            print(gdf[['candidate_id', 'lat', 'lon']].head(10))
    else:
        print(f"   ⚠️  No lat/lon columns, extracting from geometry...")
        if gdf.geometry is not None:
            gdf['lat'] = gdf.geometry.y
            gdf['lon'] = gdf.geometry.x
            print(f"   Extracted lat/lon from geometry")
            print(f"   Lat range: {gdf['lat'].min():.6f} to {gdf['lat'].max():.6f}")
            print(f"   Lon range: {gdf['lon'].min():.6f} to {gdf['lon'].max():.6f}")
    
    # Check geometry
    if gdf.geometry is not None:
        print(f"   ✅ Has geometry column")
        sample_geom = gdf.geometry.iloc[0]
        print(f"   Sample geometry type: {type(sample_geom)}")
        print(f"   Sample coordinates: lat={sample_geom.y:.6f}, lon={sample_geom.x:.6f}")
else:
    print(f"❌ GeoJSON not found: {CANDIDATES_PATH}")

# Check CSV
if os.path.exists(CSV_PATH):
    print(f"\n📄 Checking CSV: {CSV_PATH}")
    df = pd.read_csv(CSV_PATH)
    print(f"   Total candidates: {len(df)}")
    print(f"   Columns: {list(df.columns)}")
    
    if 'lat' in df.columns and 'lon' in df.columns:
        print(f"   ✅ Has lat/lon columns")
        print(f"   Lat range: {df['lat'].min():.6f} to {df['lat'].max():.6f}")
        print(f"   Lon range: {df['lon'].min():.6f} to {df['lon'].max():.6f}")
        unique_coords = df[['lat', 'lon']].drop_duplicates()
        print(f"   Unique coordinates: {len(unique_coords)}/{len(df)}")
        
        if len(unique_coords) < len(df) * 0.5:
            print(f"   ⚠️  WARNING: Many duplicate coordinates in CSV!")
            print(f"   First 10 rows:")
            print(df[['candidate_id', 'lat', 'lon', 'population']].head(10))
        
        # Check for invalid coordinates
        invalid = df[(df['lat'] < 38.5) | (df['lat'] > 39.2) | 
                    (df['lon'] < -77.2) | (df['lon'] > -76.8)]
        if len(invalid) > 0:
            print(f"   ⚠️  {len(invalid)} candidates have coordinates outside DC area")
    else:
        print(f"   ❌ Missing lat/lon columns in CSV!")
else:
    print(f"⚠️  CSV not found: {CSV_PATH}")

print("\n" + "="*60)

