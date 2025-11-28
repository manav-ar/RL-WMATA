# fix_candidates_data.py
"""
Fix candidates data by regenerating CSV from GeoJSON with correct coordinates.
"""
import geopandas as gpd
import pandas as pd
import os

CANDIDATES_PATH = "data/prepared/candidates_final.geojson"
CSV_PATH = CANDIDATES_PATH.replace(".geojson", ".csv")

print("="*60)
print("Fixing Candidates Data")
print("="*60)

if not os.path.exists(CANDIDATES_PATH):
    print(f"❌ GeoJSON not found: {CANDIDATES_PATH}")
    exit(1)

# Load GeoJSON
print(f"\n📄 Loading GeoJSON: {CANDIDATES_PATH}")
gdf = gpd.read_file(CANDIDATES_PATH)
print(f"   Loaded {len(gdf)} candidates")

# Extract lat/lon from geometry (always use geometry as source of truth)
if gdf.geometry is not None:
    print("   Extracting coordinates from geometry...")
    gdf['lon'] = gdf.geometry.x
    gdf['lat'] = gdf.geometry.y
    print(f"   ✅ Extracted lat/lon from geometry")
else:
    print("   ❌ No geometry column found!")
    exit(1)

# Validate required columns
required_cols = ['candidate_id', 'lat', 'lon', 'population']
missing_cols = [col for col in required_cols if col not in gdf.columns]
if missing_cols:
    print(f"   ❌ Missing required columns: {missing_cols}")
    exit(1)

# Check for duplicate coordinates
unique_coords = gdf[['lat', 'lon']].drop_duplicates()
print(f"\n📊 Coordinate Analysis:")
print(f"   Total candidates: {len(gdf)}")
print(f"   Unique coordinates: {len(unique_coords)}")
print(f"   Duplicate rate: {(1 - len(unique_coords)/len(gdf))*100:.1f}%")

if len(unique_coords) < len(gdf) * 0.5:
    print(f"\n⚠️  WARNING: Many duplicate coordinates detected!")
    print(f"   This will cause all selected stations to have same coordinates.")
    print(f"   Sample candidates with coordinates:")
    print(gdf[['candidate_id', 'lat', 'lon', 'population']].head(10))
    
    # Show coordinate distribution
    coord_counts = gdf.groupby(['lat', 'lon']).size().sort_values(ascending=False)
    print(f"\n   Top duplicate coordinates:")
    print(coord_counts.head(5))
else:
    print(f"   ✅ Good coordinate diversity")

# Validate coordinate ranges (DC area)
print(f"\n📍 Coordinate Ranges:")
print(f"   Lat: {gdf['lat'].min():.6f} to {gdf['lat'].max():.6f}")
print(f"   Lon: {gdf['lon'].min():.6f} to {gdf['lon'].max():.6f}")

invalid = gdf[
    (gdf['lat'] < 38.5) | (gdf['lat'] > 39.2) |
    (gdf['lon'] < -77.2) | (gdf['lon'] > -76.8)
]
if len(invalid) > 0:
    print(f"   ⚠️  {len(invalid)} candidates outside DC area")

# Save to CSV (overwrite existing)
print(f"\n💾 Saving to CSV: {CSV_PATH}")
output_df = gdf[required_cols].copy()
output_df.to_csv(CSV_PATH, index=False)
print(f"   ✅ Saved {len(output_df)} candidates to CSV")

# Verify the saved CSV
print(f"\n✅ Verification:")
df_check = pd.read_csv(CSV_PATH)
print(f"   CSV has {len(df_check)} candidates")
print(f"   Unique coordinates in CSV: {df_check[['lat', 'lon']].drop_duplicates().shape[0]}")
print(f"   Sample from CSV:")
print(df_check[['candidate_id', 'lat', 'lon', 'population']].head(5))

print("\n" + "="*60)
print("✅ Candidates data fixed!")
print("="*60)
print("\nNext steps:")
print("1. Retrain your models with the fixed data")
print("2. The models should now select candidates with different coordinates")
print("3. Run: python check_candidates_data.py to verify")

