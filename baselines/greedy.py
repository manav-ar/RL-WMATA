import geopandas as gpd

def greedy_baseline(candidates_path, M=5):
    # Read GeoJSON instead of CSV
    gdf = gpd.read_file(candidates_path)
    
    # Ensure candidate_id column exists
    if 'candidate_id' not in gdf.columns:
        gdf['candidate_id'] = [f"C{i}" for i in range(len(gdf))]

    selected = []
    # simple greedy: pick top M by population
    gdf_sorted = gdf.sort_values("population", ascending=False)
    for i in range(M):
        selected.append(gdf_sorted.iloc[i]['candidate_id'])
    
    return selected
