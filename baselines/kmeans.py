import geopandas as gpd
from sklearn.cluster import KMeans

def kmeans_baseline(candidates_path, M=5):
    """
    Place M new stations using k-means on high-density population areas.
    """
    # Read candidates GeoJSON
    gdf = gpd.read_file(candidates_path)

    # Ensure candidate_id exists
    if 'candidate_id' not in gdf.columns:
        gdf['candidate_id'] = [f"C{i}" for i in range(len(gdf))]

    # Use lat/lon as features for clustering
    coords = gdf[['geometry']].apply(lambda row: [row.geometry.y, row.geometry.x], axis=1).tolist()
    
    # KMeans clustering
    kmeans = KMeans(n_clusters=M, random_state=42, n_init=10)
    labels = kmeans.fit_predict(coords)

    # Pick candidate closest to each cluster center
    selected = []
    for i in range(M):
        cluster_points = gdf.iloc[[j for j, lab in enumerate(labels) if lab == i]]
        # Compute distance to cluster center
        center = kmeans.cluster_centers_[i]
        cluster_points['dist_to_center'] = cluster_points['geometry'].apply(
            lambda p: (p.y - center[0])**2 + (p.x - center[1])**2
        )
        nearest = cluster_points.sort_values('dist_to_center').iloc[0]['candidate_id']
        selected.append(nearest)
    
    return selected
