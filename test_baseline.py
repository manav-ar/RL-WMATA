from baselines import greedy, kmeans
print("Greedy:", greedy.greedy_baseline('data/prepared/candidates_final.geojson', M=5))
print("KMeans:", kmeans.kmeans_baseline('data/prepared/candidates_final.geojson', M=5))
