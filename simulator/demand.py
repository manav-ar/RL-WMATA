import pandas as pd
import numpy as np

class DemandSampler:
    def __init__(self, od_path):
        self.station_od = pd.read_csv(od_path)
    
    def sample_trips(self, n_trips=1000):
        """
        Returns a list of sampled trips (origin, destination)
        """
        od = self.station_od
        sampled = od.sample(n=n_trips, replace=True, weights='mean_trips_per_day')
        return list(zip(sampled.origin_candidate, sampled.dest_candidate))
