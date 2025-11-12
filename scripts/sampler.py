# scripts/demand_sampler.py
import pandas as pd
import numpy as np

station_od = pd.read_csv("data/prepared/station_od.csv")

def sample_trips(hour, n=100):
    """
    Sample n trips for a given hour
    Returns list of (origin_candidate, dest_candidate, departure_hour)
    """
    trips = []
    for _, row in station_od.iterrows():
        # pick number of trips in this hour based on hourly_profile
        hourly = eval(row["hourly_profile"])  # convert string to list if saved as CSV
        prob = hourly[hour] / sum(hourly)
        num_trips = np.random.binomial(n, prob)
        trips += [(row["origin_candidate"], row["dest_candidate"], hour)] * num_trips
    return trips
