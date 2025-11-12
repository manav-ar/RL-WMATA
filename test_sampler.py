from simulator.demand import DemandSampler
ds = DemandSampler("data/prepared/station_od.csv")
trips = ds.sample_trips(10)
print(trips)
