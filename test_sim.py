from simulator.simulator_core import Simulator
sim = Simulator("data/prepared/multi_modal_graph.pkl", 
                "data/prepared/station_od.csv",
                "data/prepared/candidates_final.geojson")
metrics = sim.simulate_day(['C0','C1'])
print(metrics)
