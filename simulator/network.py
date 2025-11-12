import networkx as nx
import pandas as pd

def load_network(network_path):
    return nx.read_gpickle(network_path)

def shortest_path_length(G, origin, dest):
    try:
        return nx.shortest_path_length(G, origin, dest, weight='travel_time_minutes')
    except:
        return float('inf')
