# scripts/download_wmata_api.py
import requests, json, os
from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv()
KEY = os.getenv("WMATA_API_KEY")
HEAD = {"api_key": KEY}

os.makedirs("data/wmata/route_details", exist_ok=True)

# 1️⃣ Rail stations, lines, entrances
endpoints = {
    "stations": "https://api.wmata.com/Rail.svc/json/jStations",
    "lines": "https://api.wmata.com/Rail.svc/json/jLines",
    "entrances": "https://api.wmata.com/Rail.svc/json/jStationEntrances"
}
for name, url in endpoints.items():
    print(f"Fetching {name}...")
    r = requests.get(url, headers=HEAD)
    r.raise_for_status()
    with open(f"data/wmata/{name}.json", "w") as f:
        json.dump(r.json(), f, indent=2)

# 2️⃣ Bus stops and routes
print("Fetching bus stops...")
bus_stops = requests.get("https://api.wmata.com/Bus.svc/json/jStops", headers=HEAD).json()
with open("data/wmata/bus_stops.json", "w") as f:
    json.dump(bus_stops, f, indent=2)

print("Fetching bus routes list...")
bus_routes = requests.get("https://api.wmata.com/Bus.svc/json/jRoutes", headers=HEAD).json()["Routes"]
with open("data/wmata/bus_routes.json", "w") as f:
    json.dump(bus_routes, f, indent=2)

# 3️⃣ Route details
print("Fetching route details (this may take several minutes)...")
for r in tqdm(bus_routes):
    rid = r["RouteID"]
    detail_url = f"https://api.wmata.com/Bus.svc/json/jRouteDetails?RouteID={rid}"
    out = requests.get(detail_url, headers=HEAD).json()
    with open(f"data/wmata/route_details/{rid}.json", "w") as f:
        json.dump(out, f)
print("✅ Done fetching WMATA data.")
