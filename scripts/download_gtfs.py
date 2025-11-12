# scripts/download_gtfs_all.py
import requests, os
from dotenv import load_dotenv

load_dotenv()
KEY = os.getenv("WMATA_API_KEY")
HEAD = {"api_key": KEY}

os.makedirs("data/gtfs/bus", exist_ok=True)
os.makedirs("data/gtfs/rail", exist_ok=True)

feeds = {
    "bus": "https://api.wmata.com/gtfs/bus-gtfs-static.zip",
    "rail": "https://api.wmata.com/gtfs/rail-gtfs-static.zip"
}

for mode, url in feeds.items():
    print(f"Downloading {mode} GTFS...")
    r = requests.get(url, headers=HEAD)
    r.raise_for_status()
    with open(f"data/gtfs/{mode}/{mode}_gtfs_static.zip", "wb") as f:
        f.write(r.content)
    print(f"✅ Saved {mode} GTFS to data/gtfs/{mode}/")
