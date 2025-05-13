import os, json, glob, requests
from fetch_nba_data import ROOT_DIR

DATA_DIR = os.path.abspath(os.path.join(ROOT_DIR, "..", "data-raw"))
season   = "2024-25"

# Build GAME_ID → YYYYMMDD map from your games_meta files
id2date = {}
for fn in glob.glob(os.path.join(DATA_DIR, f"games_meta_{season}*.json")):
    with open(fn) as f:
        data = json.load(f)
    if isinstance(data, list):
        for rec in data:
            # Now rec is a dict
            gid = rec.get("GAME_ID")
            dt  = rec.get("GAME_DATE_EST")  # e.g. "2024-10-24T00:00:00Z"
            if gid and dt:
                id2date[gid] = dt.split("T")[0].replace("-", "")

# Then check your missing IDs:
missing = ["0022400118", "0022400120", ...]  # your 34 IDs

for gid in missing:
    date = id2date.get(gid)
    if not date:
        print(f"{gid}: ❌ no date found in games_meta → can't form URL")
        continue

    url = f"https://data.nba.com/data/10s/prod/v1/{date}/{gid}_boxscore.json"
    resp = requests.get(url)
    if resp.status_code == 200:
        players = resp.json().get("stats", {}).get("activePlayers", [])
        if players:
            print(f"{gid}: ✅ {len(players)} players")
        else:
            print(f"{gid}: ⚠️ no activePlayers")
    else:
        print(f"{gid}: ❌ HTTP {resp.status_code}")
