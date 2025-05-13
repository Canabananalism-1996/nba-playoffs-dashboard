import json, glob
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data-raw"

# 1) Load all metadata into one dict (GAME_ID → first record found)
meta_map = {}
for path in glob.glob(f"{DATA_DIR}/games_meta_*.json"):
    for record in json.load(open(path, 'r')):
        gid = record["GAME_ID"]
        # only store the first occurrence
        if gid not in meta_map:
            meta_map[gid] = record

# 2) Read skipped GAME_IDs
skipped_path = DATA_DIR / "skipped_boxscores.txt"
skipped = [line.strip() for line in open(skipped_path) if line.strip()]

# 3) Print cross‐reference
print("Cross‐referenced skipped games:\n")
for gid in skipped:
    info = meta_map.get(gid)
    if info:
        # Only use fields that exist:
        date    = info.get("GAME_DATE")
        matchup = info.get("MATCHUP")
        pts     = info.get("PTS")
        wl      = info.get("WL")
        print(f"{gid}: {date} — {matchup} — {pts} pts  ({'Win' if wl=='W' else 'Loss'})")
    else:
        print(f"{gid}: ❌ metadata not found")
