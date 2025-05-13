import os
import json
import glob

from fetch_nba_data import fetch_scores, save_json, ROOT_DIR

# ---- adjust these if your layout differs ----
OUT_DIR      = os.path.abspath(os.path.join(ROOT_DIR, "..", "data-raw"))
SKIPPED_FILE = os.path.join(OUT_DIR, "skipped_boxscores.txt")

def season_str_from_game_id(gid):
    """
    NBA game IDs start with the 4-digit season year (e.g. "2021").
    We turn 2021 -> "2021-22", 2022 -> "2022-23", etc.
    """
    y = int(gid[:4])
    return f"{y}-{(y+1) % 100:02d}"

def load_ids(fn):
    """Load a list‐JSON and return the set of its GAME_IDs, or empty if missing/corrupt."""
    try:
        data = json.load(open(fn, "r"))
    except (FileNotFoundError, json.JSONDecodeError):
        return set()
    if isinstance(data, list):
        return {rec["GAME_ID"] for rec in data if "GAME_ID" in rec}
    return set()

def all_existing_ids_for(season):
    """
    Glob every JSON slice for this season across
      game_ids_{season}*.json
      player_stats_{season}*.json
      games_meta_{season}*.json
    and merge their GAME_ID sets.
    """
    patterns = [
        os.path.join(OUT_DIR, f"game_ids_{season}*.json"),
        os.path.join(OUT_DIR, f"player_stats_{season}*.json"),
        os.path.join(OUT_DIR, f"games_meta_{season}*.json"),
    ]
    ids = set()
    for pat in patterns:
        for fn in glob.glob(pat):
            ids |= load_ids(fn)
    return ids

def retry_skipped():
    # 1) load skip list
    with open(SKIPPED_FILE, "r") as f:
        skipped = [line.strip() for line in f if line.strip()]

    # 2) process each GAME_ID
    for gid in skipped[:]:
        season = season_str_from_game_id(gid)
        existing = all_existing_ids_for(season)

        if gid in existing:
            print(f"✅ {gid} already in JSONs for {season}, removing from skip list.")
            skipped.remove(gid)
            continue

        print(f"🔄 retrying fetch for {gid} …")
        recs = fetch_scores([gid])
        if recs:
            # append into your main player_stats_{season}.json
            out_fn = os.path.join(OUT_DIR, f"player_stats_{season}.json")
            try:
                master = json.load(open(out_fn, "r"))
            except (FileNotFoundError, json.JSONDecodeError):
                master = []
            master.extend(recs)
            save_json(master, out_fn)
            print(f"   💾 saved {gid} to {out_fn}")
            skipped.remove(gid)
        else:
            print(f"   ⚠️ still missing data for {gid}, keeping in skip list.")

    # 3) rewrite skip file
    with open(SKIPPED_FILE, "w") as f:
        if skipped:
            f.write("\n".join(skipped) + "\n")
        else:
            f.truncate(0)

if __name__ == "__main__":
    retry_skipped()
