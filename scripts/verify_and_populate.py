#!/usr/bin/env python3
import os
import json
from fetch_nba_data import save_json, ROOT_DIR
from nba_api.stats.endpoints import (
    BoxScoreTraditionalV2,
    BoxScoreAdvancedV2,
    BoxScoreScoringV2,
    BoxScoreMiscV2,
    BoxScoreFourFactorsV2,
    BoxScoreDefensiveV2
)

# ---- Configuration ----
OUT_DIR    = os.path.abspath(os.path.join(ROOT_DIR, '..', 'data-raw'))
SKIP_FILE  = os.path.join(OUT_DIR, 'skipped_boxscores.txt')
STATS_FILE = os.path.join(OUT_DIR, 'player_stats_2024-25.json')

# ---- Helpers ----
def load_skip_list():
    if not os.path.exists(SKIP_FILE):
        return []
    with open(SKIP_FILE, 'r') as f:
        return [line.strip() for line in f if line.strip()]


def write_skip_list(ids):
    with open(SKIP_FILE, 'w') as f:
        if ids:
            f.write("\n".join(ids) + "\n")
        else:
            f.truncate(0)


def load_stats():
    try:
        return json.load(open(STATS_FILE, 'r'))
    except (FileNotFoundError, json.JSONDecodeError):
        return []


def fetch_via_all_endpoints(game_id):
    """
    Attempt multiple boxscore endpoints in order:
      Traditional, Advanced, Scoring, Misc, FourFactors, Defensive, PlayerV2
    Returns list of records or [] if none.
    """
    endpoints = [
        ('Traditional',    BoxScoreTraditionalV2),
        ('Advanced',       BoxScoreAdvancedV2),
        ('Scoring',        BoxScoreScoringV2),
        ('Misc',           BoxScoreMiscV2),
        ('FourFactors',    BoxScoreFourFactorsV2),
        ('Defensive',      BoxScoreDefensiveV2)
    ]
    for name, Endpoint in endpoints:
        try:
            df = Endpoint(game_id=game_id).get_data_frames()[0]
            if not df.empty:
                records = df.to_dict('records')
                for rec in records:
                    rec['GAME_ID'] = game_id
                print(f"   ✅ {name} endpoint: {len(records)} rows for {game_id}")
                return records
            else:
                print(f"   ⚠️ {name} endpoint returned empty for {game_id}")
        except Exception as e:
            print(f"   ⚠️ {name} endpoint error for {game_id}: {e}")
    return []

# ---- Main ----
def main():
    skipped = load_skip_list()
    # filter for 2024-25 games (prefix '00224')
    target_ids = [gid for gid in skipped if gid.startswith('00224')]
    if not target_ids:
        print("✅ No 2024-25 GAME_IDs in skip list.")
        return

    stats = load_stats()
    fetched = []

    for gid in target_ids:
        print(f"🔄 Attempting {gid}...")
        recs = fetch_via_all_endpoints(gid)
        if recs:
            stats.extend(recs)
            save_json(stats, STATS_FILE)
            fetched.append(gid)
        else:
            print(f"   ❌ No data for {gid} from any endpoint.")

    # prune skip list
    remaining = [gid for gid in skipped if gid not in fetched]
    write_skip_list(remaining)
    print(f"🎉 Done: {len(fetched)} fetched, {len(remaining)} remain.")

if __name__ == '__main__':
    main()
