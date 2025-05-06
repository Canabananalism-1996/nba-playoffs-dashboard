#!/usr/bin/env python3
"""
Fetch NBA data (teams, players, games, box scores) for seasons 2018 through current year.
Writes JSON files into data-raw/.
"""

import os
import time
import json
from datetime import datetime

from nba_api.stats.static import teams as teams_static
from nba_api.stats.static import players as players_static
from nba_api.stats.endpoints import leaguegamefinder, boxscoretraditionalv2
from requests.exceptions import ReadTimeout

# ─── CONFIG ──────────────────────────────────────────────────────────────────────
ROOT_DIR   = os.path.dirname(__file__)
OUT_DIR    = os.path.join(ROOT_DIR, "..", "data-raw")
START_YEAR = 2022
CURRENT_YEAR = datetime.now().year  # e.g. 2025
SEASONS    = [f"{yr}-{(yr+1)%100:02d}" for yr in range(START_YEAR, CURRENT_YEAR)]
PAUSE      = 1.5  # seconds between API calls
SKIPPED_FILE = os.path.join(OUT_DIR, "skipped_boxscores.txt")
# ────────────────────────────────────────────────────────────────────────────────

os.makedirs(OUT_DIR, exist_ok=True)


def save_json(obj, filename):
    path = os.path.join(OUT_DIR, filename)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)
    print(f">>> saved {filename}")


def fetch_teams():
    print("Fetching teams…")
    lst = teams_static.get_teams()
    save_json(lst, "teams.json")


def fetch_players():
    print("Fetching players…")
    lst = players_static.get_players()
    save_json(lst, "players.json")


def fetch_games(season):
    print(f"\nFetching games for {season}…")
    finder = leaguegamefinder.LeagueGameFinder(season_nullable=season)
    df = finder.get_data_frames()[0]
    # Extract unique game IDs
    game_ids = sorted(df["GAME_ID"].unique().tolist())
    save_json(game_ids, f"game_ids_{season}.json")
    return game_ids


def log_skipped(game_id):
    with open(SKIPPED_FILE, "a") as f:
        f.write(f"{game_id}\n")

def fetch_box_scores(game_ids):
    for gid in game_ids:
        outfile = f"boxscore_{gid}.json"
        outpath = os.path.join(OUT_DIR, outfile)
        # 1) Skip if we already have it
        if os.path.exists(outpath):
            continue

        # 2) Try up to 3 times, catching any network or HTTP error
        for attempt in range(1, 4):
            try:
                print(f"→ Fetching box score for game {gid} (attempt {attempt})")
                bs = boxscoretraditionalv2.BoxScoreTraditionalV2(
                    game_id=gid,
                    timeout=60
                )
                df = bs.get_data_frames()[0]
                records = df.to_dict(orient="records")
                with open(outpath, "w", encoding="utf-8") as f:
                    json.dump(records, f, indent=2)
                print(f"  ✅ Saved boxscore_{gid}.json")
                break

            except RequestException as e:
                wait = 10 * attempt
                print(f"  ⚠️  Error on game {gid}: {e.__class__.__name__}. "
                      f"Retrying in {wait}s…")
                time.sleep(wait)

        else:
            # after 3 attempts
            print(f"❌  Skipping game {gid} after 3 failed attempts")
            log_skipped(gid)

        # 3) Be polite
        time.sleep(PAUSE)

if __name__ == "__main__":
    # 1. Teams & players
    fetch_teams()
    fetch_players()

    # 2. Games & box scores per season
    for season in SEASONS:
        ids = fetch_games(season)
        fetch_box_scores(ids)
