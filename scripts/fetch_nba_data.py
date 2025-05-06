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


def fetch_box_scores(game_ids):
    for gid in game_ids:
        outfile = f"boxscore_{gid}.json"
        if os.path.exists(os.path.join(OUT_DIR, outfile)):
            continue

        attempts, success = 0, False
        while not success and attempts < 3:
            try:
                attempts += 1
                print(f"→ Fetching box score for game {gid} (attempt {attempts})")
                bs = boxscoretraditionalv2.BoxScoreTraditionalV2(
                    game_id=gid,
                    timeout=60
                )
                df = bs.get_data_frames()[0]
                records = df.to_dict(orient="records")
                save_json(records, outfile)
                success = True
            except ReadTimeout:
                print(f"⚠️ ReadTimeout on game {gid}, retrying in 10s…")
                time.sleep(10)

        if not success:
            print(f"❌ Skipping game {gid} after 3 failed attempts")
        time.sleep(PAUSE)

if __name__ == "__main__":
    # 1. Teams & players
    fetch_teams()
    fetch_players()

    # 2. Games & box scores per season
    for season in SEASONS:
        ids = fetch_games(season)
        fetch_box_scores(ids)
