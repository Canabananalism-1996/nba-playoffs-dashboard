#!/usr/bin/env python3
"""
Fetch NBA data (teams, players, games, box scores) for seasons 2022 through current year.
If the script crashes, it waits, logs the error, and restarts—picking up where it left off.
"""

import os
import time
import json
import traceback
from datetime import datetime

from nba_api.stats.static import teams as teams_static
from nba_api.stats.static import players as players_static
from nba_api.stats.endpoints import leaguegamefinder, boxscoretraditionalv2
from requests.exceptions import RequestException

# ─── CONFIG ──────────────────────────────────────────────────────────────────────
ROOT_DIR      = os.path.dirname(os.path.abspath(__file__))
OUT_DIR       = os.path.join(ROOT_DIR, "..", "data-raw")
START_YEAR    = 2022
CURRENT_YEAR  = datetime.now().year  # e.g. 2025

# Generate seasons list - with logic for current year and NBA season
# NBA seasons typically run from October to June
current_month = datetime.now().month
# If we're in the early months of the year (Jan-June), we're in the latter part of a season
# If we're in the later months (July-Dec), we're in the early part of a new season
if current_month >= 7:  # July onwards - include current year as start of season
    SEASONS = [f"{yr}-{(yr+1)%100:02d}" for yr in range(START_YEAR, CURRENT_YEAR + 1)]
else:  # Jan-June - don't include current year as start of season
    SEASONS = [f"{yr}-{(yr+1)%100:02d}" for yr in range(START_YEAR, CURRENT_YEAR)]

PAUSE         = 1.5   # seconds between successful calls
SKIPPED_FILE  = os.path.join(OUT_DIR, "skipped_boxscores.txt")
FATAL_LOG     = os.path.join(OUT_DIR, "fatal_errors.log")
# ────────────────────────────────────────────────────────────────────────────────

def ensure_dir_exists(path):
    """Create directory if it doesn't exist"""
    if not os.path.exists(path):
        try:
            os.makedirs(path, exist_ok=True)
            print(f"Created directory: {path}")
        except Exception as e:
            print(f"ERROR: Could not create directory {path}: {e}")
            raise


def save_json(obj, filename):
    """Save object as JSON to the output directory"""
    path = os.path.join(OUT_DIR, filename)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)
    print(f">>> saved {filename}")


def fetch_teams():
    """Fetch all NBA teams"""
    print("Fetching teams…")
    lst = teams_static.get_teams()
    if not lst:
        print("WARNING: No teams data retrieved!")
        return
    save_json(lst, "teams.json")


def fetch_players():
    """Fetch all NBA players"""
    print("Fetching players…")
    lst = players_static.get_players()
    if not lst:
        print("WARNING: No players data retrieved!")
        return
    save_json(lst, "players.json")


def fetch_games(season):
    """Fetch full game metadata and unique IDs for a given season."""
    print(f"\nFetching games for {season}…")
    
    # Paths
    ids_file   = f"game_ids_{season}.json"
    meta_file  = f"games_meta_{season}.json"
    ids_path   = os.path.join(OUT_DIR, ids_file)
    meta_path  = os.path.join(OUT_DIR, meta_file)
    
    # 1) If metadata exists, load and return IDs
    if os.path.exists(meta_path):
        print(f"  → Loading existing metadata for {season}")
        games_meta = json.load(open(meta_path, 'r'))
        game_ids   = sorted({g["GAME_ID"] for g in games_meta})
        return game_ids

    # 2) Otherwise, fetch fresh
    try:
        finder = leaguegamefinder.LeagueGameFinder(season_nullable=season)
        df     = finder.get_data_frames()[0]
        
        if df.empty:
            print(f"WARNING: No games found for season {season}")
            return []
        
        # Convert DataFrame to list of dicts
        games_meta = df.to_dict(orient="records")
        # Save full metadata
        save_json(games_meta, meta_file)
        
        # Extract unique IDs
        game_ids = sorted({row["GAME_ID"] for row in games_meta})
        save_json(game_ids, ids_file)
        return game_ids

    except Exception as e:
        print(f"ERROR fetching games for season {season}: {e}")
        return []



def log_skipped(game_id):
    """Log skipped game IDs for later retry"""
    with open(SKIPPED_FILE, "a") as f:
        f.write(f"{game_id}\n")


def fetch_box_scores(game_ids):
    """Fetch box scores for a list of game IDs"""
    if not game_ids:
        print("No game IDs provided to fetch box scores")
        return
        
    print(f"Fetching box scores for {len(game_ids)} games...")
    
    for gid in game_ids:
        outfile = f"boxscore_{gid}.json"
        outpath = os.path.join(OUT_DIR, outfile)

        # Skip if already fetched
        if os.path.exists(outpath):
            continue

        # Retry up to 3 times on any network/HTTP error
        for attempt in range(1, 4):
            try:
                print(f"→ Fetching box score for game {gid} (attempt {attempt})")
                bs = boxscoretraditionalv2.BoxScoreTraditionalV2(
                    game_id=gid,
                    timeout=60
                )
                # Get all data frames - box score traditional returns multiple tables
                dfs = bs.get_data_frames()
                
                # Process and save all returned data frames 
                result = {}
                for i, df in enumerate(dfs):
                    if not df.empty:
                        result[f"table_{i}"] = df.to_dict(orient="records")
                
                if not result:
                    print(f"WARNING: No box score data found for game {gid}")
                    log_skipped(gid)
                    break
                    
                save_json(result, outfile)
                break
            except RequestException as e:
                wait = 10 * attempt
                print(f"  ⚠️ Error on game {gid}: {e.__class__.__name__}. "
                      f"Retrying in {wait}s…")
                time.sleep(wait)
            except Exception as e:
                print(f"  ❌ Unexpected error on game {gid}: {e.__class__.__name__}: {str(e)}")
                log_skipped(gid)
                break
        else:
            # all attempts failed
            print(f"❌ Skipping game {gid} after 3 failed attempts")
            log_skipped(gid)

        # pause between games
        time.sleep(PAUSE)


def process_skipped_games():
    """Retry processing any previously skipped games"""
    if not os.path.exists(SKIPPED_FILE):
        return
        
    print("\nRetrying previously skipped games...")
    with open(SKIPPED_FILE, 'r') as f:
        skipped_ids = [line.strip() for line in f if line.strip()]
    
    if skipped_ids:
        # Create a backup of the skipped file before clearing it
        backup_file = f"{SKIPPED_FILE}.bak"
        with open(backup_file, 'w') as f:
            f.write('\n'.join(skipped_ids))
            
        # Clear the skipped file
        open(SKIPPED_FILE, 'w').close()
        
        # Retry fetching
        fetch_box_scores(skipped_ids)
    else:
        print("No skipped games to retry")


def main():
    """Main execution function"""
    # Ensure output directory exists
    ensure_dir_exists(OUT_DIR)
    
    # 1. Teams & players
    fetch_teams()
    fetch_players()

    # 2. Games & box scores per season
    for season in SEASONS:
        ids = fetch_games(season)
        fetch_box_scores(ids)
    
    # 3. Retry any skipped games
    process_skipped_games()


if __name__ == "__main__":
    print(f"NBA Data Fetch - Starting at {datetime.now()}")
    print(f"Fetching data for seasons: {', '.join(SEASONS)}")
    print(f"Output directory: {OUT_DIR}")
    
    # Outer loop to catch *any* unhandled exception and restart
    while True:
        try:
            main()
            print(f"✅ All data fetched successfully. Exiting at {datetime.now()}.")
            break
        except Exception:
            err_txt = traceback.format_exc()
            print(f"❌ Fatal error encountered:\n{err_txt}")
            # Append full traceback to fatal_errors.log
            with open(FATAL_LOG, "a", encoding="utf-8") as log:
                log.write(f"\n---\n{datetime.now()} —\n{err_txt}\n")
            # Wait then retry
            backoff = 60
            print(f"⏱ Waiting {backoff}s before restarting fetch…")
            time.sleep(backoff)
            print("🔄 Restarting fetch from top…")