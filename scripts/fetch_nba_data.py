#!/usr/bin/env python3
"""
Fetch NBA data (teams, players, games, player box scores) for seasons 2021 through current date.
Extract traditional + advanced stats, append full stat names, and save per-season JSON incrementally.
Supports resuming from last saved game, automatic restart on missing data, and recovers from corrupted JSON files.
"""
import os
import sys
import time
import json
import traceback
import pkgutil
import importlib
from datetime import datetime
import socket

import pandas as pd
from requests.exceptions import RequestException, ReadTimeout
from nba_api.stats.static import teams as teams_static
from nba_api.stats.static import players as players_static
import nba_api.stats.endpoints as endpoints_module
from nba_api.stats.library.http import NBAStatsHTTP

NBAStatsHTTP.READ_TIMEOUT    = 60  # seconds
NBAStatsHTTP.CONNECT_TIMEOUT = 30


MAX_RETRIES = 5
RETRY_DELAY = 5  # seconds

def retry(fn, *args, **kwargs):
    for i in range(1, MAX_RETRIES+1):
        try:
            return fn(*args, **kwargs)
        except ReadTimeout:
            print(f"⚠️ leaguegamefinder timeout {i}/{MAX_RETRIES}, retrying in {RETRY_DELAY}s…")
            time.sleep(RETRY_DELAY)
    # last attempt (will raise if it still fails)
    return fn(*args, **kwargs)


# ─── DYNAMIC ENDPOINT IMPORT ─────────────────────────────────────────────────────
ALL_ENDPOINTS = [name for _, name, _ in pkgutil.iter_modules(endpoints_module.__path__)]
print("Available endpoints:", ALL_ENDPOINTS)
ENDPOINTS = {}
for ep in ALL_ENDPOINTS:
    try:
        module = importlib.import_module(f"nba_api.stats.endpoints.{ep}")
        ENDPOINTS[ep] = module
    except ImportError:
        continue

leaguegamefinder = ENDPOINTS['leaguegamefinder'].LeagueGameFinder
box_trad = ENDPOINTS['boxscoretraditionalv2'].BoxScoreTraditionalV2
box_adv  = ENDPOINTS['boxscoreadvancedv2'].BoxScoreAdvancedV2

# ─── CONFIG ──────────────────────────────────────────────────────────────────────
ROOT_DIR     = os.path.dirname(os.path.abspath(__file__))
OUT_DIR      = os.path.join(ROOT_DIR, '..', 'data-raw')
START_YEAR   = 2021
CURRENT_YEAR = datetime.now().year
SLEEP        = 1.0   # seconds between requests
SKIPPED_FILE = os.path.join(OUT_DIR, 'skipped_boxscores.txt')
FATAL_LOG    = os.path.join(OUT_DIR, 'fatal_errors.log')

if datetime.now().month >= 7:
    SEASONS = [f"{yr}-{(yr+1)%100:02d}" for yr in range(START_YEAR, CURRENT_YEAR+1)]
else:
    SEASONS = [f"{yr}-{(yr+1)%100:02d}" for yr in range(START_YEAR, CURRENT_YEAR)]

# ─── HELPERS ─────────────────────────────────────────────────────────────────────
def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def save_json(obj, fn):
    path = os.path.join(OUT_DIR, fn)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(obj, f, indent=2)


def load_json(fn):
    """
    Load a JSON file, recovering from decode errors by backing up and resetting.
    Returns an empty list if the file is missing or corrupted.
    """
    path = os.path.join(OUT_DIR, fn)
    if not os.path.exists(path):
        return []
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except json.JSONDecodeError as e:
        print(f"⚠️  Corrupted JSON in '{fn}' ({e}). Backing up and starting fresh.")
        backup = path + '.broken'
        os.replace(path, backup)
        return []


def log_skip(gid):
    with open(SKIPPED_FILE, 'a') as f:
        f.write(f"{gid}\n")


def restart_script():
    """Re-executes the current Python process with the same arguments."""
    print("🔄 No stats found — restarting script to pick up where we left off…")
    os.execv(sys.executable, [sys.executable] + sys.argv)

# ─── FETCH FUNCTIONS ─────────────────────────────────────────────────────────────
def fetch_teams():
    teams = teams_static.get_teams()
    save_json(teams, 'teams.json')


def fetch_players():
    players = players_static.get_players()
    save_json(players, 'players.json')


def fetch_games(season):
    meta_fn = f"games_meta_{season}.json"
    ids_fn  = f"game_ids_{season}.json"
    # ─── always refresh your games_meta JSON ─────────────────────────────────
    # load existing meta (if any) so we can preserve past records
    old_meta = pd.DataFrame(load_json(meta_fn)) if os.path.exists(os.path.join(OUT_DIR, meta_fn)) else pd.DataFrame()

    # fetch the latest Regular + Playoffs listings
    reg = retry(lambda: leaguegamefinder(
        season_nullable=season,
        season_type_nullable='Regular Season'
    ).get_data_frames()[0])
    po = retry(lambda: leaguegamefinder(
        season_nullable=season,
        season_type_nullable='Playoffs'
    ).get_data_frames()[0])
    new_meta = pd.concat([reg, po], ignore_index=True)

    # merge, dedupe by GAME_ID, keep the latest date info
    if not old_meta.empty:
        combined = pd.concat([old_meta, new_meta], ignore_index=True)
        df = combined.drop_duplicates(subset="GAME_ID", keep="last")
    else:
        df = new_meta

    # write back your updated meta and id lists
    # ─── convert GAME_DATE to plain strings for JSON serialization ───
    # copy first to avoid SettingWithCopyWarning
    df = df.copy()
    df['GAME_DATE'] = pd.to_datetime(df['GAME_DATE']).dt.strftime('%Y-%m-%d')
    save_json(df.to_dict('records'), meta_fn)
        
    # ensure GAME_DATE is string, then filter by date
    df['GAME_DATE'] = pd.to_datetime(df['GAME_DATE']).dt.strftime('%Y-%m-%d')
    df = df[pd.to_datetime(df['GAME_DATE']) <= datetime.now()]
    ids = sorted(df['GAME_ID'].unique())
    save_json(ids, ids_fn)
    return ids


def fetch_scores(gids):
    records = []
    total = len(gids)
    for i, gid in enumerate(gids, 1):
        print(f" Fetching game {i}/{total} ID {gid}")
        for attempt in range(3):
            try:
                tdf = box_trad(game_id=gid, timeout=3).get_data_frames()[0]
                adf = box_adv(game_id=gid, timeout=3).get_data_frames()[0]
                m = tdf.merge(adf, on=['GAME_ID','TEAM_ABBREVIATION','PLAYER_ID','PLAYER_NAME'], suffixes=('','_ADV'))
                records.extend(m.to_dict('records'))
                break
            except (RequestException, socket.timeout) as e:
                print(f"   ⚠️ Attempt {attempt+1} failed for game {gid}: {e}")
                time.sleep(2)
        time.sleep(5)
    return records

# ─── MAIN ────────────────────────────────────────────────────────────────────────
def main():
    ensure_dir(OUT_DIR)
    print("▶️ Seasons:", SEASONS)

    print("⏳ Teams…"); fetch_teams(); print("✅")
    print("⏳ Players…"); fetch_players(); print("✅")

    for season in SEASONS:
        print(f"\n🔍 Season {season}")
        ids = fetch_games(season)
        out_fn = f"player_stats_{season}.json"

        existing = load_json(out_fn)
        processed = {r['GAME_ID'] for r in existing} if existing else set()

        try:
            with open(SKIPPED_FILE) as f:
                skipped = {line.strip() for line in f}
        except FileNotFoundError:
            skipped = set()
        processed |= skipped

        to_do = [g for g in ids if g not in processed]
        print(f"   {len(processed)} processed, {len(to_do)} to fetch")

        if to_do:
            for idx, gid in enumerate(to_do, start=1):
                print(f"   ▶️ [{idx}/{len(to_do)}] Fetching and saving game ID {gid}")
                recs = fetch_scores([gid])

                if recs:
                    existing.extend(recs)
                    save_json(existing, out_fn)
                    print(f"   💾 Saved {out_fn} ({len(existing)} total records)")
                else:
                    log_skip(gid)
                    print(f"   ⚠️ No records returned for game {gid}")
                    # immediately restart so load_json picks up next batch cleanly
                    restart_script()

            print(f"   🎯 Season {season} up-to-date ({len(existing)} total records)")
        else:
            print("   ✅ Season already up-to-date")

    print("\n🎉 Done.")

if __name__ == '__main__':
    try:
        main()
        # ─── Auto-commit & push updated JSONs ─────────────────────────
        import subprocess
        from datetime import datetime

        # Stage any changed JSONs
        subprocess.run("git add data-raw/*.json", shell=True, check=False)

        # Commit with today's date (if there’s anything new)
        commit_msg = f"chore: auto-update NBA data {datetime.utcnow().date()}"
        subprocess.run(f'git commit -m "{commit_msg}"', shell=True, check=False)

        # Push back to main
        subprocess.run("git push origin main", shell=True, check=False)

    except Exception:
        with open(FATAL_LOG, 'a') as f:
            f.write(traceback.format_exc())
        raise
