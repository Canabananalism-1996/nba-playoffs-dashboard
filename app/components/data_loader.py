import json
from pathlib import Path
from typing import Dict, List
import pandas as pd
import streamlit as st

# ─── CONFIG ─────────────────────────────────────────────────────────────────────
DATA_DIR = Path(__file__).resolve().parents[2] / "data-raw"

# ─── PHASE WINDOWS ───────────────────────────────────────────────────────────────
_WINDOWS: Dict[str, Dict[str, tuple[pd.Timestamp, pd.Timestamp]]] = {
    "2021-22": {
        "Preseason": (pd.Timestamp("2021-10-01"), pd.Timestamp("2021-10-15")),
        "Regular Season": (pd.Timestamp("2021-10-16"), pd.Timestamp("2022-04-10")),
        "Playoffs": (pd.Timestamp("2022-04-11"), pd.Timestamp("2022-05-30")),
        "Finals": (pd.Timestamp("2022-06-01"), pd.Timestamp("2022-06-20")),
    },
    "2022-23": {
        "Preseason": (pd.Timestamp("2022-09-30"), pd.Timestamp("2022-10-15")),
        "Regular Season": (pd.Timestamp("2022-10-17"), pd.Timestamp("2023-04-10")),
        "Playoffs": (pd.Timestamp("2023-04-11"), pd.Timestamp("2023-05-28")),
        "Finals": (pd.Timestamp("2023-05-30"), pd.Timestamp("2023-06-20")),
    },
    "2023-24": {
        "Preseason": (pd.Timestamp("2023-10-01"), pd.Timestamp("2023-10-20")),
        "Regular Season": (pd.Timestamp("2023-10-22"), pd.Timestamp("2024-04-15")),
        "Playoffs": (pd.Timestamp("2024-04-16"), pd.Timestamp("2024-05-30")),
        "Finals": (pd.Timestamp("2024-06-01"), pd.Timestamp("2024-06-25")),
    },
    "2024-25": {
        "Preseason": (pd.Timestamp("2024-10-01"), pd.Timestamp("2024-10-20")),
        "Regular Season": (pd.Timestamp("2024-10-22"), pd.Timestamp("2025-04-13")),
        "Playoffs": (pd.Timestamp("2025-04-15"), pd.Timestamp("2025-05-31")),
        "Finals": (pd.Timestamp("2025-06-01"), pd.Timestamp("2025-06-25")),
    },
}

# ─── PHASE CLASSIFIER ────────────────────────────────────────────────────────────
def classify_phase(dt: pd.Timestamp, season_key: str) -> str:
    """
    Map a game date to its phase using explicit date windows by SeasonKey.
    """
    if season_key in _WINDOWS:
        for phase, (start, end) in _WINDOWS[season_key].items():
            if start <= dt <= end:
                return phase
    return "Off-Season"

# ─── DATA LOADERS ────────────────────────────────────────────────────────────────
@st.cache_data
def load_seasons() -> List[str]:
    files = DATA_DIR.glob("games_meta_*.json")
    return sorted(p.stem.replace("games_meta_", "") for p in files)

@st.cache_data
def load_metadata(season: str) -> pd.DataFrame:
    path = DATA_DIR / f"games_meta_{season}.json"
    df = pd.read_json(path.read_text())
    df["GAME_ID"] = df["GAME_ID"].astype(str)
    df = df.dropna(subset=["GAME_ID", "GAME_DATE"])
    df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"], errors="coerce")
    df = df.dropna(subset=["GAME_DATE"])
    df["SeasonKey"] = season
    return df[["GAME_ID","GAME_DATE","MATCHUP","WL","SeasonKey","TEAM_ABBREVIATION","TEAM_NAME"]]

@st.cache_data
def load_player_stats(season: str) -> pd.DataFrame:
    path = DATA_DIR / f"player_stats_{season}.json"
    df = pd.read_json(path.read_text())
    df["GAME_ID"] = df["GAME_ID"].astype(str)
    df = df.dropna(subset=["GAME_ID","PLAYER_NAME"])
    df["SeasonKey"] = season
    return df

# ─── MERGE, ANNOTATE & NUMBER ────────────────────────────────────────────────────
@st.cache_data
def load_all_games() -> pd.DataFrame:
    seasons = load_seasons()
    all_frames: List[pd.DataFrame] = []
    for season in seasons:
        meta = load_metadata(season).drop_duplicates("GAME_ID")
        stats = load_player_stats(season)
        df = stats.merge(meta, on="GAME_ID", how="left", suffixes=("","_meta"))
        df["SeasonKey"] = df["SeasonKey_meta"].fillna(df["SeasonKey"])
        df.drop(columns=["SeasonKey_meta"], inplace=True, errors="ignore")
        all_frames.append(df)
    if not all_frames:
        return pd.DataFrame()
    all_games = pd.concat(all_frames, ignore_index=True)
    all_games["GAME_DATE"] = pd.to_datetime(all_games["GAME_DATE"], errors="coerce")
    all_games["Phase"] = all_games.apply(
        lambda r: classify_phase(r["GAME_DATE"], r["SeasonKey"]), axis=1
    )
    # derive canonical Away/Home using partition to guarantee two parts
    cleaned = (
        all_games["MATCHUP"].fillna("")
          .str.replace("@", " vs ", regex=False)
          .str.replace(r"\s*vs\.?\s*", " vs ", regex=True)
          .str.upper()
    )
    parts = cleaned.str.partition(" VS ")
    all_games["Away"] = parts[0].str.strip()
    all_games["Home"] = parts[2].str.strip()
    # sort for numbering
    all_games["SeriesKey"] = all_games.apply(
    lambda r: " vs ".join(sorted([r["Away"], r["Home"]])),
    axis=1
)

# then sort & rank by that:
    all_games.sort_values(
    ["SeasonKey", "Phase", "SeriesKey", "GAME_DATE"],
    inplace=True,
    ignore_index=True,
)
    all_games["GameNum"] = (
    all_games
      .groupby(["SeasonKey", "Phase", "SeriesKey"], sort=False)["GAME_DATE"]
      .transform(lambda dates: dates.rank(method="dense"))
      .astype("Int64")
)
    return all_games

# ─── GAME MAP ───────────────────────────────────────────────────────────────────
def build_game_map(all_games: pd.DataFrame) -> Dict[str,str]:
    meta = (
        all_games[["SeasonKey","Phase","GAME_ID","GAME_DATE","GameNum","Away","Home"]]
        .drop_duplicates(subset="GAME_ID")
        .sort_values(["SeasonKey","Phase","GameNum"])
    )
    meta["DateStr"] = meta["GAME_DATE"].dt.strftime("%Y-%m-%d")
    game_nums = meta["GameNum"].astype("Int64").astype(str)
    meta["Label"] = (
        meta["Away"] + " vs " + meta["Home"] +
        " — " + meta["SeasonKey"] +
        " (" + meta["Phase"] + " Game " + game_nums + ") " + meta["DateStr"]
    )
    return dict(zip(meta["Label"], meta["GAME_ID"]))
