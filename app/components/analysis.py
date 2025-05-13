import pandas as pd
import streamlit as st

# phase→code mapping
_PHASE_CODE = {
    "Regular Season": "R",
    "Playoffs":       "P",
    "Finals":         "F",
    "Preseason":      "PS",
    "Off-Season":     "O",
}

def points_over_games(
    df: pd.DataFrame,
    a_abbr: str,
    b_abbr: str,
    games_meta: pd.DataFrame
) -> pd.DataFrame:
    """
    Builds and renders a line chart of PTS over the series of games,
    with x-axis labels of the form PH-SEASON-MM-DD, where:
      • PH is R/P/F etc.
      • SEASON is shortened (e.g. '2024-25'→'24-25')
      • MM-DD is month-day of the game.
    Returns the pivot table (indexed by those labels).
    """
    # pivot total points per team per game_id
    pivot = (
        df[df["GAME_ID"].isin(games_meta["GAME_ID"])]
          .pivot_table(
              index="GAME_ID",
              columns="TEAM_ABBREVIATION",
              values="PTS",
              aggfunc="sum",
          )
          .loc[games_meta["GAME_ID"], [a_abbr, b_abbr]]
    )

    # build custom labels
    labels = []
    for _, row in games_meta.iterrows():
        sk = row["SeasonKey"][-5:]  # '2024-25' -> '24-25'
        ph = row["Phase"]
        code = _PHASE_CODE.get(ph, ph[:1].upper())
        date = row["GAME_DATE"]
        labels.append(f"{code}-{sk}-{date.strftime('%d-%b')}")

    pivot.index = labels
    st.subheader(f"{a_abbr} vs {b_abbr} — Points Over Games")
    st.line_chart(pivot)
    return pivot

def plus_minus_table(
    pivot: pd.DataFrame,
    a_abbr: str,
    b_abbr: str
) -> pd.Series:
    """
    Computes plus-minus = PTS_A - PTS_B, renders a table,
    and returns the diff series.
    
    Fix: Added robust error handling for NaN values
    """
    # Make sure both columns exist
    if a_abbr not in pivot.columns or b_abbr not in pivot.columns:
        # Return empty series with same index if columns missing
        return pd.Series(index=pivot.index)
    
    # Fill NaN values with 0 before calculating difference
    a_pts = pivot[a_abbr].fillna(0)
    b_pts = pivot[b_abbr].fillna(0)
    
    diff = (a_pts - b_pts).astype(int)
    pm = pd.DataFrame({
        "Game": pivot.index, 
        "PlusMinus": diff.values, 
        "MATCHUP": f"{a_abbr} vs {b_abbr}"
    })
    st.subheader("Plus-Minus per Game")
    st.dataframe(pm, use_container_width=True)
    return diff


def head_to_head_record(
    diff: pd.Series,
    a_label: str,
    b_label: str
) -> None:
    """
    Renders overall wins & total plus-minus.
    """
    st.subheader("Record & Total Plus-Minus")
    st.write(f"• {a_label} wins: {(diff > 0).sum()}")
    st.write(f"• {b_label} wins: {(diff < 0).sum()}")
    st.write(f"• Total /– for {a_label}: {int(diff.sum())}")

def get_series_box_scores(all_games: pd.DataFrame,
                          team_abbr: str,
                          opponent_abbr: str,
                          season: str = None,
                          phase: str = None) -> pd.DataFrame:
    """
    Returns every player box‐score row for games where team_abbr faced opponent_abbr.
    Adds a SeriesKey and GameNum so you can filter/select a particular series.
    """
    df = all_games.copy()
    if season and season != "All":
        df = df[df["SeasonKey"] == season]
    if phase and phase != "All":
        df = df[df["Phase"] == phase]

    # find the intersection of game IDs
    team_games = set(df[df["TEAM_ABBREVIATION"] == team_abbr]["GAME_ID"])
    opp_games  = set(df[df["TEAM_ABBREVIATION"] == opponent_abbr]["GAME_ID"])
    common = sorted(team_games & opp_games)
    df_series = df[df["GAME_ID"].isin(common)].copy()

    return df_series.sort_values(["SeriesKey","GameNum","TEAM_ABBREVIATION","PTS"], ascending=[True,True,True,False])

def get_team_season_trend(all_games: pd.DataFrame,
                          team_abbr: str,
                          phase: str = None) -> pd.DataFrame:
    df = all_games[all_games["TEAM_ABBREVIATION"] == team_abbr]
    if phase and phase != "All":
        df = df[df["Phase"] == phase]

    trend = (
        df.groupby("SeasonKey")["PTS"]
          .mean()
          .sort_index()
          .rename("AvgPTS")
          .to_frame()
    )
    return trend
# ── New: per‐game team log with top 3 scorers ────────────────────────────────
def team_game_log(all_games: pd.DataFrame,
                  team: str,
                  opponent: str,
                  season: str,
                  phase: str) -> pd.DataFrame:
    """
    Builds a per-game log for a team across all opponents (when `opponent` is empty),
    or for a specific opponent when provided. Includes top-3 scorers and scoreline.
    
    Fix: Added robust error handling for NaN values and defensive checks
    """
    # Filter by season and phase
    df = all_games.loc[
        (all_games.SeasonKey == season) &
        (all_games.Phase == phase)
    ]

    # If a specific opponent is selected, restrict to that series
    if opponent:
        df = df[df.TEAM_ABBREVIATION.isin([team, opponent])]

    # Identify all game IDs for the team
    game_ids = sorted(df[df.TEAM_ABBREVIATION == team]["GAME_ID"].unique())
    
    # Return empty frame if no games found
    if not game_ids:
        return pd.DataFrame()

    logs = []
    for gid in game_ids:
        game_df = df[df["GAME_ID"] == gid]
        
        # Skip if no data for this game
        if game_df.empty:
            continue

        # Derive opponent code using the Away/Home columns
        meta_rows = game_df.drop_duplicates("GAME_ID")[['Away','Home']]
        if meta_rows.empty or pd.isna(meta_rows['Away'].iloc[0]) or pd.isna(meta_rows['Home'].iloc[0]):
            # Skip if missing Away/Home data
            continue
            
        meta = meta_rows.iloc[0]
        opp_code = meta['Away'] if meta['Home'] == team else meta['Home']

        # Compute total points for each side - safely handle NaN values
        team_pts_series = game_df.loc[game_df["TEAM_ABBREVIATION"] == team, "PTS"]
        opp_pts_series = game_df.loc[game_df["TEAM_ABBREVIATION"] == opp_code, "PTS"] 
        
        # Safe sum with NaN handling
        team_pts = int(team_pts_series.fillna(0).sum()) if not team_pts_series.empty else 0
        opp_pts = int(opp_pts_series.fillna(0).sum()) if not opp_pts_series.empty else 0

        # Top-3 scorers for the team - with safe NaN handling
        player_pts = game_df[game_df["TEAM_ABBREVIATION"] == team][["PLAYER_NAME", "PTS"]].copy()
        player_pts = player_pts.dropna(subset=["PLAYER_NAME"])  # Remove rows with no player name
        player_pts["PTS"] = player_pts["PTS"].fillna(0)  # Fill NaN points with 0
        
        # Safely format top scorers
        t3 = []
        if not player_pts.empty:
            t3 = (
                player_pts
                .sort_values("PTS", ascending=False)
                .head(3)
                .apply(lambda r: f"{r.PLAYER_NAME} ({int(r.PTS)})", axis=1)
                .tolist()
            )
        t3 += [""] * (3 - len(t3))  # pad to exactly 3 entries

        # Game date - safely handle potential NaN
        date_val = game_df["GAME_DATE"].iloc[0] if not game_df.empty else None
        date_str = date_val.strftime('%d %b') if pd.notna(date_val) else "Unknown"

        logs.append({
            "GAME_ID": gid,
            "Date": date_str,
            "Matchup": f"{team} vs {opp_code}",
            "Score": f"{team_pts}–{opp_pts}",
            "Top1": t3[0],
            "Top2": t3[1],
            "Top3": t3[2],
        })

    return pd.DataFrame(logs)
# ── New: pivot any stat over a series ───────────────────────────────────────
def pivot_stat_over_series(all_games: pd.DataFrame,
                           team: str,
                           opponent: str,
                           series_games: list[str],
                           stat: str) -> pd.DataFrame:
    df = all_games[all_games.GAME_ID.isin(series_games)]
    pivot = (
        df
        .pivot_table(index="GAME_ID",
                     columns="TEAM_ABBREVIATION",
                     values=stat,
                     aggfunc="sum")
        .loc[series_games, [team, opponent]]
    )
    return pivot

# ── New: break down PTS into FGM/FG3M/FTM per game ────────────────────────
def pts_breakdown(all_games: pd.DataFrame,
                  team: str,
                  series_games: list) -> pd.DataFrame:
    """
    Returns a breakdown of total points by type (2PT, 3PT, FT)
    for each GAME_ID in the series, in the order of series_games.
    
    Fix: Robust handling of missing values
    """
    df = all_games[
        all_games.GAME_ID.isin(series_games) &
        (all_games.TEAM_ABBREVIATION == team)
    ]
    
    if df.empty:
        return pd.DataFrame()
    
    # Only include columns that exist
    cols_needed = ['FGM','FG3M','FTM']
    cols_present = [col for col in cols_needed if col in df.columns]
    
    if not all(col in cols_present for col in cols_needed):
        # Create empty DataFrame with right columns if missing data
        empty_df = pd.DataFrame(index=series_games, columns=['2PT', '3PT', 'FT']).fillna(0)
        return empty_df
    
    # Fill NaN values with 0 before aggregation
    df = df.copy()
    for col in cols_needed:
        df[col] = df[col].fillna(0)
    
    stats = df.groupby("GAME_ID")[cols_needed].sum()
    
    # 2-point field goals = total FGM minus 3PM, each worth 2 points
    two_pt_pts = (stats['FGM'] - stats['FG3M']) * 2
    three_pt_pts = stats['FG3M'] * 3
    ft_pts = stats['FTM']
    
    bd = pd.DataFrame({
        '2PT': two_pt_pts,
        '3PT': three_pt_pts,
        'FT': ft_pts
    })
    
    # ensure it follows the series order
    return bd.reindex(series_games)

def get_opponent_from_matchup(matchup, player_team):
    """Helper function to correctly extract the opponent team from a matchup string.
    
    Args:
        matchup (str): The matchup string (e.g., "LAL vs GSW" or "LAL @ BOS")
        player_team (str): The player's team code (e.g., "LAL")
        
    Returns:
        str: The opponent team code
    """
    # Handle NaN matchup
    if pd.isna(matchup):
        return None
        
    # First standardize the matchup format
    clean_matchup = matchup.replace("@", "vs")
    
    # Split on "vs" and handle any variations
    if " vs " in clean_matchup:
        teams = clean_matchup.split(" vs ")
    elif " vs. " in clean_matchup:
        teams = clean_matchup.split(" vs. ")
    else:
        # Fallback for other formats
        parts = clean_matchup.split()
        opponent_candidates = [p for p in parts if p != player_team and len(p) <= 3]
        return opponent_candidates[0] if opponent_candidates else None
    
    # Return the team that isn't the player's team
    if len(teams) < 2:
        return None
    return teams[1] if teams[0] == player_team else teams[0]

def get_series_box_scores(all_games: pd.DataFrame,
                          team_abbr: str,
                          opponent_abbr: str,
                          season: str = None,
                          phase: str = None) -> pd.DataFrame:
    """
    Returns every player box‐score row for games where team_abbr faced opponent_abbr.
    Adds a SeriesKey and GameNum so you can filter/select a particular series.
    
    Fix: Added robust error handling
    """
    df = all_games.copy()
    if season and season != "All":
        df = df[df["SeasonKey"] == season]
    if phase and phase != "All":
        df = df[df["Phase"] == phase]

    # find the intersection of game IDs
    team_games = set(df[df["TEAM_ABBREVIATION"] == team_abbr]["GAME_ID"])
    opp_games = set(df[df["TEAM_ABBREVIATION"] == opponent_abbr]["GAME_ID"])
    common = sorted(team_games & opp_games)
    
    if not common:
        return pd.DataFrame()  # Return empty DataFrame if no common games
        
    df_series = df[df["GAME_ID"].isin(common)].copy()
    
    # Check if SeriesKey column exists, create if needed
    if "SeriesKey" not in df_series.columns:
        teams_sorted = sorted([team_abbr, opponent_abbr])
        df_series["SeriesKey"] = f"{teams_sorted[0]} vs {teams_sorted[1]}"
    
    # Ensure GameNum column exists
    if "GameNum" not in df_series.columns:
        # Try to extract from existing data or default to sequential numbering
        game_nums = {}
        for i, gid in enumerate(sorted(df_series["GAME_ID"].unique())):
            game_nums[gid] = i + 1
        
        df_series["GameNum"] = df_series["GAME_ID"].map(game_nums)

    return df_series.sort_values(["SeriesKey","GameNum","TEAM_ABBREVIATION","PTS"], ascending=[True,True,True,False])