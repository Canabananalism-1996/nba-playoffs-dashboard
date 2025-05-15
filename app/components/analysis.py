import pandas as pd
import streamlit as st
import numpy as np
from scipy import stats

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
    
    # Reset the index and add 1 to make it 1-based instead of 0-based
    pm = pm.reset_index(drop=True)
    pm.index = pm.index + 1
    
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

# ── New: Player vs Player comparison ───────────────────────────────────────
def compare_players(all_games: pd.DataFrame,
                   player1_name: str,
                   player2_name: str,
                   season: str = None,
                   phase: str = None,
                   min_games: int = 5) -> dict:
    """
    Performs statistical comparison between two players with significance testing.
    
    Args:
        all_games: DataFrame containing all game data
        player1_name: Name of first player to compare
        player2_name: Name of second player to compare
        season: Season to filter by (optional)
        phase: Phase to filter by (optional)
        min_games: Minimum games required for valid comparison
        
    Returns:
        Dictionary containing comparison results and statistical analysis
    """
    # DEBUG: Print season and phase selections
    print(f"DEBUG: Selected season: {season}, phase: {phase}")
    
    # Original, unfiltered data for each player
    orig_p1_data = all_games[all_games["PLAYER_NAME"] == player1_name]
    orig_p2_data = all_games[all_games["PLAYER_NAME"] == player2_name]
    
    # Print original game counts
    print(f"DEBUG: Original data - {player1_name}: {orig_p1_data['GAME_ID'].nunique()} games, {player2_name}: {orig_p2_data['GAME_ID'].nunique()} games")
    
    # Filter data for the specific players and conditions
    df = all_games.copy()
    if season and season != "All":
        df = df[df["SeasonKey"] == season]
        print(f"DEBUG: After season filter - Games in DataFrame: {df['GAME_ID'].nunique()}")
    if phase and phase != "All":
        df = df[df["Phase"] == phase]
        print(f"DEBUG: After phase filter - Games in DataFrame: {df['GAME_ID'].nunique()}")
    
    # Get data for each player
    p1_data = df[df["PLAYER_NAME"] == player1_name]
    p2_data = df[df["PLAYER_NAME"] == player2_name]
    
    # Print filtered game counts
    print(f"DEBUG: Filtered data - {player1_name}: {p1_data['GAME_ID'].nunique()} games, {player2_name}: {p2_data['GAME_ID'].nunique()} games")
    
    # If phase is specified, ensure we're only seeing those games
    if phase and phase != "All":
        # Get unique phases for each player in filtered data
        p1_phases = p1_data["Phase"].unique()
        p2_phases = p2_data["Phase"].unique()
        print(f"DEBUG: Filtered phases - {player1_name}: {p1_phases}, {player2_name}: {p2_phases}")
    
    # Check if we have enough data
    if p1_data.empty or p2_data.empty:
        return {"error": "One or both players have no data for the selected criteria"}
    
    # Get the players' team information (using the filtered data)
    p1_team = p1_data["TEAM_ABBREVIATION"].mode().iloc[0] if not p1_data.empty else "Unknown"
    p2_team = p2_data["TEAM_ABBREVIATION"].mode().iloc[0] if not p2_data.empty else "Unknown"
    
    # Calculate player record based on filtered data
    p1_games = p1_data["GAME_ID"].nunique()
    p2_games = p2_data["GAME_ID"].nunique()
    
    # Get unique game IDs for each player
    p1_game_ids = p1_data["GAME_ID"].unique()
    p2_game_ids = p2_data["GAME_ID"].unique()

    # ---- COMPLETELY REVISED: Get player team records using metadata only ----
    # CRITICAL FIX: More thorough approach to calculating team records
    print("\n----- PLAYER TEAM RECORD CALCULATION DEBUG -----")
    print(f"Calculating team records for {player1_name} ({p1_team}) and {player2_name} ({p2_team})...")

    # Show the total number of game IDs for each player
    print(f"Total unique games for {player1_name}: {len(p1_game_ids)}")
    print(f"Total unique games for {player2_name}: {len(p2_game_ids)}")

    # Calculate player1's team record using the improved point-based method
    p1_team_wins = 0
    p1_team_losses = 0
    
    for game_id in p1_game_ids:
        # Get all teams in this game
        game_data = all_games[all_games["GAME_ID"] == game_id]
        game_teams = game_data["TEAM_ABBREVIATION"].unique()
        
        # Skip if we can't find both teams
        if len(game_teams) < 2:
            continue
            
        # Find the opponent
        opponents = [t for t in game_teams if t != p1_team]
        if not opponents:
            continue
            
        opponent = opponents[0]
        
        # Apply phase filter if needed
        if phase and phase != "All":
            if not any(game_data["Phase"] == phase):
                continue
        
        # Calculate points for each team
        team_pts = game_data[game_data["TEAM_ABBREVIATION"] == p1_team]["PTS"].sum()
        opp_pts = game_data[game_data["TEAM_ABBREVIATION"] == opponent]["PTS"].sum()
        
        # Determine result based on points
        if team_pts > opp_pts:
            p1_team_wins += 1
        elif team_pts < opp_pts:
            p1_team_losses += 1

    # Same for player2's team
    p2_team_wins = 0
    p2_team_losses = 0
    
    for game_id in p2_game_ids:
        # Get all teams in this game
        game_data = all_games[all_games["GAME_ID"] == game_id]
        game_teams = game_data["TEAM_ABBREVIATION"].unique()
        
        # Skip if we can't find both teams
        if len(game_teams) < 2:
            continue
            
        # Find the opponent
        opponents = [t for t in game_teams if t != p2_team]
        if not opponents:
            continue
            
        opponent = opponents[0]
        
        # Apply phase filter if needed
        if phase and phase != "All":
            if not any(game_data["Phase"] == phase):
                continue
        
        # Calculate points for each team
        team_pts = game_data[game_data["TEAM_ABBREVIATION"] == p2_team]["PTS"].sum()
        opp_pts = game_data[game_data["TEAM_ABBREVIATION"] == opponent]["PTS"].sum()
        
        # Determine result based on points
        if team_pts > opp_pts:
            p2_team_wins += 1
        elif team_pts < opp_pts:
            p2_team_losses += 1

    print(f"\nFINAL TEAM RECORD CALCULATION:")
    print(f"{player1_name}'s team ({p1_team}): {p1_team_wins}-{p1_team_losses} from {len(p1_game_ids)} games")
    print(f"{player2_name}'s team ({p2_team}): {p2_team_wins}-{p2_team_losses} from {len(p2_game_ids)} games")

    # Extra detailed debug for GSW and MIN players
    for debug_team, debug_player, game_ids in [
        (p1_team, player1_name, p1_game_ids) if p1_team in ["GSW", "MIN"] else (None, None, None),
        (p2_team, player2_name, p2_game_ids) if p2_team in ["GSW", "MIN"] else (None, None, None)
    ]:
        if debug_team is None:
            continue
            
        # Get the correct record for this team
        debug_wins = p1_team_wins if debug_team == p1_team else p2_team_wins
        debug_losses = p1_team_losses if debug_team == p1_team else p2_team_losses
            
        print(f"\nDETAILED {debug_player}'s TEAM ({debug_team}) GAME LOG: {debug_wins}-{debug_losses} record")
        
        # Go through each game and show detailed information
        for i, game_id in enumerate(game_ids):
            game_data = df[df["GAME_ID"] == game_id]
            if game_data.empty:
                continue
                
            # Get game information
            date_val = game_data["GAME_DATE"].iloc[0] if "GAME_DATE" in game_data.columns else None
            date_str = date_val.strftime("%Y-%m-%d") if isinstance(date_val, pd.Timestamp) else str(date_val)
            
            # Get matchup information
            matchup = game_data["MATCHUP"].iloc[0] if "MATCHUP" in game_data.columns else "Unknown"
            
            # Get home/away status
            home_away = "Home"
            if "@" in matchup:
                home_away = "Away" if matchup.startswith(debug_team) else "Home"
            else:
                home_away = "Home" if matchup.startswith(debug_team) else "Away"
            
            # Get opponent
            game_teams = game_data["TEAM_ABBREVIATION"].unique()
            opponents = [t for t in game_teams if t != debug_team]
            opponent = opponents[0] if opponents else "Unknown"
            
            # Get result based on points
            team_pts = game_data[game_data["TEAM_ABBREVIATION"] == debug_team]["PTS"].sum()
            opp_pts = game_data[game_data["TEAM_ABBREVIATION"] == opponent]["PTS"].sum() if opponent != "Unknown" else 0
            result = "W" if team_pts > opp_pts else "L" if team_pts < opp_pts else "Unknown"
            
            # Get phase
            phase_val = game_data["Phase"].iloc[0] if "Phase" in game_data.columns else "Unknown"
            
            print(f"{i+1}. GAME_ID: {game_id}, Date: {date_str}, Matchup: {matchup}, Location: {home_away}, " +
                  f"vs {opponent}, Result: {result}, Phase: {phase_val}")
    
    # Define important statistics to compare
    key_stats = [
        'PTS', 'REB', 'AST', 'STL', 'BLK', 'TOV', 'FGM', 'FGA', 'FG3M', 
        'FG3A', 'FTM', 'FTA', 'PLUS_MINUS'
    ]
    
    # Keep only stats that exist in the data
    available_stats = [stat for stat in key_stats if stat in df.columns]
    
    # Check game count
    if p1_games < min_games or p2_games < min_games:
        return {
            "error": f"Insufficient games for comparison. {player1_name}: {p1_games} games, {player2_name}: {p2_games} games. Minimum required: {min_games} games."
        }
    
    # Initialize results
    comparison = {
        "player1": {
            "name": player1_name,
            "team": p1_team,
            "games": p1_games,
            "record": f"{p1_team_wins}-{p1_team_losses}",
            "win_pct": round(p1_team_wins / max(p1_team_wins + p1_team_losses, 1) * 100, 1)
        },
        "player2": {
            "name": player2_name,
            "team": p2_team,
            "games": p2_games,
            "record": f"{p2_team_wins}-{p2_team_losses}",
            "win_pct": round(p2_team_wins / max(p2_team_wins + p2_team_losses, 1) * 100, 1)
        },
        "stats": {},
        "derived_stats": {},
        "statistical_tests": {}
    }
    
    # Calculate per-game averages, standard deviations, and totals for each player
    for stat in available_stats:
        # Skip if data missing
        if stat not in p1_data.columns or stat not in p2_data.columns:
            continue
            
        # Handle NaN values safely
        p1_values = p1_data[stat].fillna(0)
        p2_values = p2_data[stat].fillna(0)
        
        # Basic statistics
        p1_avg = p1_values.mean()
        p2_avg = p2_values.mean()
        p1_std = p1_values.std() if len(p1_values) > 1 else 0
        p2_std = p2_values.std() if len(p2_values) > 1 else 0
        p1_total = p1_values.sum()
        p2_total = p2_values.sum()
        
        # Store in comparison dict
        comparison["stats"][stat] = {
            "player1_avg": round(p1_avg, 2),
            "player2_avg": round(p2_avg, 2),
            "player1_std": round(p1_std, 2),
            "player2_std": round(p2_std, 2),
            "player1_total": int(p1_total) if stat in ['PTS', 'REB', 'AST', 'STL', 'BLK', 'TOV'] else round(p1_total, 1),
            "player2_total": int(p2_total) if stat in ['PTS', 'REB', 'AST', 'STL', 'BLK', 'TOV'] else round(p2_total, 1),
            "difference": round(p1_avg - p2_avg, 2),
            "percent_diff": round(((p1_avg - p2_avg) / max(p2_avg, 0.001)) * 100, 1) if p2_avg != 0 else float('inf')
        }
        
        # Statistical significance testing (t-test for unequal variances)
        # Only if we have sufficient samples
        if len(p1_values) >= min_games and len(p2_values) >= min_games:
            t_stat, p_value = stats.ttest_ind(p1_values, p2_values, equal_var=False, nan_policy='omit')
            comparison["statistical_tests"][stat] = {
                "t_statistic": round(t_stat, 3),
                "p_value": round(p_value, 4),
                "significant": p_value < 0.05
            }
    
    # Calculate derived statistics
    if all(stat in available_stats for stat in ['FGM', 'FGA']):
        # Field Goal Percentage
        p1_fgp = p1_data['FGM'].sum() / max(p1_data['FGA'].sum(), 1) * 100
        p2_fgp = p2_data['FGM'].sum() / max(p2_data['FGA'].sum(), 1) * 100
        
        comparison["derived_stats"]["FG_PCT"] = {
            "player1": round(p1_fgp, 1),
            "player2": round(p2_fgp, 1),
            "difference": round(p1_fgp - p2_fgp, 1)
        }
    
    if all(stat in available_stats for stat in ['FG3M', 'FG3A']):
        # Three Point Percentage
        p1_3p = p1_data['FG3M'].sum() / max(p1_data['FG3A'].sum(), 1) * 100
        p2_3p = p2_data['FG3M'].sum() / max(p2_data['FG3A'].sum(), 1) * 100
        
        comparison["derived_stats"]["FG3_PCT"] = {
            "player1": round(p1_3p, 1),
            "player2": round(p2_3p, 1),
            "difference": round(p1_3p - p2_3p, 1)
        }
    
    if all(stat in available_stats for stat in ['FTM', 'FTA']):
        # Free Throw Percentage
        p1_ft = p1_data['FTM'].sum() / max(p1_data['FTA'].sum(), 1) * 100
        p2_ft = p2_data['FTM'].sum() / max(p2_data['FTA'].sum(), 1) * 100
        
        comparison["derived_stats"]["FT_PCT"] = {
            "player1": round(p1_ft, 1),
            "player2": round(p2_ft, 1),
            "difference": round(p1_ft - p2_ft, 1)
        }
    
    # Per-game distribution analysis
    performance_comparison = {}
    for stat in ['PTS', 'REB', 'AST']:
        if stat not in available_stats:
            continue
            
        # Group data by games and analyze performances
        p1_game_stats = p1_data.groupby('GAME_ID')[stat].sum().fillna(0)
        p2_game_stats = p2_data.groupby('GAME_ID')[stat].sum().fillna(0)
        
        # Calculate percentage of games above various thresholds
        thresholds = [10, 15, 20, 25, 30] if stat == 'PTS' else [5, 10, 15, 20]
        
        threshold_data = {}
        for threshold in thresholds:
            p1_pct = (p1_game_stats >= threshold).mean() * 100
            p2_pct = (p2_game_stats >= threshold).mean() * 100
            
            threshold_data[threshold] = {
                "player1_pct": round(p1_pct, 1),
                "player2_pct": round(p2_pct, 1),
                "difference": round(p1_pct - p2_pct, 1)
            }
        
        performance_comparison[stat] = threshold_data
    
    comparison["performance_thresholds"] = performance_comparison
    
    # Calculate consistency metrics (coefficient of variation)
    consistency = {}
    for stat in ['PTS', 'REB', 'AST']:
        if stat not in available_stats:
            continue
        
        p1_cv = (p1_data.groupby('GAME_ID')[stat].sum().std() / 
                 p1_data.groupby('GAME_ID')[stat].sum().mean()) * 100 if p1_games > 1 else 0
        
        p2_cv = (p2_data.groupby('GAME_ID')[stat].sum().std() / 
                 p2_data.groupby('GAME_ID')[stat].sum().mean()) * 100 if p2_games > 1 else 0
        
        # Lower CV = more consistent
        consistency[stat] = {
            "player1_cv": round(p1_cv, 1), 
            "player2_cv": round(p2_cv, 1),
            "more_consistent": player1_name if p1_cv < p2_cv else player2_name
        }
    
    comparison["consistency"] = consistency
    
    return comparison

# New function: Compare player performance in different contexts with paired t-tests
def player_performance_analysis(all_games: pd.DataFrame,
                               player_name: str,
                               # Period 1: Overall/Baseline performance period
                               overall_season: str = None,
                               overall_phase: str = None,
                               # Period 2: Comparison context performance period
                               comparison_season: str = None,
                               comparison_phase: str = None,
                               comparison_opponent_name: str = None,
                               comparison_opponent_team: str = None,
                               comparison_series_id: str = None,
                               min_games: int = 3) -> dict:
    """
    Analyzes a player's performance by comparing two distinct periods/contexts.
    
    Compares a player's performance in an 'overall_period' (defined by overall_season/phase)
    with their performance in a 'comparison_period' (defined by comparison_season/phase
    and optionally refined by opponent or series).
    
    Args:
        all_games: DataFrame with all game data
        player_name: Name of the player to analyze
        overall_season: Season for the 'overall' performance period
        overall_phase: Phase for the 'overall' performance period
        comparison_season: Season for the 'comparison' performance period
        comparison_phase: Phase for the 'comparison' performance period
        comparison_opponent_name: Optional opponent player for the 'comparison' period
        comparison_opponent_team: Optional opponent team for the 'comparison' period
        comparison_series_id: Optional specific series for the 'comparison' period
        min_games: Minimum number of games required for valid analysis in each period
        
    Returns:
        Dictionary with analysis results.
    """
    player_df_full = all_games[all_games["PLAYER_NAME"] == player_name].copy()
    
    if player_df_full.empty:
        return {"error": f"No data found for player {player_name}"}
    
    # Accurate player team determination (using canonical map)
    canonical_name_to_abbr_map = all_games.groupby('TEAM_NAME')['TEAM_ABBREVIATION'].apply(
        lambda x: x.mode().iloc[0] if not x.mode().empty else None
    ).dropna().to_dict()
    player_team_determined = False
    player_team = None
    if "TEAM_NAME" in player_df_full.columns and not player_df_full["TEAM_NAME"].mode().empty:
        most_frequent_player_team_name = player_df_full["TEAM_NAME"].mode().iloc[0]
        resolved_player_team_abbr = canonical_name_to_abbr_map.get(most_frequent_player_team_name)
        if resolved_player_team_abbr is not None:
            player_team = resolved_player_team_abbr
            player_team_determined = True
    if not player_team_determined:
        if "TEAM_ABBREVIATION" in player_df_full.columns and not player_df_full["TEAM_ABBREVIATION"].mode().empty:
            player_team = player_df_full["TEAM_ABBREVIATION"].mode().iloc[0]
        else:
            return {"error": f"Cannot determine team for player {player_name}: Missing TEAM_NAME and TEAM_ABBREVIATION."}

    key_stats = ['PTS', 'REB', 'AST', 'STL', 'BLK', 'TOV', 'FGM', 'FGA', 'FG3M', 
                 'FG3A', 'FTM', 'FTA', 'PLUS_MINUS']
    available_stats = [stat for stat in key_stats if stat in player_df_full.columns]

    # --- Define Period 1 DataFrame (Overall/Baseline) ---
    overall_period_df = player_df_full.copy()
    overall_period_desc_parts = []
    if overall_season:
        overall_period_df = overall_period_df[overall_period_df["SeasonKey"] == overall_season]
        overall_period_desc_parts.append(f"{overall_season}")
    else:
        overall_period_desc_parts.append("All Seasons")
    if overall_phase:
        overall_period_df = overall_period_df[overall_period_df["Phase"] == overall_phase]
        overall_period_desc_parts.append(f"{overall_phase}")
    else:
        overall_period_desc_parts.append("All Phases")
    
    overall_period_description = ", ".join(overall_period_desc_parts)
    if overall_period_df["GAME_ID"].nunique() < min_games:
        return {"error": f"Insufficient games ({overall_period_df['GAME_ID'].nunique()}) for Period 1 ({overall_period_description}). Need at least {min_games}."}

    # --- Define Period 2 DataFrame (Comparison Context) ---
    comparison_context_df = player_df_full.copy()
    comparison_context_desc_parts = []

    if comparison_season:
        comparison_context_df = comparison_context_df[comparison_context_df["SeasonKey"] == comparison_season]
        comparison_context_desc_parts.append(f"in {comparison_season}")
    if comparison_phase:
        comparison_context_df = comparison_context_df[comparison_context_df["Phase"] == comparison_phase]
        comparison_context_desc_parts.append(f"during {comparison_phase}")
    
    # Apply opponent/series filters to comparison_context_df
    if comparison_opponent_name:
        opp_team_df = all_games[all_games["PLAYER_NAME"] == comparison_opponent_name]
        if not opp_team_df.empty:
            opponent_team_abbr_for_filter = opp_team_df["TEAM_ABBREVIATION"].mode().iloc[0]
            player_game_ids_in_comp_period = set(comparison_context_df["GAME_ID"])
            opp_team_game_ids = set(all_games[all_games["TEAM_ABBREVIATION"] == opponent_team_abbr_for_filter]["GAME_ID"])
            common_games = player_game_ids_in_comp_period.intersection(opp_team_game_ids)
            comparison_context_df = comparison_context_df[comparison_context_df["GAME_ID"].isin(common_games)]
            comparison_context_desc_parts.append(f"against {comparison_opponent_name} ({opponent_team_abbr_for_filter})")
    
    if comparison_opponent_team:
        player_game_ids_in_comp_period = set(comparison_context_df["GAME_ID"])
        opp_team_game_ids = set(all_games[all_games["TEAM_ABBREVIATION"] == comparison_opponent_team]["GAME_ID"])
        common_games = player_game_ids_in_comp_period.intersection(opp_team_game_ids)
        comparison_context_df = comparison_context_df[comparison_context_df["GAME_ID"].isin(common_games)]
        comparison_context_desc_parts.append(f"against {comparison_opponent_team}")

    if comparison_series_id:
        comparison_context_df = comparison_context_df[comparison_context_df["SeriesKey"] == comparison_series_id]
        comparison_context_desc_parts.append(f"in series {comparison_series_id}")
    
    final_comparison_context_description = " ".join(comparison_context_desc_parts).strip()
    if not final_comparison_context_description and not (comparison_season or comparison_phase): # Must have some definition for period 2
        return {"error": "No comparison context specified for Period 2. Please define season/phase or apply further filters."}
    if not final_comparison_context_description and (comparison_season or comparison_phase): # Base description for period 2 if no further filters
        final_comparison_context_description = f"{comparison_season if comparison_season else 'All Seasons'}, {comparison_phase if comparison_phase else 'All Phases'}"


    if comparison_context_df["GAME_ID"].nunique() < min_games:
        return {
            "error": f"Insufficient games ({comparison_context_df['GAME_ID'].nunique()}) for Period 2 ({final_comparison_context_description}). Need at least {min_games}.",
            "overall_period_description": overall_period_description,
            "comparison_context_description": final_comparison_context_description
        }
        
    # Initialize results dictionary
    results = {
        "player_name": player_name,
        "player_team": player_team,
        "overall_period_description": overall_period_description,
        "comparison_context_description": final_comparison_context_description,
        "overall_games": overall_period_df["GAME_ID"].nunique(),
        "context_games": comparison_context_df["GAME_ID"].nunique(),
        "overall_stats": {},
        "context_stats": {},
        "statistical_tests": {}, # Renamed from paired_tests
        "performance_diff": {}
    }
    if overall_season: results["overall_season"] = overall_season
    if overall_phase: results["overall_phase"] = overall_phase
    if comparison_season: results["comparison_season"] = comparison_season
    if comparison_phase: results["comparison_phase"] = comparison_phase

    # Calculate averages and totals for both periods
    for stat in available_stats:
        # Period 1 (Overall) stats
        overall_s_vals = overall_period_df.groupby("GAME_ID")[stat].sum()
        results["overall_stats"][stat] = {
            "avg": round(overall_s_vals.mean(), 2),
            "std": round(overall_s_vals.std(), 2),
            "total": round(overall_s_vals.sum(), 2)
        }
        
        # Period 2 (Comparison Context) stats
        context_s_vals = comparison_context_df.groupby("GAME_ID")[stat].sum()
        results["context_stats"][stat] = {
            "avg": round(context_s_vals.mean(), 2),
            "std": round(context_s_vals.std(), 2),
            "total": round(context_s_vals.sum(), 2)
        }
        
        # Calculate performance difference (Context Avg - Overall Avg)
        diff = results["context_stats"][stat]["avg"] - results["overall_stats"][stat]["avg"]
        pct_diff = (diff / results["overall_stats"][stat]["avg"] * 100) if results["overall_stats"][stat]["avg"] != 0 else 0
        
        results["performance_diff"][stat] = {
            "diff": round(diff, 2),
            "pct_diff": round(pct_diff, 2),
            "is_better": diff > 0 if stat != 'TOV' else diff < 0 # TOV is better if lower
        }

        # Perform independent t-test between the two sets of game stats
        if len(overall_s_vals) >= min_games and len(context_s_vals) >= min_games:
            # Using .values ensures we pass numpy arrays to ttest_ind
            t_stat, p_value = stats.ttest_ind(context_s_vals.values, overall_s_vals.values, 
                                              equal_var=False, nan_policy='omit')
            results["statistical_tests"][stat] = {
                "t_statistic": round(t_stat, 3),
                "p_value": round(p_value, 4),
                "significant": p_value < 0.05
            }
        else:
            results["statistical_tests"][stat] = {"error": "Insufficient samples in one or both periods for t-test."}

    # Calculate shooting percentages for both periods
    shooting_results = {}
    # FG%
    if all(s in available_stats for s in ['FGM', 'FGA']):
        overall_fgp = overall_period_df['FGM'].sum() / max(overall_period_df['FGA'].sum(), 1) * 100
        context_fgp = comparison_context_df['FGM'].sum() / max(comparison_context_df['FGA'].sum(), 1) * 100
        shooting_results["FG_PCT"] = {
            "overall": round(overall_fgp, 1), "context": round(context_fgp, 1),
            "diff": round(context_fgp - overall_fgp, 1)
        }
    # FG3%
    if all(s in available_stats for s in ['FG3M', 'FG3A']):
        overall_3pp = overall_period_df['FG3M'].sum() / max(overall_period_df['FG3A'].sum(), 1) * 100
        context_3pp = comparison_context_df['FG3M'].sum() / max(comparison_context_df['FG3A'].sum(), 1) * 100
        shooting_results["FG3_PCT"] = {
            "overall": round(overall_3pp, 1), "context": round(context_3pp, 1),
            "diff": round(context_3pp - overall_3pp, 1)
        }
    # FT%
    if all(s in available_stats for s in ['FTM', 'FTA']):
        overall_ftp = overall_period_df['FTM'].sum() / max(overall_period_df['FTA'].sum(), 1) * 100
        context_ftp = comparison_context_df['FTM'].sum() / max(comparison_context_df['FTA'].sum(), 1) * 100
        shooting_results["FT_PCT"] = {
            "overall": round(overall_ftp, 1), "context": round(context_ftp, 1),
            "diff": round(context_ftp - overall_ftp, 1)
        }
    if shooting_results: results["shooting_pct"] = shooting_results
    
    return results

def calculate_team_record(all_games: pd.DataFrame,
                           team_abbr: str,
                           season: str = None,
                           phase: str = None) -> tuple:
    """
    Calculate a team's win-loss record based on player stats data.
    
    Args:
        all_games: DataFrame containing all games data
        team_abbr: Team abbreviation to calculate record for
        season: Season to filter by (optional)
        phase: Phase to filter by (optional)
        
    Returns:
        tuple: (wins, losses, total_games)
    """
    # Filter data for the team
    team_data = all_games[all_games["TEAM_ABBREVIATION"] == team_abbr].copy()
    
    # Apply season filter if specified
    if season and season != "All":
        team_data = team_data[team_data["SeasonKey"] == season]
    
    # Apply phase filter if specified
    if phase and phase != "All":
        team_data = team_data[team_data["Phase"] == phase]
    
    # If we have no data, return 0-0
    if team_data.empty:
        return 0, 0, 0
    
    # Get all game IDs for this team
    team_game_ids = team_data["GAME_ID"].unique()
    
    # Special handling for playoff records
    if phase == "Playoffs":
        # In playoffs, we want to track series results rather than individual games
        # Identify all opponents this team played against
        opponents = []
        for gid in team_game_ids:
            game_teams = all_games[all_games["GAME_ID"] == gid]["TEAM_ABBREVIATION"].unique()
            for team in game_teams:
                if team != team_abbr and team not in opponents:
                    opponents.append(team)
        
        # Calculate head-to-head record against each opponent
        total_wins = 0
        total_losses = 0
        
        for opponent in opponents:
            # Find all games between these two teams
            common_games = []
            for gid in team_game_ids:
                if opponent in all_games[all_games["GAME_ID"] == gid]["TEAM_ABBREVIATION"].unique():
                    common_games.append(gid)
            
            # Calculate head-to-head record based on points
            wins = 0
            losses = 0
            
            for gid in common_games:
                # Get points for both teams
                game_data = all_games[all_games["GAME_ID"] == gid]
                team_pts = game_data[game_data["TEAM_ABBREVIATION"] == team_abbr]["PTS"].sum()
                opp_pts = game_data[game_data["TEAM_ABBREVIATION"] == opponent]["PTS"].sum()
                
                # Determine win/loss based on actual points
                if team_pts > opp_pts:
                    wins += 1
                elif team_pts < opp_pts:
                    losses += 1
            
            # Add to total record
            if wins > 0 or losses > 0:
                total_wins += wins
                total_losses += losses
                print(f"DEBUG: {team_abbr} vs {opponent} playoff record: {wins}-{losses}")
        
        return total_wins, total_losses, total_wins + total_losses
    else:
        # Regular season: Count wins and losses based on points scored
        wins = 0
        losses = 0
        
        for gid in team_game_ids:
            # Get all teams in this game
            game_data = all_games[all_games["GAME_ID"] == gid]
            game_teams = game_data["TEAM_ABBREVIATION"].unique()
            
            # Skip if we can't find both teams
            if len(game_teams) < 2:
                continue
                
            # Find the opponent
            opponents = [t for t in game_teams if t != team_abbr]
            if not opponents:
                continue
                
            opponent = opponents[0]
            
            # Calculate points for each team
            team_pts = game_data[game_data["TEAM_ABBREVIATION"] == team_abbr]["PTS"].sum()
            opp_pts = game_data[game_data["TEAM_ABBREVIATION"] == opponent]["PTS"].sum()
            
            # Determine result based on points
            if team_pts > opp_pts:
                wins += 1
            elif team_pts < opp_pts:
                losses += 1
        
        return wins, losses, wins + losses

def compare_teams(all_games: pd.DataFrame,
                 team1_abbr: str,
                 team2_abbr: str,
                 season: str = None,
                 phase: str = None,
                 min_games: int = 5) -> dict:
    """
    Performs statistical comparison between two teams with significance testing.
    
    Args:
        all_games: DataFrame containing all game data
        team1_abbr: Abbreviation of first team to compare
        team2_abbr: Abbreviation of second team to compare
        season: Season to filter by (optional)
        phase: Phase to filter by (optional)
        min_games: Minimum games required for valid comparison
        
    Returns:
        Dictionary containing comparison results and statistical analysis
    """
    # DEBUG: Print season and phase selections
    print(f"DEBUG: Selected season: {season}, phase: {phase}")
    
    # Original, unfiltered data for each team
    orig_t1_data = all_games[all_games["TEAM_ABBREVIATION"] == team1_abbr]
    orig_t2_data = all_games[all_games["TEAM_ABBREVIATION"] == team2_abbr]
    
    # Print original game counts
    print(f"DEBUG: Original data - {team1_abbr}: {orig_t1_data['GAME_ID'].nunique()} games, {team2_abbr}: {orig_t2_data['GAME_ID'].nunique()} games")
    
    # Filter data for the specific teams and conditions
    df = all_games.copy()
    if season and season != "All":
        df = df[df["SeasonKey"] == season]
        print(f"DEBUG: After season filter - Games in DataFrame: {df['GAME_ID'].nunique()}")
    if phase and phase != "All":
        # CRITICAL FIX: Make sure we are correctly filtering by phase
        df = df[df["Phase"] == phase]
        print(f"DEBUG: After phase filter - Games in DataFrame: {df['GAME_ID'].nunique()}")
        # Verify phase filtering is working
        if phase == "Playoffs":
            print(f"DEBUG: Playoff games count: {df[df['Phase'] == 'Playoffs']['GAME_ID'].nunique()}")
    
    # Get data for each team
    t1_data = df[df["TEAM_ABBREVIATION"] == team1_abbr]
    t2_data = df[df["TEAM_ABBREVIATION"] == team2_abbr]
    
    # Print filtered game counts
    print(f"DEBUG: Filtered data - {team1_abbr}: {t1_data['GAME_ID'].nunique()} games, {team2_abbr}: {t2_data['GAME_ID'].nunique()} games")
    
    # If phase is specified, ensure we're only seeing those games
    if phase and phase != "All":
        # Get unique phases for each team in filtered data
        t1_phases = t1_data["Phase"].unique()
        t2_phases = t2_data["Phase"].unique()
        print(f"DEBUG: Filtered phases - {team1_abbr}: {t1_phases}, {team2_abbr}: {t2_phases}")
        
        # Double-check the phase filtering
        if phase == "Playoffs":
            t1_playoff_games = t1_data[t1_data["Phase"] == "Playoffs"]["GAME_ID"].nunique()
            t2_playoff_games = t2_data[t2_data["Phase"] == "Playoffs"]["GAME_ID"].nunique()
            print(f"DEBUG: Playoff games - {team1_abbr}: {t1_playoff_games}, {team2_abbr}: {t2_playoff_games}")
    
    # Check if we have enough data
    if t1_data.empty or t2_data.empty:
        return {"error": "One or both teams have no data for the selected criteria"}
    
    # Get the teams' full names
    t1_name = t1_data["TEAM_NAME"].mode().iloc[0] if not t1_data.empty else "Unknown"
    t2_name = t2_data["TEAM_NAME"].mode().iloc[0] if not t2_data.empty else "Unknown"
    
    # Define important team statistics to compare
    key_stats = [
        'PTS', 'REB', 'AST', 'STL', 'BLK', 'TOV', 'FGM', 'FGA', 'FG3M', 
        'FG3A', 'FTM', 'FTA', 'PLUS_MINUS'
    ]
    
    # Define team-specific stats that are summarized
    team_aggregate_stats = key_stats + ['OREB', 'DREB', 'PF', 'FG_PCT', 'FG3_PCT', 'FT_PCT']
    
    # Keep only stats that exist in the data
    available_stats = [stat for stat in team_aggregate_stats if stat in df.columns]
    
    # Check game count
    t1_games = t1_data["GAME_ID"].nunique()
    t2_games = t2_data["GAME_ID"].nunique()
    
    if t1_games < min_games or t2_games < min_games:
        return {
            "error": f"Insufficient games for comparison. {team1_abbr}: {t1_games} games, {team2_abbr}: {t2_games} games. Minimum required: {min_games} games."
        }

    # Calculate overall records using player stats data
    t1_wins, t1_losses, t1_total_games = calculate_team_record(all_games, team1_abbr, season, phase)
    t2_wins, t2_losses, t2_total_games = calculate_team_record(all_games, team2_abbr, season, phase)
    
    print(f"\nOVERALL RECORD CALCULATION (from player stats):")
    print(f"{team1_abbr}: {t1_wins}-{t1_losses} from {t1_total_games} games")
    print(f"{team2_abbr}: {t2_wins}-{t2_losses} from {t2_total_games} games")
    
    # Calculate head-to-head records
    # Find common game IDs
    t1_game_ids = set(t1_data["GAME_ID"].unique())
    t2_game_ids = set(t2_data["GAME_ID"].unique())
    common_games = t1_game_ids.intersection(t2_game_ids)
    
    h2h_results = []
    h2h_t1_wins = 0
    h2h_t2_wins = 0
    
    if common_games:
        print("\nHead-to-head games found:")
        for game_id in common_games:
            game_df = df[df["GAME_ID"] == game_id]
            t1_game = game_df[game_df["TEAM_ABBREVIATION"] == team1_abbr]
            t2_game = game_df[game_df["TEAM_ABBREVIATION"] == team2_abbr]
            
            if t1_game.empty or t2_game.empty:
                continue
                
            # Get points
            t1_pts = t1_game["PTS"].sum()
            t2_pts = t2_game["PTS"].sum()
            
            # Get game date
            game_date = game_df["GAME_DATE"].iloc[0] if "GAME_DATE" in game_df.columns else None
            date_str = game_date.strftime('%Y-%m-%d') if pd.notna(game_date) else "Unknown"
            
            # Determine winner
            if t1_pts > t2_pts:
                winner = team1_abbr
                h2h_t1_wins += 1
                print(f"Game {game_id}: {team1_abbr} {int(t1_pts)}-{int(t2_pts)} {team2_abbr}")
            else:
                winner = team2_abbr
                h2h_t2_wins += 1
                print(f"Game {game_id}: {team1_abbr} {int(t1_pts)}-{int(t2_pts)} {team2_abbr}")
                
            margin = abs(t1_pts - t2_pts)
            
            h2h_results.append({
                'date': date_str,
                'team1_pts': int(t1_pts),
                'team2_pts': int(t2_pts),
                'winner': winner,
                'margin': int(margin)
            })
            
        print(f"\nHead-to-head record:")
        print(f"{team1_abbr}: {h2h_t1_wins}-{h2h_t2_wins}")
        print(f"{team2_abbr}: {h2h_t2_wins}-{h2h_t1_wins}")
    else:
        print("\nNo direct head-to-head games found in the selected timeframe")
    
    # Create comparison object with the calculated records
    comparison = {
        "team1": {
            "abbr": team1_abbr,
            "name": t1_name,
            "games": t1_total_games,
            "record": f"{t1_wins}-{t1_losses}",
            "win_pct": round(t1_wins / max(t1_wins + t1_losses, 1) * 100, 1)
        },
        "team2": {
            "abbr": team2_abbr,
            "name": t2_name,
            "games": t2_total_games,
            "record": f"{t2_wins}-{t2_losses}",
            "win_pct": round(t2_wins / max(t2_wins + t2_losses, 1) * 100, 1)
        },
        "stats": {},
        "derived_stats": {},
        "statistical_tests": {},
        "pace_and_efficiency": {},
        "strength_of_schedule": {},
        "quarters_analysis": {},
        "head_to_head": {
            "games": len(h2h_results),
            "team1_wins": h2h_t1_wins,
            "team2_wins": h2h_t2_wins,
            "results": h2h_results
        }
    }
    
    # Calculate per-game averages, standard deviations, and totals for each team
    for stat in available_stats:
        # Skip if data missing
        if stat not in t1_data.columns or stat not in t2_data.columns:
            continue
        
        # Team stats need to be aggregated per game first
        t1_game_stats = t1_data.groupby('GAME_ID')[stat].sum().fillna(0)
        t2_game_stats = t2_data.groupby('GAME_ID')[stat].sum().fillna(0)
        
        # Basic statistics
        t1_avg = t1_game_stats.mean()
        t2_avg = t2_game_stats.mean()
        t1_std = t1_game_stats.std() if len(t1_game_stats) > 1 else 0
        t2_std = t2_game_stats.std() if len(t2_game_stats) > 1 else 0
        t1_total = t1_game_stats.sum()
        t2_total = t2_game_stats.sum()
        
        # Store in comparison dict
        comparison["stats"][stat] = {
            "team1_avg": round(t1_avg, 2),
            "team2_avg": round(t2_avg, 2),
            "team1_std": round(t1_std, 2),
            "team2_std": round(t2_std, 2),
            "team1_total": int(t1_total) if stat in ['PTS', 'REB', 'AST', 'STL', 'BLK', 'TOV'] else round(t1_total, 1),
            "team2_total": int(t2_total) if stat in ['PTS', 'REB', 'AST', 'STL', 'BLK', 'TOV'] else round(t2_total, 1),
            "difference": round(t1_avg - t2_avg, 2),
            "percent_diff": round(((t1_avg - t2_avg) / max(t2_avg, 0.001)) * 100, 1) if t2_avg != 0 else float('inf')
        }
        
        # Statistical significance testing (t-test for unequal variances)
        # Only if we have sufficient samples
        if len(t1_game_stats) >= min_games and len(t2_game_stats) >= min_games:
            t_stat, p_value = stats.ttest_ind(t1_game_stats, t2_game_stats, equal_var=False, nan_policy='omit')
            comparison["statistical_tests"][stat] = {
                "t_statistic": round(t_stat, 3),
                "p_value": round(p_value, 4),
                "significant": p_value < 0.05
            }
    
    # Calculate derived shooting statistics
    # Need to handle division differently for teams since we're aggregating totals
    if all(stat in available_stats for stat in ['FGM', 'FGA']):
        # Field Goal Percentage - team total
        t1_fg_total = t1_data['FGM'].sum() / max(t1_data['FGA'].sum(), 1) * 100
        t2_fg_total = t2_data['FGM'].sum() / max(t2_data['FGA'].sum(), 1) * 100
        
        # Field Goal Percentage - per game average
        t1_fg_per_game = t1_data.groupby('GAME_ID').apply(lambda x: x['FGM'].sum() / max(x['FGA'].sum(), 1) * 100).mean()
        t2_fg_per_game = t2_data.groupby('GAME_ID').apply(lambda x: x['FGM'].sum() / max(x['FGA'].sum(), 1) * 100).mean()
        
        comparison["derived_stats"]["FG_PCT"] = {
            "team1_total": round(t1_fg_total, 1),
            "team2_total": round(t2_fg_total, 1),
            "team1_per_game": round(t1_fg_per_game, 1),
            "team2_per_game": round(t2_fg_per_game, 1),
            "difference": round(t1_fg_total - t2_fg_total, 1)
        }
    
    if all(stat in available_stats for stat in ['FG3M', 'FG3A']):
        # Three Point Percentage
        t1_3p_total = t1_data['FG3M'].sum() / max(t1_data['FG3A'].sum(), 1) * 100
        t2_3p_total = t2_data['FG3M'].sum() / max(t2_data['FG3A'].sum(), 1) * 100
        
        t1_3p_per_game = t1_data.groupby('GAME_ID').apply(lambda x: x['FG3M'].sum() / max(x['FG3A'].sum(), 1) * 100).mean()
        t2_3p_per_game = t2_data.groupby('GAME_ID').apply(lambda x: x['FG3M'].sum() / max(x['FG3A'].sum(), 1) * 100).mean()
        
        comparison["derived_stats"]["FG3_PCT"] = {
            "team1_total": round(t1_3p_total, 1),
            "team2_total": round(t2_3p_total, 1),
            "team1_per_game": round(t1_3p_per_game, 1),
            "team2_per_game": round(t2_3p_per_game, 1),
            "difference": round(t1_3p_total - t2_3p_total, 1)
        }
    
    if all(stat in available_stats for stat in ['FTM', 'FTA']):
        # Free Throw Percentage
        t1_ft_total = t1_data['FTM'].sum() / max(t1_data['FTA'].sum(), 1) * 100
        t2_ft_total = t2_data['FTM'].sum() / max(t2_data['FTA'].sum(), 1) * 100
        
        t1_ft_per_game = t1_data.groupby('GAME_ID').apply(lambda x: x['FTM'].sum() / max(x['FTA'].sum(), 1) * 100).mean()
        t2_ft_per_game = t2_data.groupby('GAME_ID').apply(lambda x: x['FTM'].sum() / max(x['FTA'].sum(), 1) * 100).mean()
        
        comparison["derived_stats"]["FT_PCT"] = {
            "team1_total": round(t1_ft_total, 1),
            "team2_total": round(t2_ft_total, 1),
            "team1_per_game": round(t1_ft_per_game, 1),
            "team2_per_game": round(t2_ft_per_game, 1),
            "difference": round(t1_ft_total - t2_ft_total, 1)
        }
    
    # Team-specific metrics: Offensive and Defensive Efficiency
    # Points per 100 possessions (estimate)
    if 'PTS' in available_stats and 'FGA' in available_stats and 'FTA' in available_stats and 'TOV' in available_stats:
        # Estimate possessions using FGA, FTA, TOV
        def calc_possessions(team_df):
            possessions = (
                team_df.groupby('GAME_ID')['FGA'].sum() + 
                0.4 * team_df.groupby('GAME_ID')['FTA'].sum() - 
                1.07 * (team_df.groupby('GAME_ID')['OREB'].sum() if 'OREB' in team_df else 0) +
                team_df.groupby('GAME_ID')['TOV'].sum()
            )
            return possessions
        
        def calc_efficiency(team_df, possessions):
            # Calculate offensive efficiency
            pts = team_df.groupby('GAME_ID')['PTS'].sum()
            off_eff = (pts / possessions) * 100
            return off_eff.mean()
        
        # Get opponent stats by matching GAME_ID
        t1_game_ids = set(t1_data['GAME_ID'].unique())
        t2_game_ids = set(t2_data['GAME_ID'].unique())
        
        # For team 1
        t1_possessions = calc_possessions(t1_data)
        t1_opp_data = df[(df['GAME_ID'].isin(t1_game_ids)) & (df['TEAM_ABBREVIATION'] != team1_abbr)]
        t1_opp_possessions = calc_possessions(t1_opp_data)
        
        # For team 2
        t2_possessions = calc_possessions(t2_data)
        t2_opp_data = df[(df['GAME_ID'].isin(t2_game_ids)) & (df['TEAM_ABBREVIATION'] != team2_abbr)]
        t2_opp_possessions = calc_possessions(t2_opp_data)
        
        # Calculate offensive and defensive efficiency
        t1_off_eff = calc_efficiency(t1_data, t1_possessions)
        t1_def_eff = calc_efficiency(t1_opp_data, t1_opp_possessions)
        t2_off_eff = calc_efficiency(t2_data, t2_possessions)
        t2_def_eff = calc_efficiency(t2_opp_data, t2_opp_possessions)
        
        # Calculate pace (possessions per 48 minutes)
        try:
            t1_pace = t1_possessions.mean() * 48 / 40  # Assuming 40 minute games, adjust if needed
            t2_pace = t2_possessions.mean() * 48 / 40
        except:
            t1_pace = t1_possessions.mean() if hasattr(t1_possessions, 'mean') else 0
            t2_pace = t2_possessions.mean() if hasattr(t2_possessions, 'mean') else 0
        
        comparison["pace_and_efficiency"] = {
            "offensive_efficiency": {
                "team1": round(t1_off_eff, 2),
                "team2": round(t2_off_eff, 2),
                "difference": round(t1_off_eff - t2_off_eff, 2)
            },
            "defensive_efficiency": {
                "team1": round(t1_def_eff, 2),
                "team2": round(t2_def_eff, 2),
                "difference": round(t1_def_eff - t2_def_eff, 2)
            },
            "pace": {
                "team1": round(t1_pace, 2),
                "team2": round(t2_pace, 2),
                "difference": round(t1_pace - t2_pace, 2)
            }
        }
    
    # Point distribution analysis (margin of victory/defeat)
    t1_margins = []
    for gid in t1_data['GAME_ID'].unique():
        game_data = df[df['GAME_ID'] == gid]
        if len(game_data['TEAM_ABBREVIATION'].unique()) < 2:
            continue  # Skip if we don't have both teams
        
        team_pts = game_data[game_data['TEAM_ABBREVIATION'] == team1_abbr]['PTS'].sum()
        opp_pts = game_data[game_data['TEAM_ABBREVIATION'] != team1_abbr]['PTS'].sum()
        margin = team_pts - opp_pts
        t1_margins.append(margin)
    
    t2_margins = []
    for gid in t2_data['GAME_ID'].unique():
        game_data = df[df['GAME_ID'] == gid]
        if len(game_data['TEAM_ABBREVIATION'].unique()) < 2:
            continue  # Skip if we don't have both teams
        
        team_pts = game_data[game_data['TEAM_ABBREVIATION'] == team2_abbr]['PTS'].sum()
        opp_pts = game_data[game_data['TEAM_ABBREVIATION'] != team2_abbr]['PTS'].sum()
        margin = team_pts - opp_pts
        t2_margins.append(margin)
    
    if t1_margins and t2_margins:
        t1_avg_margin = np.mean(t1_margins)
        t2_avg_margin = np.mean(t2_margins)
        t1_close_games = sum(1 for m in t1_margins if abs(m) <= 5) / len(t1_margins) * 100
        t2_close_games = sum(1 for m in t2_margins if abs(m) <= 5) / len(t2_margins) * 100
        t1_blowouts = sum(1 for m in t1_margins if abs(m) >= 15) / len(t1_margins) * 100
        t2_blowouts = sum(1 for m in t2_margins if abs(m) >= 15) / len(t2_margins) * 100
        
        comparison["margins"] = {
            "average_margin": {
                "team1": round(t1_avg_margin, 2),
                "team2": round(t2_avg_margin, 2)
            },
            "close_games_pct": {  # Games decided by 5 or fewer points
                "team1": round(t1_close_games, 1),
                "team2": round(t2_close_games, 1)
            },
            "blowout_pct": {  # Games decided by 15+ points
                "team1": round(t1_blowouts, 1),
                "team2": round(t2_blowouts, 1)
            }
        }
    
    # Calculate consistency metrics (coefficient of variation)
    consistency = {}
    for stat in ['PTS', 'REB', 'AST']:
        if stat not in available_stats:
            continue
        
        t1_game_stats = t1_data.groupby('GAME_ID')[stat].sum()
        t2_game_stats = t2_data.groupby('GAME_ID')[stat].sum()
        
        if len(t1_game_stats) > 1 and t1_game_stats.mean() > 0:
            t1_cv = (t1_game_stats.std() / t1_game_stats.mean()) * 100
        else:
            t1_cv = 0
            
        if len(t2_game_stats) > 1 and t2_game_stats.mean() > 0:
            t2_cv = (t2_game_stats.std() / t2_game_stats.mean()) * 100
        else:
            t2_cv = 0
        
        # Lower CV = more consistent
        consistency[stat] = {
            "team1_cv": round(t1_cv, 1), 
            "team2_cv": round(t2_cv, 1),
            "more_consistent": team1_abbr if t1_cv < t2_cv else team2_abbr
        }
    
    comparison["consistency"] = consistency
    
    return comparison