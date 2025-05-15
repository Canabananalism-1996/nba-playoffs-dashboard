import streamlit as st
import pandas as pd
import difflib
from typing import Dict
import matplotlib.pyplot as plt
import numpy as np

from components.analysis import (
    points_over_games,
    plus_minus_table,
    head_to_head_record,
    get_series_box_scores,
    get_team_season_trend,
    team_game_log,
    pivot_stat_over_series,
    pts_breakdown,
    compare_players,
    compare_teams,
    player_performance_analysis
)


def _resolve_column(df: pd.DataFrame, desired: str, aliases: list[str]) -> str | None:
    """
    Return the column in df matching desired or any alias, else a close fuzzy match.
    """
    if desired in df.columns:
        return desired
    for alias in aliases:
        if alias in df.columns:
            return alias
    matches = difflib.get_close_matches(desired, df.columns, n=1, cutoff=0.6)
    return matches[0] if matches else None


def render_game_explorer(all_games: pd.DataFrame):
    """
    Sidebar + display logic for the Game Explorer.
    """
    st.sidebar.header("Game Explorer")

    # build team code → full name map
    name_map = (
        all_games.groupby("TEAM_ABBREVIATION")["TEAM_NAME"]
                 .agg(lambda s: s.dropna().mode().iat[0] if not s.dropna().mode().empty else "")
                 .to_dict()
    )
    teams = sorted(name_map.keys())
    team_label = st.sidebar.selectbox(
        "Select Team", [f"{name_map[c]} ({c})" for c in teams]
    )
    abbr = team_label.split("(")[-1].strip(")")

    seasons = sorted(all_games["SeasonKey"].dropna().unique())
    season = st.sidebar.selectbox("Select Season", seasons)

    phases = sorted(
        all_games.loc[
            (all_games["TEAM_ABBREVIATION"] == abbr) &
            (all_games["SeasonKey"] == season),
            "Phase"
        ].dropna().unique()
    )
    phase = st.sidebar.selectbox("Select Phase", phases)

    subset = all_games[
        (all_games["TEAM_ABBREVIATION"] == abbr) &
        (all_games["SeasonKey"] == season) &
        (all_games["Phase"] == phase)
    ]
    if subset.empty:
        st.warning("No games found for that selection.")
        return

    games_df = (
        subset[["GAME_ID","Away","Home","GameNum","GAME_DATE"]]
        .drop_duplicates("GAME_ID")
        .assign(
            SeriesKey=lambda df: df.apply(
                lambda r: " vs ".join(sorted([r["Away"], r["Home"]])),
                axis=1
            ),
            DateStr=lambda df: df["GAME_DATE"].dt.strftime("%d %b")
        )
    )
    games_df["label"] = (
        games_df["SeriesKey"]
        + " — " + season
        + " (" + phase + " Game "
        + games_df["GameNum"].astype("Int64").astype(str)
        + ") " + games_df["DateStr"]
    )
    labels = sorted(games_df["label"].dropna().astype(str).unique())
    game_label = st.sidebar.selectbox("Select Game", labels)
    gid = games_df.set_index("label").at[game_label, "GAME_ID"]

    st.header(f"Box Score — {game_label}")
    df = all_games[
        (all_games["GAME_ID"] == gid) &
        (all_games["TEAM_ABBREVIATION"] == abbr)
    ]
    tov_col = _resolve_column(df, "TOV", ["TO","TOVER","TOVERS"])
    if tov_col and tov_col != "TOV":
        df = df.rename(columns={tov_col: "TOV"})

    metrics = ["PLAYER_NAME", "PTS"]
    min_col = _resolve_column(df, "MIN", ["MINUTES","MINS"])
    if min_col:
        metrics.append(min_col)
    metrics += ["REB","AST","STL","BLK"]
    if "TOV" in df.columns:
        metrics.append("TOV")

    disp = df[metrics].copy()
    for c in metrics:
        if pd.api.types.is_numeric_dtype(disp[c]):
            disp[c] = disp[c].fillna(0).astype(int)

    st.subheader("Player Box Score")
    # Reset index and use 1-based indexing
    disp = disp.reset_index(drop=True)
    disp.index = disp.index + 1
    st.dataframe(disp, use_container_width=True)

    st.subheader("Team Totals")
    full = all_games[all_games["GAME_ID"] == gid]
    team_mets = ["PTS","REB","AST","STL","BLK"]
    if "TOV" in full.columns:
        team_mets.append("TOV")
    totals = (
        full.groupby("TEAM_ABBREVIATION")[team_mets]
            .sum()
            .astype(int)
            .reset_index()
    )
    st.table(totals)


def render_head_to_head(all_games: pd.DataFrame):
    """
    1) Team A/B, Season, Phase, Game dropdowns
    2) Winner label
    3) Box Score tables (serials start at 1; columns: GAME_ID, PLAYER_ID,
       COMMENT, GAME_DATE(day Month), NICKNAME_ADV)
    4) Points Over Games chart
    5) Plus-Minus table
    6) Record & total plus-minus
    """
    st.sidebar.header("Head-to-Head")

    # 1) team code→name map
    name_map = (
        all_games
          .groupby("TEAM_ABBREVIATION")["TEAM_NAME"]
          .agg(lambda s: s.dropna().mode().iat[0] if not s.dropna().mode().empty else "")
          .to_dict()
    )
    codes = sorted(name_map)

    # Team A & Team B
    a_label = st.sidebar.selectbox("Team A", [f"{name_map[c]} ({c})" for c in codes])
    b_label = st.sidebar.selectbox(
        "Team B",
        [f"{name_map[c]} ({c})" for c in codes],
        index=1 if len(codes) > 1 else 0,
    )
    a_abbr = a_label.split("(")[-1].strip(")")
    b_abbr = b_label.split("(")[-1].strip(")")

    # Season & Phase
    seasons = ["All"] + sorted(all_games["SeasonKey"].dropna().unique())
    season  = st.sidebar.selectbox("Season", seasons)
    phases  = ["All"] + sorted(all_games["Phase"].dropna().unique())
    phase   = st.sidebar.selectbox("Phase", phases)

    # filter games
    df = all_games.copy()
    if season != "All":
        df = df[df["SeasonKey"] == season]
    if phase != "All":
        df = df[df["Phase"] == phase]

    # find common GAME_IDs
    ga = set(df[df["TEAM_ABBREVIATION"] == a_abbr]["GAME_ID"])
    gb = set(df[df["TEAM_ABBREVIATION"] == b_abbr]["GAME_ID"])
    common = sorted(ga & gb)
    if not common:
        st.warning("No common games found.")
        return

    # Game dropdown meta
    games_meta = (
        df[df["GAME_ID"].isin(common)]
          .drop_duplicates("GAME_ID")[["GAME_ID","SeasonKey","Phase","GameNum","GAME_DATE"]]
          .sort_values("GAME_DATE", ignore_index=True)
    )
    games_meta = games_meta.assign(
        label=lambda d: (
            d["SeasonKey"] + " " + d["Phase"]
            + " Game " + d["GameNum"].astype(int).astype(str)
            + " — " + d["GAME_DATE"].dt.strftime("%d %b")
        )
    )
    game_label = st.sidebar.selectbox("Select Game", games_meta["label"].tolist())
    gid = games_meta.set_index("label").at[game_label, "GAME_ID"]

    # 2) Winner line
    df_w = df[df["GAME_ID"] == gid]
    # pick A's WL
    wl = df_w[df_w["TEAM_ABBREVIATION"] == a_abbr]["WL"].iat[0]
    winner = a_label if wl == "W" else b_label
    st.subheader(f"Winner : {winner}")


    # ─── Matchup Results Table ────────────────────────────────────────────────
    st.subheader("Matchup Results")

    # Build a small pivot of total PTS by team & game
    pts_pivot = (
        all_games[all_games["GAME_ID"].isin(common)]
          .pivot_table(
              index="GAME_ID",
              columns="TEAM_ABBREVIATION",
              values="PTS",
              aggfunc="sum"
          )
          .loc[games_meta["GAME_ID"], [a_abbr, b_abbr]]  # ensure chronological order
    )

    rows = []
    for gid, row in pts_pivot.iterrows():
        a_pts, b_pts = int(row[a_abbr]), int(row[b_abbr])
        matchup = f"{a_abbr} vs {b_abbr}"
        winner = a_label if a_pts > b_pts else b_label
        score = f"{a_pts}–{b_pts}"
        phase = games_meta.set_index("GAME_ID").at[gid,"Phase"]
        date = games_meta.set_index("GAME_ID").at[gid,"GAME_DATE"].strftime("%d %b")
        Seasonkey = games_meta.set_index("GAME_ID").at[gid,"SeasonKey"]

        rows.append({
            "MATCHUP": matchup,
            "Winner":   winner,
            "Score":    score,
            "Date": date,
            "Season": Seasonkey,
            "Phase" : phase
        })
    results_df = pd.DataFrame(rows)
    st.table(results_df)


     # 7) Box Score for the selected game
    st.header(f"{a_label} vs {b_label} — Box Score\n{game_label}")
    df_game = df[df["GAME_ID"] == gid].copy()

    # drop truly useless IDs
    df_game.drop(columns=["TEAM_ID","TEAM_ID_ADV"], errors="ignore", inplace=True)
    # format date
    df_game["GAME_DATE"] = pd.to_datetime(df_game["GAME_DATE"]).dt.strftime("%d %b")

    # list of cols to *exclude*
    irrelevant = {
      "GAME_ID", "PLAYER_ID", "COMMENT", 
      "NICKNAME_ADV", "SEASON_ID","TEAM_ID_ADV","TEAM_CITY_ADV",
      "START_POSITION_ADV","COMMENT_ADV","MIN_ADV","WL","TEAM_ABBREVIATION_meta", "TEAM NAME","TEAM_CITY","PLUS_MINUS", "MATCHUP","TEAM_ABBREVIATION"
    }
    # everything else is "relevant"
    keep = [c for c in df_game.columns if c not in irrelevant]
    front = ["GAME_DATE","SeasonKey","SeriesKey","GameNum"]
    keep_cols = front + [c for c in keep if c not in front]

    # Team A
    st.subheader(f"{a_label} Box Score")
    dfA = (
      df_game.loc[df_game["TEAM_ABBREVIATION"] == a_abbr, keep_cols]
      .reset_index(drop=True)
    )
    dfA.index = dfA.index + 1           # 1-based serials
    st.dataframe(dfA, use_container_width=True)

    # Team B
    st.subheader(f"{b_label} Box Score")
    dfB = (
      df_game.loc[df_game["TEAM_ABBREVIATION"] == b_abbr, keep_cols]
      .reset_index(drop=True)
    )
    dfB.index = dfB.index + 1
    st.dataframe(dfB, use_container_width=True)
    st.subheader(f"Record and Total Plus-Min\n{season}, {phase}")

    # 4–6) delegate to analysis.py
    pivot = points_over_games(df, a_abbr, b_abbr, games_meta)
    diff  = plus_minus_table(pivot, a_abbr, b_abbr,)
    head_to_head_record(diff, a_label, b_label)


def get_opponent_from_matchup(matchup, player_team):
    """Helper function to correctly extract the opponent team from a matchup string.
    
    Args:
        matchup (str): The matchup string (e.g., "LAL vs GSW" or "LAL @ BOS")
        player_team (str): The player's team code (e.g., "LAL")
        
    Returns:
        str: The opponent team code
    """
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
        return [p for p in parts if p != player_team and len(p) <= 3][0]
    
    # Return the team that isn't the player's team
    return teams[1] if teams[0] == player_team else teams[0]


def render_player_explorer(all_games: pd.DataFrame):
    """
    Render the player explorer view with consistent formatting.
    All numeric values are rounded to 2 decimal places for display.
    Games played count is based on actual games where the player appeared in the box score.
    Opponent is correctly identified as the team playing against the player's team.
    """
    st.sidebar.header("Player Explorer")

    # 1) Player & Season selects
    players = sorted(all_games["PLAYER_NAME"].dropna().unique())
    plyr    = st.sidebar.selectbox("Player", players)
    seasons = sorted(all_games["SeasonKey"].dropna().unique())
    season  = st.sidebar.selectbox("Season", seasons)

    # 2) Filter to this player & season
    df = all_games.query("PLAYER_NAME == @plyr and SeasonKey == @season").copy()
    if df.empty:
        st.warning("No games found for this player/season.")
        return

    # 3) Determine player's own team from their box-score rows
    main_abbr = df["TEAM_ABBREVIATION"].mode().iat[0]
    name_map  = (
        all_games
        .groupby("TEAM_ABBREVIATION")["TEAM_NAME"]
        .agg(lambda s: s.dropna().mode().iat[0] if not s.dropna().mode().empty else "")
        .to_dict()
    )
    main_name = name_map[main_abbr]
    df["PLAYER_TEAM_ABBR"] = main_abbr
    df["PLAYER_TEAM_NAME"] = main_name
    
    # 4) Games Played 
    games_played = df["GAME_ID"].nunique()
    
    st.subheader("Games Played by Team")
    
    # Create summary DataFrame
    summary_df = pd.DataFrame([{
        "PLAYER_NAME":       plyr,
        "TEAM_ABBREVIATION": main_abbr,
        "TEAM_NAME":         main_name,
        "GamesPlayed":       games_played
    }])
    
    st.table(summary_df)

    # 5) Opponent Selector
    # Fix: Ensure opponent is CORRECTLY identified as the team that is NOT the player's team
    match_clean = df["MATCHUP"].str.replace(r"\s*@\s*", " vs ", regex=True)
    split_df = match_clean.str.split(" vs ", n=1, expand=True)
    
    # Properly identify the opponent (the team that's NOT the player's team)
    df["OPPONENT"] = split_df.apply(
        lambda row: row[1] if row[0] == main_abbr else row[0], 
        axis=1
    )
    
    # Additional cleaning to extract just the opponent's team code
    df["OPPONENT"] = df["OPPONENT"].str.replace(r"vs\.\s*", "", regex=True).str.strip()
    df["OPPONENT"] = df["OPPONENT"].apply(
        lambda x: x.split()[0] if " " in x else x
    )
    
    # IMPORTANT: Make sure LAL (or player's own team) is NEVER listed as opponent
    df.loc[df["OPPONENT"] == main_abbr, "OPPONENT"] = df.loc[df["OPPONENT"] == main_abbr].apply(
        lambda row: get_opponent_from_matchup(row["MATCHUP"], main_abbr), axis=1
    )
    
    opponents = sorted(df["OPPONENT"].dropna().unique())
    opp_sel = st.sidebar.selectbox("Select Opponent Team", ["(All)"] + opponents)

    # 6) Game Log — Basic Stats
    st.subheader("Game Log — Basic Stats")
    tov_col = _resolve_column(df, "TOV", ["TO","TOVER","TOVERS"])
    cols1 = ["MATCHUP","GAME_DATE","SeasonKey","Phase","PLAYER_TEAM_NAME",
             "PTS","REB","AST","STL","BLK"]
    if tov_col:
        cols1.append(tov_col)

    log1 = df[cols1].copy()
    # Add opponent consistently
    log1["Opponent"] = df["OPPONENT"]
    log1["GAME_DATE"] = pd.to_datetime(log1["GAME_DATE"]).dt.strftime("%d %b")
    
    # Format numeric columns to 2 decimal places
    numeric_cols = ["PTS","REB","AST","STL","BLK"]
    if tov_col:
        numeric_cols.append(tov_col)
    
    for col in numeric_cols:
        if col in log1.columns:
            log1[col] = log1[col].round(2)
    
    if opp_sel != "(All)":
        log1 = log1[log1["Opponent"] == opp_sel]

    out1_cols = ["Opponent","GAME_DATE","SeasonKey","Phase",
                 "PLAYER_TEAM_NAME","PTS","REB","AST","STL","BLK"]
    if tov_col:
        out1_cols.append(tov_col)
    
    # Prepare dataframe with 1-based indexing
    display_log = log1[out1_cols].sort_values("GAME_DATE").reset_index(drop=True)
    display_log.index = display_log.index + 1
    
    st.dataframe(display_log, use_container_width=True)

    # 7) Total Stats
    st.subheader(
        "Total Stats — " + ("vs " + opp_sel if opp_sel != "(All)" else f"for {main_name}")
    )
    sum_cols = ["PTS","REB","AST","STL","BLK"]
    if tov_col:
        sum_cols.append(tov_col)
    if opp_sel != "(All)":
        df_tot = log1
        df_tot = df_tot.groupby("Opponent", as_index=False)[sum_cols].sum()
    else:
        df_tot = df[df["TEAM_ABBREVIATION"] == main_abbr].groupby(
            "TEAM_ABBREVIATION", as_index=False
        )[sum_cols].sum()
        df_tot["TEAM_NAME"] = main_name
        df_tot = df_tot[["TEAM_NAME"] + sum_cols]
    
    # Format all numeric columns to 2 decimal places
    for col in sum_cols:
        if col in df_tot.columns:
            df_tot[col] = df_tot[col].round(2)
    
    st.table(df_tot)

    # 8) Points vs Opponents
    st.subheader("Points vs Opponents")
    # Fix: Use the corrected OPPONENT column from above
    # Ensure the OPPONENT column is used consistently
    pts_vs = (
        df[df["TEAM_ABBREVIATION"] == main_abbr]  # Only include stats when player is on their main team
          .groupby("OPPONENT", as_index=False)["PTS"]
          .sum()
          .rename(columns={"PTS": "TotalPTS"})
          .loc[lambda d: ~d["OPPONENT"].isin([main_abbr, ""])]  # Exclude empty opponent entries and own team
          .sort_values("OPPONENT")  # Sort for better readability
    )
    
    # Format TotalPTS to 2 decimal places
    pts_vs["TotalPTS"] = pts_vs["TotalPTS"].round(2)
    
    st.table(pts_vs)

    # 9) Series Viewer
    st.subheader("Series Viewer")
    
    # Create clean matchup labels using the standard team codes
    # First, ensure all matchups use the same format
    df["CleanMatchup"] = df.apply(
        lambda row: f"{main_abbr} vs {row['OPPONENT']}" if row["OPPONENT"] != main_abbr else f"{row['OPPONENT']} vs {main_abbr}",
        axis=1
    )
    
    # Create series of unique, clean matchups for the dropdown
    series_list = sorted(df["CleanMatchup"].dropna().unique().tolist())
    series_sel = st.selectbox("Select Series (Opponent)", series_list)
    
    # Filter games based on the selected matchup
    ser = df[df["CleanMatchup"] == series_sel].sort_values("GameNum").copy()
    
    # Format numeric columns to 2 decimal places
    for col in ser.columns:
        if pd.api.types.is_numeric_dtype(ser[col]) and col not in ["GAME_ID", "PLAYER_ID", "GameNum"]:
            ser[col] = ser[col].round(2)
    
    # Reset index and make it 1-based
    display_ser = ser.reset_index(drop=True)
    display_ser.index = display_ser.index + 1
    
    st.dataframe(display_ser, use_container_width=True)

def render_team_explorer(all_games: pd.DataFrame):
    """
    Render the team explorer view with clean UI and accurate team data.
    
    Fix: Added error handling for missing or NaN values
    """
    st.sidebar.header("Team Explorer")
    
    # Create team name mapping to ensure consistent display
    name_map = (
        all_games
        .groupby("TEAM_ABBREVIATION")["TEAM_NAME"]
        .agg(lambda s: s.dropna().mode().iat[0] if not s.dropna().mode().empty else "")
        .to_dict()
    )
    
    # Team selector with team codes
    teams = sorted(name_map.keys())
    team_label = st.sidebar.selectbox(
        "Select Team", 
        [f"{name_map[c]} ({c})" for c in teams]
    )
    team = team_label.split("(")[-1].strip(")")
    
    # Season & Phase selectors (without "All" option)
    seasons = sorted(all_games["SeasonKey"].dropna().unique())
    season = st.sidebar.selectbox("Season", seasons)
    
    # Filter data for the selected team and season
    df_team = all_games[
        (all_games["TEAM_ABBREVIATION"] == team) &
        (all_games["SeasonKey"] == season)
    ]
    
    if df_team.empty:
        st.warning("No games found for this team and season.")
        return
    
    # Phase selection based on filtered data
    phases = sorted(df_team["Phase"].dropna().unique())
    phase = st.sidebar.selectbox("Phase", phases)
    
    # Further filter by phase
    df_team = df_team[df_team["Phase"] == phase]
    
    if df_team.empty:
        st.warning("No games found for this team, season, and phase combination.")
        return
    
    # Extract all opponents accurately with defensive coding
    df_team["MATCHUP_CLEAN"] = df_team["MATCHUP"].str.replace(r"\s*@\s*", " vs ", regex=True)
    
    # Handle potential NaN values in MATCHUP_CLEAN before splitting
    df_team = df_team.dropna(subset=["MATCHUP_CLEAN"])
    if df_team.empty:
        st.warning("Missing matchup data for selected criteria.")
        return
        
    split_df = df_team["MATCHUP_CLEAN"].str.split(" vs ", n=1, expand=True)
    
    # Properly identify the opponent with NaN handling
    df_team["OPPONENT"] = split_df.apply(
        lambda row: row[1] if pd.notna(row[0]) and row[0] == team else row[0] if pd.notna(row[1]) else None, 
        axis=1
    )
    
    # Drop rows with null opponents
    df_team = df_team.dropna(subset=["OPPONENT"])
    if df_team.empty:
        st.warning("Unable to determine opponents for selected criteria.")
        return
    
    # Clean opponent codes
    df_team["OPPONENT"] = df_team["OPPONENT"].str.replace(r"vs\.\s*", "", regex=True).str.strip()
    df_team["OPPONENT"] = df_team["OPPONENT"].apply(
        lambda x: x.split()[0] if pd.notna(x) and " " in x else x
    )
    
    # Fix: Make sure team's own code is NEVER listed as opponent using safe accessor
    is_self_opponent = df_team["OPPONENT"] == team
    if any(is_self_opponent):
        # Use a safe approach to extract opponent from matchup
        for idx in df_team.loc[is_self_opponent].index:
            match_val = df_team.at[idx, "MATCHUP"]
            if pd.isna(match_val):
                df_team.at[idx, "OPPONENT"] = None
                continue
                
            parts = match_val.replace("@", "vs").split("vs")
            parts = [p.strip() for p in parts]
            df_team.at[idx, "OPPONENT"] = next((p for p in parts if p != team), None)
    
    # Filter rows with valid opponents
    df_team = df_team.dropna(subset=["OPPONENT"])
    if df_team.empty:
        st.warning("No valid opponent data for selected criteria.")
        return
    
    # Get unique opponents
    opponents = sorted(df_team["OPPONENT"].dropna().unique())
    
    # Header with team details
    st.header(f"{name_map[team]} ({team}) — Game Log")
    
    # Display game log without opponent selection first
    # This is a temporary log to show all games for this team
    temp_log = team_game_log(all_games, team, "", season, phase)
    
    # Handle case with no games
    if temp_log.empty:
        st.warning(f"No games found for {team} in the selected season and phase.")
        return
    
    st.dataframe(temp_log, use_container_width=True)
    
    # MOVED: Opponent selector to main content area after game log
    st.subheader("Select Series Details")
    
    # Check if we have opponents to show
    if not opponents:
        st.warning("No opponents found for this team in the selected season and phase.")
        return
        
    # Opponent selector is now in the main content area
    opp = st.selectbox("Select Opponent", opponents)
    
    # Get all series between these teams for this season
    series_df = get_series_box_scores(all_games, team, opp, season, phase)
    
    # Series and game selectors
    if series_df.empty:
        st.warning(f"No games found between {team} and {opp} in the selected season and phase.")
        return
        
    series_keys = sorted(series_df["SeriesKey"].unique())
    series_picker = st.selectbox("Select Series", series_keys)
    
    # Get games in this series with defensive coding
    games_in_series = sorted(
        series_df
        .loc[series_df["SeriesKey"] == series_picker, "GameNum"]
        .dropna()
        .astype(int)
        .unique()
    )
    
    if not games_in_series:
        st.warning(f"No games found in series {series_picker}.")
        return
        
    game_picker = st.selectbox("Select Game", games_in_series)
    
    # Get box score for the selected game
    box_df = series_df[
        (series_df["SeriesKey"] == series_picker) &
        (series_df["GameNum"] == game_picker)
    ]
    
    if box_df.empty:
        st.warning(f"No data found for Game {game_picker} in series {series_picker}.")
        return
    
    # Show box score for selected game in series
    st.subheader(f"{name_map[team]} ({team}) Box Score")
    
    # Handle potential missing columns
    drop_cols = [
        "SeriesKey", "GameNum", "MATCHUP", "WL",
        "TEAM_ABBREVIATION_meta", "TEAM_NAME",
        "Away", "Home", "TEAM_ID", "GAME_ID", "PLAYER_ID",
        "COMMENT", "MIN_ADV", "TEAM_ID_ADV", "NICKNAME_ADV",
        "COMMENT_ADV", "TEAM_CITY_ADV", "START_POSITION_ADV", "GAME_DATE"
    ]
    
    # Only drop columns that actually exist
    cols_to_drop = [col for col in drop_cols if col in box_df.columns]
    
    team_box = box_df[box_df["TEAM_ABBREVIATION"] == team].drop(columns=cols_to_drop, errors='ignore').reset_index(drop=True)
    team_box.index = team_box.index + 1
    st.dataframe(team_box, use_container_width=True)

    st.subheader(f"{name_map[opp]} ({opp}) Box Score")
    opp_box = box_df[box_df["TEAM_ABBREVIATION"] == opp].drop(columns=cols_to_drop, errors='ignore').reset_index(drop=True)
    opp_box.index = opp_box.index + 1
    st.dataframe(opp_box, use_container_width=True)
    
    # --- Dynamic Stat Selector & Charts ---
    # pick only stats available in this series
    default_stats = ["PTS", "FGM", "FG3M", "FTM", "REB", "AST", "STL", "BLK", "TOV"]
    stats_options = [s for s in default_stats if s in series_df.columns]
    
    if not stats_options:
        st.warning("No statistics available for comparison.")
        return
        
    stat = st.selectbox("Select Statistic to Compare", stats_options)

    # build list of games and map them to game numbers
    log_df = team_game_log(all_games, team, opp, season, phase)
    
    if log_df.empty:
        st.warning("No game log data available for comparison.")
        return
        
    series_games = log_df["GAME_ID"].tolist()
    
    # Check if we have any games to map
    if not series_games:
        st.warning("No games found for comparison.")
        return
        
    # Create mapping with defensive coding
    mapping_df = series_df[["GAME_ID", "GameNum"]].drop_duplicates()
    if mapping_df.empty:
        st.warning("Missing game number mapping data.")
        return
        
    mapping = mapping_df.set_index("GAME_ID")
    valid_games = [gid for gid in series_games if gid in mapping.index]

    if not valid_games:
        st.warning("No games with valid data available for comparison")
        return
        
    # pivot the chosen stat across both teams
    df_stat = all_games[all_games["GAME_ID"].isin(valid_games)]
    
    # Ensure both teams have data
    if not all(team in df_stat["TEAM_ABBREVIATION"].values for team in [team, opp]):
        st.warning(f"Missing team data for either {team} or {opp}")
        return
        
    # Create pivot with defensive coding
    try:
        pivot = (
            df_stat.pivot_table(
                index="GAME_ID",
                columns="TEAM_ABBREVIATION",
                values=stat,
                aggfunc="sum",
            )
            .reindex(valid_games)  # Use reindex instead of loc for safety
            .fillna(0)
        )
        
        # Check if both teams present in pivot columns
        if team not in pivot.columns or opp not in pivot.columns:
            pivot_cols = list(pivot.columns)
            st.warning(f"Missing team data in pivot. Available teams: {pivot_cols}")
            return
            
        # Get the specific columns we need
        pivot = pivot[[team, opp]]
        
        # swap x-axis from GAME_ID to GameNum safely
        new_index = []
        for gid in pivot.index:
            try:
                new_index.append(mapping.at[gid, "GameNum"])
            except (KeyError, ValueError):
                new_index.append(None)
                
        # Skip any None values
        if all(x is None for x in new_index):
            st.warning("Unable to map game IDs to game numbers")
            return
            
        pivot.index = pd.Index(new_index, name="Game").astype(str)

        st.subheader(f"{stat} Comparison Over Series")
        st.line_chart(pivot)

        # if we're looking at total points, show the internal breakdown
        if stat == "PTS":
            try:
                bd_raw = pts_breakdown(all_games, team, valid_games)
                if bd_raw.empty:
                    st.warning("No points breakdown data available")
                    return
                    
                bd = bd_raw.reindex(valid_games).fillna(0)
                
                # Map game IDs to game numbers safely
                new_bd_index = []
                for gid in bd.index:
                    try:
                        new_bd_index.append(mapping.at[gid, "GameNum"])
                    except (KeyError, ValueError):
                        new_bd_index.append(None)
                
                if all(x is None for x in new_bd_index):
                    st.warning("Unable to map game IDs to game numbers for points breakdown")
                    return
                    
                bd.index = new_bd_index
                bd.index.name = "Game"

                st.subheader(f"{team} Points Breakdown by Game")
                st.bar_chart(bd)
            except Exception as e:
                st.warning(f"Error generating points breakdown: {str(e)}")
                
    except Exception as e:
        st.warning(f"Error creating stat comparison: {str(e)}")

def render_player_comparison(all_games: pd.DataFrame):
    """
    Render the player vs player comparison view with statistical analysis.
    
    Args:
        all_games: DataFrame containing all game stats
    """
    st.sidebar.header("Player Comparison")
    
    # Player selection
    players = sorted(all_games["PLAYER_NAME"].dropna().unique())
    
    col1, col2 = st.sidebar.columns(2)
    with col1:
        player1 = st.selectbox("Player 1", players, key="player1")
    with col2:
        # Default player2 to the second player or first if only one exists
        default_idx = 1 if len(players) > 1 else 0
        player2 = st.selectbox("Player 2", players, index=default_idx, key="player2")
    
    # Season & Phase selectors
    seasons = ["All"] + sorted(all_games["SeasonKey"].dropna().unique())
    season = st.sidebar.selectbox("Season", seasons)
    
    phases = ["All"] + sorted(all_games["Phase"].dropna().unique())
    phase = st.sidebar.selectbox("Phase", phases)
    
    # Set minimum games threshold
    min_games = st.sidebar.slider("Minimum Games", 1, 20, 5)
    
    # Run comparison when button is clicked
    if st.sidebar.button("Compare Players"):
        st.header(f"Player Comparison: {player1} vs {player2}")
        
        # Verify that the required column TEAM_ABBREVIATION exists
        if "TEAM_ABBREVIATION" not in all_games.columns:
            st.error("Critical error: TEAM_ABBREVIATION column is missing from data. Cannot perform comparison.")
            return
            
        # Check if both players exist in the dataset
        player1_exists = any(all_games["PLAYER_NAME"] == player1)
        player2_exists = any(all_games["PLAYER_NAME"] == player2)
        
        if not player1_exists or not player2_exists:
            missing_players = []
            if not player1_exists:
                missing_players.append(player1)
            if not player2_exists:
                missing_players.append(player2)
            st.error(f"The following player(s) are not found in the dataset: {', '.join(missing_players)}")
            return
        
        with st.spinner("Analyzing player statistics..."):
            try:
                comparison = compare_players(
                    all_games=all_games,
                    player1_name=player1,
                    player2_name=player2,
                    season=season,
                    phase=phase,
                    min_games=min_games
                )
            except KeyError as e:
                error_key = str(e).strip("''")
                if error_key == "TEAM_ABBREVIATION":
                    st.error("Error: 'TEAM_ABBREVIATION' column not found in one of the player data subsets. This may be due to data inconsistency when filtering.")
                    st.info("Debugging tip: Check if the data for these players contains the 'TEAM_ABBREVIATION' key in all entries.")
                else:
                    st.error(f"Error in data processing: KeyError - {e}. This might be due to missing data for one of the players.")
                return
            except Exception as e:
                st.error(f"Unexpected error: {str(e)}")
                return
        
        # Check for errors in comparison
        if "error" in comparison:
            st.error(comparison["error"])
            return
        
        # --- Player Information Section ---
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader(f"{comparison['player1']['name']} ({comparison['player1']['team']})")
            st.write(f"Games Played: {comparison['player1']['games']}")
            # Display record if available
            if "record" in comparison["player1"]:
                st.write(f"Team Record: {comparison['player1']['record']} ({comparison['player1']['win_pct']}%)")
        
        with col2:
            st.subheader(f"{comparison['player2']['name']} ({comparison['player2']['team']})")
            st.write(f"Games Played: {comparison['player2']['games']}")
            # Display record if available
            if "record" in comparison["player2"]:
                st.write(f"Team Record: {comparison['player2']['record']} ({comparison['player2']['win_pct']}%)")
        
        # Additional context about the records
        if "record" in comparison["player1"] and "record" in comparison["player2"]:
            season_phase_context = []
            if season != "All":
                season_phase_context.append(f"season {season}")
            if phase != "All":
                season_phase_context.append(f"{phase.lower()}")
            
            context_str = " and ".join(season_phase_context)
            if context_str:
                st.write(f"*Team records shown are for games when the player appeared in {context_str} only.*")
            else:
                st.write(f"*Team records shown are for games when the player appeared.*")
        
        # --- Basic Stats Comparison ---
        st.subheader("Basic Stats Comparison (Per Game)")
        
        # Create main stats dataframe
        basic_stats = ['PTS', 'REB', 'AST', 'STL', 'BLK']
        if 'TOV' in comparison['stats']:
            basic_stats.append('TOV')
        
        stats_data = []
        for stat in basic_stats:
            if stat in comparison['stats']:
                stat_info = comparison['stats'][stat]
                stats_data.append({
                    'Statistic': stat,
                    f"{player1}": f"{stat_info['player1_avg']} ± {stat_info['player1_std']}",
                    f"{player2}": f"{stat_info['player2_avg']} ± {stat_info['player2_std']}",
                    'Difference': stat_info['difference'],
                    'Significant': "✓" if stat in comparison['statistical_tests'] and 
                                  comparison['statistical_tests'][stat]['significant'] else "✗"
                })
        
        stats_df = pd.DataFrame(stats_data)
        st.dataframe(stats_df, use_container_width=True)
        
        # --- Shooting Percentages ---
        if comparison['derived_stats']:
            st.subheader("Shooting Percentages")
            
            pct_data = []
            for pct_type, pct_info in comparison['derived_stats'].items():
                label = {
                    'FG_PCT': 'Field Goal %', 
                    'FG3_PCT': '3-Point %', 
                    'FT_PCT': 'Free Throw %'
                }.get(pct_type, pct_type)
                
                pct_data.append({
                    'Type': label,
                    f"{player1}": pct_info['player1'],  # Store numeric values
                    f"{player2}": pct_info['player2'],  # Store numeric values
                    'Difference': pct_info['difference']  # Store numeric values
                })
            
            pct_df = pd.DataFrame(pct_data)
            # Reset index and use 1-based indexing
            pct_df = pct_df.reset_index(drop=True)
            pct_df.index = pct_df.index + 1
            
            # Format numbers in the display
            st.dataframe(pct_df.style.format({
                f"{player1}": "{:.1f}%",
                f"{player2}": "{:.1f}%",
                'Difference': "{:.1f}%"
            }), use_container_width=True)
        
        # --- Visualization of Key Stats ---
        st.subheader("Key Stats Comparison")
        
        # Prepare data for horizontal bar chart
        key_stats = ['PTS', 'REB', 'AST']
        key_stat_data = {stat: [] for stat in key_stats if stat in comparison['stats']}
        
        for stat in key_stat_data:
            key_stat_data[stat] = [
                comparison['stats'][stat]['player1_avg'],
                comparison['stats'][stat]['player2_avg']
            ]
        
        # Create streamlit columns for charts
        chart_cols = st.columns(len(key_stat_data))
        
        # Plot each stat side by side
        for i, (stat, values) in enumerate(key_stat_data.items()):
            with chart_cols[i]:
                # Create a simple bar chart
                chart_data = pd.DataFrame({
                    'Player': [player1, player2],
                    stat: values
                })
                st.write(f"{stat} Per Game")
                st.bar_chart(chart_data, x='Player', y=stat, use_container_width=True)
        
        # --- Performance Thresholds ---
        if comparison['performance_thresholds']:
            st.subheader("Performance Thresholds")
            st.write("Percentage of games with at least X points/rebounds/assists:")
            
            # Create tabs for different stats
            tabs = st.tabs(['Points', 'Rebounds', 'Assists'])
            
            for i, stat in enumerate(['PTS', 'REB', 'AST']):
                if stat not in comparison['performance_thresholds']:
                    continue
                    
                with tabs[i]:
                    threshold_data = []
                    for threshold, thresh_info in comparison['performance_thresholds'][stat].items():
                        threshold_data.append({
                            f"{stat} ≥": threshold,
                            f"{player1} (% of games)": thresh_info['player1_pct'],  # Store numeric value
                            f"{player2} (% of games)": thresh_info['player2_pct'],  # Store numeric value
                            'Difference': thresh_info['difference']  # Store numeric value
                        })
                    
                    thresh_df = pd.DataFrame(threshold_data)
                    # Reset index and use 1-based indexing
                    thresh_df = thresh_df.reset_index(drop=True)
                    thresh_df.index = thresh_df.index + 1
                    
                    # Format percentage columns
                    format_dict = {
                        f"{player1} (% of games)": "{:.1f}%",
                        f"{player2} (% of games)": "{:.1f}%",
                        'Difference': "{:.1f}%"
                    }
                    
                    st.dataframe(thresh_df.style.format(format_dict), use_container_width=True)
        
        # --- Consistency Analysis ---
        if comparison['consistency']:
            st.subheader("Consistency Analysis")
            st.write("Coefficient of Variation (lower = more consistent)")
            
            consistency_data = []
            for stat, consist_info in comparison['consistency'].items():
                stat_label = {'PTS': 'Points', 'REB': 'Rebounds', 'AST': 'Assists'}.get(stat, stat)
                
                consistency_data.append({
                    'Statistic': stat_label,
                    f"{player1} CV": consist_info['player1_cv'],  # Store numeric value
                    f"{player2} CV": consist_info['player2_cv'],  # Store numeric value
                    'More Consistent Player': consist_info['more_consistent']
                })
            
            consist_df = pd.DataFrame(consistency_data)
            # Reset index and use 1-based indexing
            consist_df = consist_df.reset_index(drop=True)
            consist_df.index = consist_df.index + 1
            
            # Format percentage columns
            format_dict = {
                f"{player1} CV": "{:.1f}%",
                f"{player2} CV": "{:.1f}%"
            }
            
            st.dataframe(consist_df.style.format(format_dict), use_container_width=True)
            
            # Highlight the player who is more consistent overall
            p1_wins = sum(1 for info in comparison['consistency'].values() 
                          if info['more_consistent'] == player1)
            p2_wins = sum(1 for info in comparison['consistency'].values() 
                          if info['more_consistent'] == player2)
            
            if p1_wins > p2_wins:
                st.info(f"Overall, {player1} shows more consistency across key statistics.")
            elif p2_wins > p1_wins:
                st.info(f"Overall, {player2} shows more consistency across key statistics.")
            else:
                st.info(f"Both players show similar levels of consistency.")
    else:
        # Initial message
        st.write("Select two players and click 'Compare Players' to see statistical comparison.")

def render_team_comparison(all_games: pd.DataFrame):
    """
    Render the team vs team comparison view with statistical analysis.
    
    Args:
        all_games: DataFrame containing all game stats
    """
    st.sidebar.header("Team Comparison")
    
    # Get team mapping
    name_map = (
        all_games
        .groupby("TEAM_ABBREVIATION")["TEAM_NAME"]
        .agg(lambda s: s.dropna().mode().iat[0] if not s.dropna().mode().empty else "")
        .to_dict()
    )
    teams = sorted(name_map.keys())
    
    # Team selection
    col1, col2 = st.sidebar.columns(2)
    with col1:
        team1_label = st.selectbox(
            "Team 1", 
            [f"{name_map[c]} ({c})" for c in teams], 
            key="team1"
        )
        team1_abbr = team1_label.split("(")[-1].strip(")")
    
    with col2:
        # Default team2 to the second team or first if only one exists
        default_idx = 1 if len(teams) > 1 else 0
        team2_label = st.selectbox(
            "Team 2", 
            [f"{name_map[c]} ({c})" for c in teams],
            index=default_idx, 
            key="team2"
        )
        team2_abbr = team2_label.split("(")[-1].strip(")")
    
    # Season & Phase selectors
    seasons = ["All"] + sorted(all_games["SeasonKey"].dropna().unique())
    season = st.sidebar.selectbox("Season", seasons)
    
    phases = ["All"] + sorted(all_games["Phase"].dropna().unique())
    phase = st.sidebar.selectbox("Phase", phases)
    
    # Set minimum games threshold
    min_games = st.sidebar.slider("Minimum Games", 1, 20, 5)
    
    # Run comparison when button is clicked
    if st.sidebar.button("Compare Teams"):
        st.header(f"Team Comparison: {team1_abbr} vs {team2_abbr}")
        
        # Verify that the required column TEAM_ABBREVIATION exists
        if "TEAM_ABBREVIATION" not in all_games.columns:
            st.error("Critical error: TEAM_ABBREVIATION column is missing from data. Cannot perform comparison.")
            return
        
        # Check if both teams exist in the dataset
        team1_exists = team1_abbr in all_games["TEAM_ABBREVIATION"].unique()
        team2_exists = team2_abbr in all_games["TEAM_ABBREVIATION"].unique()
        
        if not team1_exists or not team2_exists:
            missing_teams = []
            if not team1_exists:
                missing_teams.append(team1_abbr)
            if not team2_exists:
                missing_teams.append(team2_abbr)
            st.error(f"The following team(s) are not found in the dataset: {', '.join(missing_teams)}")
            return
        
        with st.spinner("Analyzing team statistics..."):
            try:
                comparison = compare_teams(
                    all_games=all_games,
                    team1_abbr=team1_abbr,
                    team2_abbr=team2_abbr,
                    season=season,
                    phase=phase,
                    min_games=min_games
                )
            except KeyError as e:
                error_key = str(e).strip("''")
                if error_key == "TEAM_ABBREVIATION":
                    st.error("Error: 'TEAM_ABBREVIATION' column not found in one of the team data subsets. This may be due to data inconsistency when filtering.")
                    st.info("Debugging tip: Check if the data for these teams contains the 'TEAM_ABBREVIATION' key in all entries.")
                else:
                    st.error(f"Error in data processing: KeyError - {e}. This might be due to missing data for one of the teams.")
                return
            except Exception as e:
                st.error(f"Unexpected error: {str(e)}")
                return
        
        # Check for errors in comparison
        if "error" in comparison:
            st.error(comparison["error"])
            return
        
        # --- Team Records Section ---
        st.subheader("Team Records")
        
        # Add context about the filtered records
        season_phase_context = []
        if season != "All":
            season_phase_context.append(f"season {season}")
        if phase != "All":
            season_phase_context.append(f"{phase.lower()}")
        
        context_str = " and ".join(season_phase_context)
        if context_str:
            st.write(f"*Team records shown are for {context_str} only.*")
        
        col1, col2 = st.columns(2)
        with col1:
            st.write(f"### {comparison['team1']['name']} ({comparison['team1']['abbr']})")
            st.write(f"Team Record: {comparison['team1']['record']} ({comparison['team1']['win_pct']}%)")
            st.write(f"Games Played: {comparison['team1']['games']}")
        
        with col2:
            st.write(f"### {comparison['team2']['name']} ({comparison['team2']['abbr']})")
            st.write(f"Team Record: {comparison['team2']['record']} ({comparison['team2']['win_pct']}%)")
            st.write(f"Games Played: {comparison['team2']['games']}")
        
        # --- Basic Stats Comparison ---
        st.subheader("Basic Stats Comparison (Per Game)")
        
        # Create main stats dataframe
        basic_stats = ['PTS', 'REB', 'AST', 'STL', 'BLK']
        if 'TOV' in comparison['stats']:
            basic_stats.append('TOV')
        
        stats_data = []
        for stat in basic_stats:
            if stat in comparison['stats']:
                stat_info = comparison['stats'][stat]
                stats_data.append({
                    'Statistic': stat,
                    f"{team1_abbr}": f"{stat_info['team1_avg']} ± {stat_info['team1_std']}",
                    f"{team2_abbr}": f"{stat_info['team2_avg']} ± {stat_info['team2_std']}",
                    'Difference': stat_info['difference'],
                    'Significant': "✓" if stat in comparison['statistical_tests'] and 
                                  comparison['statistical_tests'][stat]['significant'] else "✗"
                })
        
        stats_df = pd.DataFrame(stats_data)
        st.dataframe(stats_df, use_container_width=True)
        
        # --- Visualization of Key Stats ---
        st.subheader("Key Stats Comparison")
        
        # Prepare data for horizontal bar chart
        key_stats = ['PTS', 'REB', 'AST']
        key_stat_data = {stat: [] for stat in key_stats if stat in comparison['stats']}
        
        for stat in key_stat_data:
            key_stat_data[stat] = [
                comparison['stats'][stat]['team1_avg'],
                comparison['stats'][stat]['team2_avg']
            ]
        
        # Create streamlit columns for charts
        chart_cols = st.columns(len(key_stat_data))
        
        # Plot each stat side by side
        for i, (stat, values) in enumerate(key_stat_data.items()):
            with chart_cols[i]:
                # Create a simple bar chart
                chart_data = pd.DataFrame({
                    'Team': [team1_abbr, team2_abbr],
                    stat: values
                })
                st.write(f"{stat} Per Game")
                st.bar_chart(chart_data, x='Team', y=stat, use_container_width=True)
        
        # --- Shooting Percentages ---
        if comparison['derived_stats']:
            st.subheader("Shooting Percentages")
            
            pct_data = []
            for pct_type, pct_info in comparison['derived_stats'].items():
                label = {
                    'FG_PCT': 'Field Goal %', 
                    'FG3_PCT': '3-Point %', 
                    'FT_PCT': 'Free Throw %'
                }.get(pct_type, pct_type)
                
                pct_data.append({
                    'Type': label,
                    f"{team1_abbr} (Season)": pct_info['team1_total'],  # Store as numbers, not strings with '%'
                    f"{team2_abbr} (Season)": pct_info['team2_total'],
                    'Difference': pct_info['difference']
                })
            
            pct_df = pd.DataFrame(pct_data)
            # Reset index and use 1-based indexing
            pct_df = pct_df.reset_index(drop=True)
            pct_df.index = pct_df.index + 1
            
            # Format numbers in the display
            st.dataframe(pct_df.style.format({
                f"{team1_abbr} (Season)": "{:.1f}%",
                f"{team2_abbr} (Season)": "{:.1f}%",
                'Difference': "{:.1f}%"
            }), use_container_width=True)
        
        # --- Pace and Efficiency Metrics ---
        if comparison['pace_and_efficiency']:
            st.subheader("Pace and Efficiency Metrics")
            
            # Create metrics dataframe
            metrics_data = []
            
            # Offensive Efficiency
            off_eff = comparison['pace_and_efficiency']['offensive_efficiency']
            metrics_data.append({
                'Metric': 'Offensive Rating (pts/100 poss)',
                team1_abbr: off_eff['team1'],
                team2_abbr: off_eff['team2'],
                'Difference': off_eff['difference']
            })
            
            # Defensive Efficiency
            def_eff = comparison['pace_and_efficiency']['defensive_efficiency']
            metrics_data.append({
                'Metric': 'Defensive Rating (pts allowed/100 poss)',
                team1_abbr: def_eff['team1'],
                team2_abbr: def_eff['team2'],
                'Difference': def_eff['difference']
            })
            
            # Net Rating (Offensive - Defensive)
            net1 = off_eff['team1'] - def_eff['team1']
            net2 = off_eff['team2'] - def_eff['team2']
            metrics_data.append({
                'Metric': 'Net Rating',
                team1_abbr: round(net1, 2),
                team2_abbr: round(net2, 2),
                'Difference': round(net1 - net2, 2)
            })
            
            # Pace
            pace = comparison['pace_and_efficiency']['pace']
            metrics_data.append({
                'Metric': 'Pace (poss/48 min)',
                team1_abbr: pace['team1'],
                team2_abbr: pace['team2'],
                'Difference': pace['difference']
            })
            
            # Create and display dataframe
            pace_df = pd.DataFrame(metrics_data)
            # Reset index and use 1-based indexing
            pace_df = pace_df.reset_index(drop=True)
            pace_df.index = pace_df.index + 1
            st.dataframe(pace_df, use_container_width=True)
            
            # Add some contextual insights
            if net1 > net2:
                st.info(f"{team1_abbr} has a stronger net rating by {round(net1 - net2, 2)} points per 100 possessions.")
            elif net2 > net1:
                st.info(f"{team2_abbr} has a stronger net rating by {round(net2 - net1, 2)} points per 100 possessions.")
            
            if abs(pace['difference']) > 2:
                faster = team1_abbr if pace['team1'] > pace['team2'] else team2_abbr
                slower = team2_abbr if faster == team1_abbr else team1_abbr
                st.info(f"{faster} plays at a significantly faster pace than {slower}.")
        
        # --- Margin Analysis ---
        if 'margins' in comparison:
            st.subheader("Margin Analysis")
            
            margin_data = []
            
            # Average margin
            avg_margin = comparison['margins']['average_margin']
            margin_data.append({
                'Metric': 'Average Margin',
                team1_abbr: avg_margin['team1'],
                team2_abbr: avg_margin['team2']
            })
            
            # Close games percentage
            close_games = comparison['margins']['close_games_pct']
            margin_data.append({
                'Metric': 'Close Games (≤ 5 pts)',
                team1_abbr: close_games['team1'],  # Store numeric value
                team2_abbr: close_games['team2']   # Store numeric value
            })
            
            # Blowout percentage
            blowouts = comparison['margins']['blowout_pct']
            margin_data.append({
                'Metric': 'Blowouts (≥ 15 pts)',
                team1_abbr: blowouts['team1'],  # Store numeric value
                team2_abbr: blowouts['team2']   # Store numeric value
            })
            
            margin_df = pd.DataFrame(margin_data)
            # Reset index and use 1-based indexing
            margin_df = margin_df.reset_index(drop=True)
            margin_df.index = margin_df.index + 1
            
            # Format only percentage fields
            format_dict = {}
            for col in margin_df.columns:
                if col not in ['Metric']:
                    format_dict[col] = lambda x: f"{x:.1f}%" if isinstance(x, (int, float)) else x
            
            st.dataframe(margin_df.style.format(format_dict), use_container_width=True)
            
            # Add insights about margins
            if avg_margin['team1'] > 0 and avg_margin['team2'] > 0:
                st.info("Both teams have positive average margins, indicating they outscore opponents on average.")
            elif avg_margin['team1'] < 0 and avg_margin['team2'] < 0:
                st.info("Both teams have negative average margins, indicating they are outscored by opponents on average.")
            
            # Compare close game performance
            if abs(close_games['team1'] - close_games['team2']) > 10:
                more_close = team1_abbr if close_games['team1'] > close_games['team2'] else team2_abbr
                st.info(f"{more_close} plays in more close games, which could indicate they're more frequently in competitive situations.")
        
        # --- Head-to-Head Record ---
        if 'head_to_head' in comparison and comparison['head_to_head']:
            st.subheader("Head-to-Head Record")
            
            h2h = comparison['head_to_head']
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric(f"{team1_abbr} Wins", h2h['team1_wins'])
            
            with col2:
                st.metric(f"{team2_abbr} Wins", h2h['team2_wins'])
            
            # Display individual game results
            if h2h['results']:
                st.write("#### Game Results")
                
                # Format results for display
                results_data = []
                for game in h2h['results']:
                    results_data.append({
                        'Date': game['date'],
                        f"{team1_abbr} Score": game['team1_pts'],
                        f"{team2_abbr} Score": game['team2_pts'],
                        'Winner': game['winner'],
                        'Margin': game['margin']
                    })
                
                results_df = pd.DataFrame(results_data).sort_values('Date')
                # Reset index and use 1-based indexing
                results_df = results_df.reset_index(drop=True)
                results_df.index = results_df.index + 1
                st.dataframe(results_df, use_container_width=True)
        
        # --- Consistency Analysis ---
        if comparison['consistency']:
            st.subheader("Consistency Analysis")
            st.write("Coefficient of Variation (lower = more consistent)")
            
            consistency_data = []
            for stat, consist_info in comparison['consistency'].items():
                stat_label = {'PTS': 'Points', 'REB': 'Rebounds', 'AST': 'Assists'}.get(stat, stat)
                
                consistency_data.append({
                    'Statistic': stat_label,
                    f"{team1_abbr} CV": consist_info['team1_cv'],  # Store numeric value
                    f"{team2_abbr} CV": consist_info['team2_cv'],  # Store numeric value
                    'More Consistent Team': consist_info['more_consistent']
                })
            
            consist_df = pd.DataFrame(consistency_data)
            # Reset index and use 1-based indexing
            consist_df = consist_df.reset_index(drop=True)
            consist_df.index = consist_df.index + 1
            
            # Format percentage columns
            format_dict = {
                f"{team1_abbr} CV": "{:.1f}%",
                f"{team2_abbr} CV": "{:.1f}%"
            }
            
            st.dataframe(consist_df.style.format(format_dict), use_container_width=True)
            
            # Highlight the team that is more consistent overall
            t1_wins = sum(1 for info in comparison['consistency'].values() 
                          if info['more_consistent'] == team1_abbr)
            t2_wins = sum(1 for info in comparison['consistency'].values() 
                          if info['more_consistent'] == team2_abbr)
            
            if t1_wins > t2_wins:
                st.info(f"Overall, {team1_abbr} shows more consistency across key statistics.")
            elif t2_wins > t1_wins:
                st.info(f"Overall, {team2_abbr} shows more consistency across key statistics.")
            else:
                st.info(f"Both teams show similar levels of consistency.")
    else:
        # Initial message
        st.write("Select two teams and click 'Compare Teams' to see statistical comparison.")

def render_player_performance_analysis(all_games: pd.DataFrame):
    """
    Renders the player performance analysis view to compare a player's overall stats
    versus their performance in specific contexts using a structured UI flow.
    
    Args:
        all_games: DataFrame containing all game stats
    """
    st.sidebar.header("Player Performance Analysis")

    # --- 1. Select Player to Analyze ---
    all_available_players_global = sorted(all_games["PLAYER_NAME"].dropna().unique())
    player_to_analyze = st.sidebar.selectbox(
        "Select Player to Analyze", 
        all_available_players_global,
        key="main_player_selectbox"
    )
    if not player_to_analyze:
        st.warning("Please select a player to analyze.")
        return

    # --- Global Season/Phase Options (used for both period selectors) ---
    unique_seasons_global = ["All Seasons"] + sorted(all_games["SeasonKey"].dropna().unique())
    unique_phases_global = ["All Phases"] + sorted(all_games["Phase"].dropna().unique())
    
    # --- Team Name Map (globally defined) ---
    teams_map_for_display_global = (
        all_games
        .groupby("TEAM_ABBREVIATION")["TEAM_NAME"]
        .agg(lambda s: s.dropna().mode().iat[0] if not s.dropna().mode().empty else "")
        .to_dict()
    )

    # --- 2. Define Period 1 (Baseline) ---
    st.sidebar.subheader("1. Define Period 1 (Baseline)")
    overall_season_selected = st.sidebar.selectbox(
        "Period 1: Season", 
        unique_seasons_global, 
        key="overall_season_selectbox"
    )
    overall_phase_selected = st.sidebar.selectbox(
        "Period 1: Phase", 
        unique_phases_global, 
        key="overall_phase_selectbox"
    )

    # --- 3. Define Period 2 (Comparison Context) ---
    st.sidebar.subheader("2. Define Period 2 (Comparison Context)")
    comparison_season_selected = st.sidebar.selectbox(
        "Period 2: Season", 
        unique_seasons_global, 
        key="comparison_season_selectbox"
    )
    comparison_phase_selected = st.sidebar.selectbox(
        "Period 2: Phase", 
        unique_phases_global, 
        key="comparison_phase_selectbox"
    )

    # --- Scope for Period 2 refinements ---
    games_in_comparison_scope = all_games.copy()
    if comparison_season_selected != "All Seasons":
        games_in_comparison_scope = games_in_comparison_scope[games_in_comparison_scope["SeasonKey"] == comparison_season_selected]
    if comparison_phase_selected != "All Phases":
        games_in_comparison_scope = games_in_comparison_scope[games_in_comparison_scope["Phase"] == comparison_phase_selected]

    if games_in_comparison_scope.empty and (comparison_season_selected != "All Seasons" or comparison_phase_selected != "All Phases"):
        st.sidebar.warning("No games found for the selected Period 2 season/phase to refine further.")
        # Allow proceeding if user wants to compare Period 1 to a broad Period 2 (e.g. all of 2024-25 Playoffs)

    main_player_games_in_comp_scope = games_in_comparison_scope[
        games_in_comparison_scope["PLAYER_NAME"] == player_to_analyze
    ]
    
    actual_series_opponent_team_abbrs_for_comp = set()
    actual_series_opponent_players_for_comp = set()
    main_player_team_abbr_for_comp_series_parsing = None

    if not main_player_games_in_comp_scope.empty:
        if "TEAM_ABBREVIATION" in main_player_games_in_comp_scope.columns and not main_player_games_in_comp_scope["TEAM_ABBREVIATION"].mode().empty:
            main_player_team_abbr_for_comp_series_parsing = main_player_games_in_comp_scope["TEAM_ABBREVIATION"].mode().iloc[0]
        
        if main_player_team_abbr_for_comp_series_parsing and "SeriesKey" in main_player_games_in_comp_scope.columns:
            player_series_keys_in_comp = main_player_games_in_comp_scope["SeriesKey"].dropna().unique()
            for series_key in player_series_keys_in_comp:
                teams_in_series = series_key.split(' vs ')
                opponent_in_this_series = None
                if len(teams_in_series) == 2:
                    if teams_in_series[0] == main_player_team_abbr_for_comp_series_parsing:
                        opponent_in_this_series = teams_in_series[1]
                    elif teams_in_series[1] == main_player_team_abbr_for_comp_series_parsing:
                        opponent_in_this_series = teams_in_series[0]
                
                if opponent_in_this_series:
                    actual_series_opponent_team_abbrs_for_comp.add(opponent_in_this_series)
                    series_games_for_opponent = games_in_comparison_scope[
                        (games_in_comparison_scope["SeriesKey"] == series_key) &
                        (games_in_comparison_scope["TEAM_ABBREVIATION"] == opponent_in_this_series)
                    ]
                    actual_series_opponent_players_for_comp.update(series_games_for_opponent["PLAYER_NAME"].dropna().unique())

    if player_to_analyze in actual_series_opponent_players_for_comp:
        actual_series_opponent_players_for_comp.remove(player_to_analyze)

    # --- 4. Refine Period 2 Further (Optional) ---
    st.sidebar.subheader("3. Refine Period 2 Further (Optional)")
    comparison_opponent_name_selected = None
    comparison_opponent_team_abbr_selected = None
    comparison_series_id_selected = None

    context_type = st.sidebar.radio(
        "Refinement Type for Period 2",
        ["None (Use entire Period 2 as defined above)", "Against Opponent Player", "Against Opponent Team", "In Specific Series"],
        key="context_type_radio"
    )

    if context_type == "Against Opponent Player":
        sorted_series_opponent_players = sorted(list(actual_series_opponent_players_for_comp))
        if sorted_series_opponent_players:
            comparison_opponent_name_selected = st.sidebar.selectbox(
                "Select Opponent Player (for Period 2)",
                ["None"] + sorted_series_opponent_players,
                key="comparison_opponent_player_selectbox",
                help=f"Players whom {player_to_analyze} faced in a series during Period 2 ({comparison_season_selected}, {comparison_phase_selected})."
            )
            comparison_opponent_name_selected = None if comparison_opponent_name_selected == "None" else comparison_opponent_name_selected
        else:
            st.sidebar.text("No direct series opponent players found in Period 2 scope.")

    elif context_type == "Against Opponent Team":
        sorted_series_opponent_teams = sorted(list(actual_series_opponent_team_abbrs_for_comp))
        if sorted_series_opponent_teams:
            selected_opponent_team_display = st.sidebar.selectbox(
                "Select Opponent Team (for Period 2)",
                ["None"] + [f"{teams_map_for_display_global.get(c, c)} ({c})" for c in sorted_series_opponent_teams],
                key="comparison_opponent_team_selectbox",
                help=f"Teams {player_to_analyze} faced in a series during Period 2 ({comparison_season_selected}, {comparison_phase_selected})."
            )
            if selected_opponent_team_display != "None":
                comparison_opponent_team_abbr_selected = selected_opponent_team_display.split("(")[-1].strip(")")
        else:
            st.sidebar.text("No direct series opponent teams found in Period 2 scope.")
    
    # Series Selector Logic for Period 2 refinement
    show_series_selector_for_comp = False
    series_help_text_for_comp = f"Series {player_to_analyze} participated in during Period 2."
    possible_series_keys_for_selector_for_comp = []

    if main_player_team_abbr_for_comp_series_parsing and "SeriesKey" in main_player_games_in_comp_scope.columns:
        all_player_series_in_comp_scope = sorted(main_player_games_in_comp_scope["SeriesKey"].dropna().unique())

        if context_type == "In Specific Series":
            show_series_selector_for_comp = True
            possible_series_keys_for_selector_for_comp = all_player_series_in_comp_scope
        
        elif comparison_opponent_name_selected: # An opponent player is selected for Period 2
            opponent_player_data_for_series = games_in_comparison_scope[games_in_comparison_scope["PLAYER_NAME"] == comparison_opponent_name_selected]
            if not opponent_player_data_for_series.empty:
                actual_opponent_player_team_abbr = opponent_player_data_for_series["TEAM_ABBREVIATION"].mode().iloc[0]
                for s_key in all_player_series_in_comp_scope:
                    if main_player_team_abbr_for_comp_series_parsing in s_key and actual_opponent_player_team_abbr in s_key:
                        possible_series_keys_for_selector_for_comp.append(s_key)
                if possible_series_keys_for_selector_for_comp:
                    show_series_selector_for_comp = True
                    series_help_text_for_comp = f"Series between {player_to_analyze} ({main_player_team_abbr_for_comp_series_parsing}) and {comparison_opponent_name_selected} ({actual_opponent_player_team_abbr}) in Period 2."

        elif comparison_opponent_team_abbr_selected: # An opponent team is selected for Period 2
            for s_key in all_player_series_in_comp_scope:
                if main_player_team_abbr_for_comp_series_parsing in s_key and comparison_opponent_team_abbr_selected in s_key:
                    possible_series_keys_for_selector_for_comp.append(s_key)
            if possible_series_keys_for_selector_for_comp:
                show_series_selector_for_comp = True
                series_help_text_for_comp = f"Series between {player_to_analyze} ({main_player_team_abbr_for_comp_series_parsing}) and {comparison_opponent_team_abbr_selected} in Period 2."
    
    if show_series_selector_for_comp and possible_series_keys_for_selector_for_comp:
        comparison_series_id_selected = st.sidebar.selectbox(
            "Refine Period 2 with Specific Series",
            ["None"] + sorted(list(set(possible_series_keys_for_selector_for_comp))), 
            key="comparison_series_selectbox",
            help=series_help_text_for_comp
        )
        comparison_series_id_selected = None if comparison_series_id_selected == "None" else comparison_series_id_selected
    elif show_series_selector_for_comp:
        st.sidebar.text("No relevant series found for this player/opponent combination in Period 2.")

    min_games_threshold = st.sidebar.slider("Minimum Games per Period for Analysis", 1, 20, 3, key="min_games_slider")

    analysis_params = {
        "all_games": all_games,
        "player_name": player_to_analyze,
        "overall_season": None if overall_season_selected == "All Seasons" else overall_season_selected,
        "overall_phase": None if overall_phase_selected == "All Phases" else overall_phase_selected,
        "comparison_season": None if comparison_season_selected == "All Seasons" else comparison_season_selected,
        "comparison_phase": None if comparison_phase_selected == "All Phases" else comparison_phase_selected,
        "comparison_opponent_name": comparison_opponent_name_selected,
        "comparison_opponent_team": comparison_opponent_team_abbr_selected,
        "comparison_series_id": comparison_series_id_selected,
        "min_games": min_games_threshold
    }
    
    if st.sidebar.button("Analyze Player Performance", key="analyze_button"):        
        # Validation for Period 2 refinement selections
        if context_type == "Against Opponent Player" and not comparison_opponent_name_selected:
            st.warning("Please select an opponent player for Period 2 refinement, or choose a different refinement type.")
            return 
        if context_type == "Against Opponent Team" and not comparison_opponent_team_abbr_selected:
            st.warning("Please select an opponent team for Period 2 refinement, or choose a different refinement type.")
            return
        if context_type == "In Specific Series" and not comparison_series_id_selected:
             st.warning("Please select a specific series for Period 2 refinement, or choose a different refinement type.")
             return
        # If context_type is "None...", no further refinement is selected, which is valid.

        with st.spinner("Analyzing player statistics..."):
            try:
                analysis_results = player_performance_analysis(**analysis_params)
            except Exception as e:
                st.error(f"Error during analysis: {str(e)}")
                # import traceback
                # st.error(traceback.format_exc())
                return

        if "error" in analysis_results:
            st.error(analysis_results["error"])
            if "overall_period_description" in analysis_results:
                 st.info(f"Period 1 (Baseline) attempted: {analysis_results['overall_period_description']}")
            if "comparison_context_description" in analysis_results:
                 st.info(f"Period 2 (Comparison) attempted: {analysis_results['comparison_context_description']}")
            return

        st.header(f"Performance Analysis: {player_to_analyze}")
        st.subheader("Analysis Configuration")
        
        overall_desc_display = analysis_results.get('overall_period_description', 'N/A')
        st.write(f"**Period 1 (Baseline):** {overall_desc_display}")
        
        comp_context_desc_display = analysis_results.get('comparison_context_description', 'N/A')
        st.write(f"**Period 2 (Comparison Context):** {comp_context_desc_display}")
        
        overall_games_count = analysis_results.get('overall_games', 0)
        context_games_count = analysis_results.get('context_games', 0)
        st.write(f"**Games in Period 1 for {player_to_analyze}:** {overall_games_count}")
        st.write(f"**Games in Period 2 for {player_to_analyze}:** {context_games_count}")
        
        st.info(
            f"This analysis compares {player_to_analyze}'s performance in 'Period 2 ({comp_context_desc_display})' "
            f"against their performance in 'Period 1 ({overall_desc_display})'."
        )

        st.subheader("Statistics Comparison (Avg per Game)")
        stats_data = []
        if "overall_stats" in analysis_results and "context_stats" in analysis_results and "performance_diff" in analysis_results:
            # Ensure all stats in overall_stats are attempted for display, even if missing in context/diff
            for stat, overall_vals in analysis_results['overall_stats'].items():
                context_vals = analysis_results['context_stats'].get(stat, {})
                diff_vals = analysis_results['performance_diff'].get(stat, {})
                stat_test_info = analysis_results.get('statistical_tests', {}).get(stat, {})
                
                stats_data.append({
                    'Statistic': stat,
                    'Period 1 Avg': overall_vals.get('avg', 'N/A'),
                    'Period 2 Avg': context_vals.get('avg', 'N/A'),
                    'Difference (P2-P1)': diff_vals.get('diff', 'N/A'),
                    'Change (%)': f"{diff_vals.get('pct_diff', 'N/A')}%" if diff_vals.get('pct_diff', 'N/A') not in ['N/A', None] else 'N/A',
                    'Period 2 Better': '✓' if diff_vals.get('is_better', False) else '✗' if diff_vals.get('is_better') is False else '-',
                    'Significant (p < 0.05)': '✓' if stat_test_info.get('significant', False) else '✗' if stat_test_info.get('significant') is False else stat_test_info.get('error','-')
                })
        if stats_data:
            stats_df = pd.DataFrame(stats_data).reset_index(drop=True)
            stats_df.index = stats_df.index + 1
            st.dataframe(stats_df, use_container_width=True)
        else:
            st.write("No statistical data to display for this comparison.")

        if 'shooting_pct' in analysis_results and isinstance(analysis_results['shooting_pct'], dict):
            st.subheader("Shooting Percentages")
            pct_data_display = []
            for pct_type, pct_info_vals in analysis_results['shooting_pct'].items():
                label_map = {'FG_PCT': 'Field Goal %', 'FG3_PCT': '3-Point %', 'FT_PCT': 'Free Throw %'}
                pct_data_display.append({
                    'Type': label_map.get(pct_type, pct_type),
                    'Period 1': pct_info_vals.get('overall', 'N/A'), # 'overall' here means Period 1
                    'Period 2': pct_info_vals.get('context', 'N/A'), # 'context' here means Period 2
                    'Difference (P2-P1)': pct_info_vals.get('diff', 'N/A')
                })
            if pct_data_display:
                pct_df_display = pd.DataFrame(pct_data_display).reset_index(drop=True)
                pct_df_display.index = pct_df_display.index + 1
                st.dataframe(pct_df_display.style.format({
                    'Period 1': "{:.1f}%", 
                    'Period 2': "{:.1f}%", 
                    'Difference (P2-P1)': "{:.1f}%"})
                , use_container_width=True)

        st.subheader("Key Stats Visualization (Average per Game)")
        key_stats_for_viz = ['PTS', 'REB', 'AST']
        available_stats_for_viz = [s for s in key_stats_for_viz 
                                   if s in analysis_results.get('overall_stats', {}) 
                                   and s in analysis_results.get('context_stats', {})]
        if available_stats_for_viz:
            chart_df_data = pd.DataFrame([{
                'Statistic': s,
                'Period 1': analysis_results['overall_stats'][s]['avg'],
                'Period 2': analysis_results['context_stats'][s]['avg']
            } for s in available_stats_for_viz])
            fig_viz, ax_viz = plt.subplots(figsize=(10, 6))
            bar_width_viz = 0.35
            x_indices_viz = np.arange(len(available_stats_for_viz))
            period1_bars = ax_viz.bar(x_indices_viz - bar_width_viz/2, chart_df_data['Period 1'], bar_width_viz, label='Period 1')
            period2_bars_viz = ax_viz.bar(x_indices_viz + bar_width_viz/2, chart_df_data['Period 2'], bar_width_viz, label='Period 2')
            ax_viz.set_xlabel('Statistic'); ax_viz.set_ylabel('Average Value'); ax_viz.set_title('Period 1 vs. Period 2 Performance')
            ax_viz.set_xticks(x_indices_viz); ax_viz.set_xticklabels(available_stats_for_viz); ax_viz.legend()
            def add_bar_labels(bars_to_label):
                for bar_item_label in bars_to_label:
                    height_label = bar_item_label.get_height()
                    ax_viz.annotate(f'{height_label:.1f}', xy=(bar_item_label.get_x() + bar_item_label.get_width() / 2, height_label), xytext=(0, 3), textcoords="offset points", ha='center', va='bottom')
            add_bar_labels(period1_bars); add_bar_labels(period2_bars_viz)
            st.pyplot(fig_viz)

            st.subheader("Performance Change in Period 2 vs. Period 1")
            pct_diff_chart_data = pd.DataFrame([{
                'Statistic': s,
                'Change (%)': analysis_results['performance_diff'][s]['pct_diff'],
                'Direction': 'Better' if analysis_results['performance_diff'][s]['is_better'] else 'Worse'
            } for s in available_stats_for_viz if s in analysis_results.get('performance_diff', {})])
            if not pct_diff_chart_data.empty:
                fig_pct_diff, ax_pct_diff = plt.subplots(figsize=(10, len(available_stats_for_viz) * 0.8))
                chart_colors = ['green' if row['Direction'] == 'Better' else 'red' for _, row in pct_diff_chart_data.iterrows()]
                h_bars = ax_pct_diff.barh(pct_diff_chart_data['Statistic'], pct_diff_chart_data['Change (%)'], color=chart_colors)
                for bar_h in h_bars:
                    width_h = bar_h.get_width()
                    x_pos = width_h + (5 if width_h >= 0 else -5)
                    ha_val = 'left' if width_h >= 0 else 'right'
                    ax_pct_diff.annotate(f'{width_h:.1f}%', xy=(width_h, bar_h.get_y() + bar_h.get_height() / 2), xytext=(5 if width_h >=0 else -5, 0), textcoords="offset points", ha=ha_val, va='center')
                ax_pct_diff.axvline(x=0, color='grey', linestyle='--', linewidth=0.8)
                ax_pct_diff.set_xlabel('Percentage Change (%) vs Period 1'); ax_pct_diff.set_title('Player Performance Change in Period 2'); plt.tight_layout()
                st.pyplot(fig_pct_diff)
            
            st.subheader("Summary & Insights")
            if "performance_diff" in analysis_results:
                num_metrics = len(analysis_results['performance_diff'])
                better_metrics_count = sum(1 for stat_info_insight in analysis_results['performance_diff'].values() if stat_info_insight['is_better'])
                worse_metrics_count = num_metrics - better_metrics_count
                improvements_list = [(s_insight, info_insight['pct_diff']) for s_insight, info_insight in analysis_results['performance_diff'].items() if info_insight['is_better'] and info_insight['pct_diff'] is not None]
                declines_list = [(s_insight, info_insight['pct_diff']) for s_insight, info_insight in analysis_results['performance_diff'].items() if not info_insight['is_better'] and info_insight['pct_diff'] is not None]
                if improvements_list:
                    biggest_improvement_stat, biggest_improvement_val = max(improvements_list, key=lambda item: item[1])
                    st.markdown(f"<p style='color:green;'>Biggest improvement in Period 2: **{biggest_improvement_stat} (+{biggest_improvement_val:.1f}%)**</p>", unsafe_allow_html=True)
                if declines_list:
                    biggest_decline_stat, biggest_decline_val = min(declines_list, key=lambda item: item[1]) # min for negative numbers is "most negative"
                    st.markdown(f"<p style='color:red;'>Biggest decline in Period 2: **{biggest_decline_stat} ({biggest_decline_val:.1f}%)**</p>", unsafe_allow_html=True)
                
                if better_metrics_count > worse_metrics_count:
                    st.success(f"Overall: {player_to_analyze} tended to perform better in Period 2 ({comp_context_desc_display}) across {better_metrics_count}/{num_metrics} key metrics compared to Period 1 ({overall_desc_display}).")
                elif worse_metrics_count > better_metrics_count:
                    st.warning(f"Overall: {player_to_analyze} tended to perform worse in Period 2 ({comp_context_desc_display}) across {worse_metrics_count}/{num_metrics} key metrics compared to Period 1 ({overall_desc_display}).")
                else:
                    st.info(f"Overall: {player_to_analyze}'s performance showed a mixed pattern in Period 2 ({comp_context_desc_display}) with {better_metrics_count} improvements and {worse_metrics_count} declines out of {num_metrics} metrics, compared to Period 1 ({overall_desc_display})." )

            significant_diff_stats = [s_sig for s_sig, test_res in analysis_results.get('statistical_tests', {}).items() if test_res.get('significant', False)]
            if significant_diff_stats:
                st.write("**Statistically Significant Differences (p < 0.05) between Period 1 and Period 2 were observed for:**")
                md_list = []
                for stat_s in significant_diff_stats:
                    p_val = analysis_results['statistical_tests'][stat_s]['p_value']
                    t_stat_val = analysis_results['statistical_tests'][stat_s]['t_statistic']
                    diff_s_val = analysis_results['performance_diff'][stat_s]['diff']
                    direction_s = "higher" if diff_s_val > 0 else "lower"
                    color_s = "green" if diff_s_val > 0 else "red"
                    if stat_s == 'TOV': # Lower TOV is better
                         direction_s = "lower" if diff_s_val < 0 else "higher"
                         color_s = "green" if diff_s_val < 0 else "red"
                    md_list.append(f"  - <span style='color:{color_s};'>**{stat_s}**: {abs(diff_s_val):.2f} points {direction_s} in Period 2</span> (p={p_val:.3f}, t={t_stat_val:.2f})")
                st.markdown("\n".join(md_list), unsafe_allow_html=True)
            else:
                st.write("No statistically significant differences found for individual metrics between Period 1 and Period 2 (p ≥ 0.05)." )
        else:
            st.warning("No key statistics available for visualization for this comparison.")
    else:
        st.info("Select a player, define Period 1 (Baseline) and Period 2 (Comparison), optionally refine Period 2, and then click 'Analyze Player Performance'.")
