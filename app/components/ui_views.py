import streamlit as st
import pandas as pd
import difflib
from typing import Dict

from components.analysis import (
    points_over_games,
    plus_minus_table,
    head_to_head_record,
    get_series_box_scores,
    get_team_season_trend,
    team_game_log,
    pivot_stat_over_series,
    pts_breakdown
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
    st.dataframe(log1[out1_cols].sort_values("GAME_DATE"), use_container_width=True)

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
    
    st.dataframe(ser.reset_index(drop=True), use_container_width=True)

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
