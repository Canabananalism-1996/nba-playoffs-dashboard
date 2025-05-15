# app/app.py

import streamlit as st
from components.data_loader import load_all_games, build_game_map
from components.ui_views    import (
    render_game_explorer,
    render_head_to_head,
    render_player_explorer,
    render_team_explorer,
    render_player_comparison,
    render_team_comparison,
    render_player_performance_analysis
)

st.set_page_config(page_title="NBA Analytics", layout="wide")
st.title("NBA Analytics")

all_games = load_all_games()
if all_games.empty:
    st.error("No data loaded.")
    st.stop()

game_map = build_game_map(all_games)

view = st.sidebar.radio("Select View", [
    "Game Explorer",
    "Head-to-Head",
    "Player Explorer",
    "Team Explorer",
    "Player Comparison",
    "Team Comparison",
    "Player Performance Analysis"
])

if view == "Game Explorer":
    render_game_explorer(all_games)
elif view == "Head-to-Head":
    render_head_to_head(all_games)
elif view == "Player Explorer":
    render_player_explorer(all_games)
elif view == "Team Explorer":
    render_team_explorer(all_games)
elif view == "Player Comparison":
    render_player_comparison(all_games)
elif view == "Team Comparison":
    render_team_comparison(all_games)
elif view == "Player Performance Analysis":
    render_player_performance_analysis(all_games)
