from settings import IMPECT_DIR
import json
import os
import pandas as pd
import streamlit as st
from data_helpers.kpi_helper import kpi_long_df, load_player_kpi, get_player_kpi, kpi_percentiles, kpi_statistics

POSITION_MAP = {
    "GOALKEEPER": "GK",
    "CENTRAL_DEFENDER": "DEF", "LEFT_WINGBACK_DEFENDER": "DEF", "RIGHT_WINGBACK_DEFENDER": "DEF",
    "DEFENSE_MIDFIELD": "MID", "CENTRAL_MIDFIELD": "MID", "ATTACKING_MIDFIELD": "MID",
    "LEFT_WINGER": "FWD", "RIGHT_WINGER": "FWD", "CENTER_FORWARD": "FWD",
}


@st.cache_data
def load_players(squad_id: int = None):
    player_path = os.path.join(IMPECT_DIR, 'players', f'players_743.json')
    with open(player_path) as f:
        data = json.load(f)
    df_players = pd.json_normalize(data)
    if squad_id is not None:
        df_players = df_players[df_players["currentSquadId"] == squad_id]
    return df_players

def merge_playernames(playerStats: pd.DataFrame, players: pd.DataFrame = None) -> pd.DataFrame:
    """
    Add names of players to the dataframe
    """
    if players is None:
        players = load_players()
    squad_name_map = players.rename(columns={"id": "playerId"}).set_index("playerId")["commonname"]
    playerStats["playerName"] = playerStats["playerId"].map(squad_name_map)
    return playerStats

def merge_playerpositions(playerStats: pd.DataFrame, df: pd.DataFrame = None) -> pd.DataFrame:
    """
    Add positions of the players to the dataframe
    For each match the player has a position recorded
    Use mode to determine at which position player has played most often
    """
    if df is None:
        df = kpi_long_df()
    # we have again for each match/player/kpi one entry. each entry has position assigned
    # we have to find middleground
    match_position = df.drop_duplicates(subset=["id", "matchId"])  # we have only one position per game
    positions = match_position.groupby("id")["position"].agg(lambda pos: pos.mode().iloc[0])  # mode == value with highest count
    playerStats = playerStats.join(positions.rename({"id": "playerId"}), on="playerId", how="right").rename(columns={"position": "playerPosition"})
    playerStats["playerLine"] = playerStats["playerPosition"].map(POSITION_MAP)
    return playerStats



def player_pipeline(matches):
    # we could also first load the player table
    # but as we load all matches we also find all players who have registered an KPI
    # and we don't care about the others
    df_kpi_all = load_player_kpi(matches.id)
    long_df = kpi_long_df(matches, df_kpi_all)
    kpi_range = kpi_statistics(long_df)
    all_match_ids = df_kpi_all["matchId"].tolist()

    players_df = get_player_kpi(matches, long_df, df_kpi_all)
    players_df = merge_playernames(players_df)
    players_df = merge_playerpositions(players_df, long_df)
    players_df = kpi_percentiles(players_df)

    kpi_columns_all = [c for c in players_df.columns if c.startswith("kpi_")]
    n_matches = len(all_match_ids)
    # player_kpi sometimes returns just a None value instead of list with None
    # in this case just replace with 0
    for col in kpi_columns_all:
        players_df[col] = players_df[col].apply(
            lambda lst: [-0] * n_matches if not isinstance(lst, list)
            else [float(x) if pd.notnull(x) else -0 for x in lst]
        )
    players_df = players_df.copy()  # just here to prevent the fragmented df warning
    return players_df, long_df, kpi_range

def compute_scores(playerStats, selected_cols):
    X = playerStats[selected_cols].apply(pd.to_numeric, errors="coerce")
    score = X.mean(axis=1, skipna=True).round(1)
    return playerStats.assign(score=score)
