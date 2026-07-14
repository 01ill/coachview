from settings import IMPECT_DIR
import json
import os
import pandas as pd
from data_helpers.kpi_helper import kpi_long_df

def load_players():
    player_path = os.path.join(IMPECT_DIR, 'players', f'players_743.json')
    with open(player_path) as f:
        data = json.load(f)
    df_players = pd.json_normalize(data)
    return df_players

def merge_playernames(playerStats: pd.DataFrame, players: pd.DataFrame = None) -> pd.DataFrame:
    if players is None:
        players = load_players()
    squad_name_map = players.rename(columns={"id": "playerId"}).set_index("playerId")["commonname"]
    playerStats["playerName"] = playerStats["playerId"].map(squad_name_map)
    return playerStats

def merge_playerpositions(playerStats: pd.DataFrame, df: pd.DataFrame = None) -> pd.DataFrame:
    if df is None:
        df = kpi_long_df()
    # we have again for each match/player/kpi one entry. each entry has position assigned
    # we have to find middleground
    match_position = df.drop_duplicates(subset=["id", "matchId"])  # we have only one position per game
    positions = match_position.groupby("id")["position"].agg(lambda pos: pos.mode().iloc[0])  # mode == value with highest count
    playerStats = playerStats.join(positions.rename({"id": "playerId"}), on="playerId", how="right")
    return playerStats
