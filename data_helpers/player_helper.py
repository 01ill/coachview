from settings import IMPECT_DIR
import json
import os
import pandas as pd

def load_players():
    player_path = os.path.join(IMPECT_DIR, 'players', f'players_743.json')
    with open(player_path) as f:
        data = json.load(f)
        #save it in dataframe
    df_players = pd.json_normalize(data)
    return df_players

def merge_playernames(playerStats: pd.DataFrame, players: pd.DataFrame = None) -> pd.DataFrame:
    if players is None:
        players = load_players()
    squad_name_map = players.rename(columns={"id": "playerId"}).set_index("playerId")["commonname"]
    playerStats["playerName"] = playerStats["playerId"].map(squad_name_map)
    return playerStats