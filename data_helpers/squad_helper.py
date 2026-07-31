from settings import IMPECT_DIR
import json
import os
import pandas as pd
import streamlit as st

@st.cache_data
def load_squads():
    squad_path = os.path.join(IMPECT_DIR, 'squads', f'squads_743.json')
    with open(squad_path) as f:
        data = json.load(f)
    df_squads = pd.json_normalize(data)
    return df_squads

@st.cache_data
def load_lineups(match_id: int):
    lineup_path = os.path.join(IMPECT_DIR, 'lineups', f'lineups_{match_id}.json')
    with open(lineup_path) as f:
        data = json.load(f)
    df_lineup = pd.json_normalize(data)
    return df_lineup

def merge_squadnames_matches(matches: pd.DataFrame, squads: pd.DataFrame = None) -> pd.DataFrame:
    if squads is None:
        squads = load_squads()
    squad_name_map = squads.rename(columns={"id": "squadId"}).set_index("squadId")["name"]
    matches["homeSquadName"] = matches["homeSquadId"].map(squad_name_map)
    matches["awaySquadName"] = matches["awaySquadId"].map(squad_name_map)
    matches["label"] = f"{matches["homeSquadName"]} - {matches["awaySquadName"]}"
    return matches
