from settings import IMPECT_DIR
import json
import os
import pandas as pd

def load_squads():
    squad_path = os.path.join(IMPECT_DIR, 'squads', f'squads_743.json')
    with open(squad_path) as f:
        data = json.load(f)
        #save it in dataframe
    df_squads = pd.json_normalize(data)
    return df_squads

def merge_squadnames_matches(matches: pd.DataFrame, squads: pd.DataFrame = None) -> pd.DataFrame:
    if squads is None:
        squads = load_squads()
    squad_name_map = squads.rename(columns={"id": "squadId"}).set_index("squadId")["name"]
    matches["homeSquadName"] = matches["homeSquadId"].map(squad_name_map)
    matches["awaySquadName"] = matches["awaySquadId"].map(squad_name_map)
    matches["label"] = f"{matches["homeSquadName"]} - {matches["awaySquadName"]}"
    print(matches)
    return matches