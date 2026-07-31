from settings import IMPECT_DIR, FREIBURG_ID
import os
import json
import pandas as pd
from data_helpers.squad_helper import merge_squadnames_matches
import streamlit as st

@st.cache_data
def load_freiburg_matches():
    matches_path = os.path.join(IMPECT_DIR, 'matches', 'matches_743.json')
    with open(matches_path) as f:
        data = json.load(f)
    df_matches = pd.json_normalize(data)
    df_matches = merge_squadnames_matches(df_matches)
    df_matches_scf = df_matches.loc[(df_matches["homeSquadId"] == FREIBURG_ID) | (df_matches["awaySquadId"] == FREIBURG_ID)]
    return df_matches, df_matches_scf
