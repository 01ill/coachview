import streamlit as st
import os
import pathlib
import json

from settings import IMPECT_DIR, FREIBURG_ID
from data_helpers.squad_helper import merge_squadnames_matches
import pandas as pd
import numpy as np

matches_path = os.path.join(IMPECT_DIR, 'matches', 'matches_743.json')
with open(matches_path) as f:
    data = json.load(f)
    #save it in dataframe
df_matches = pd.json_normalize(data)
df_matches = merge_squadnames_matches(df_matches)
df_matches_scf = df_matches.loc[(df_matches["homeSquadId"] == FREIBURG_ID) | (df_matches["awaySquadId"] == FREIBURG_ID)]


def sidebar_select_match():
    matches = df_matches_scf.reset_index(drop=True)
    selected_match = st.sidebar.selectbox("Match", options=matches.index, format_func=lambda i: f"{matches.loc[i, 'id']} - {matches.loc[i, "homeSquadName"]} - {matches.loc[i, "awaySquadName"]}")
    print(selected_match)
    return matches, matches.loc[selected_match]


main_page = st.Page("pages/main.py", title="Main Page")
preparation_page = st.Page("pages/preparation.py", title="Preparation")
pg = st.navigation([main_page, preparation_page])

matches, selected_match = sidebar_select_match()
st.session_state.matches = matches
st.session_state.selected_match = selected_match

# now get the lineup associated with the match
pg.run()