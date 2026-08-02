import streamlit as st
from data_helpers.match_helper import load_freiburg_matches

st.set_page_config(layout="wide")

df_matches, df_matches_scf = load_freiburg_matches()

def sidebar_select_match():
    matches = df_matches_scf.reset_index(drop=True)
    selected_match = st.sidebar.selectbox(
        "Match", 
        options=matches.index,
        format_func=lambda i: f"{matches.loc[i, 'id']} - {matches.loc[i, "homeSquadName"]} - {matches.loc[i, "awaySquadName"]}"
    )
    return matches, matches.loc[selected_match]

main_page = st.Page("pages/main.py", title="Main Page")
seasonal_page= st.Page("pages/season_insight.py", title="Season Shot Profiles")
shots_page = st.Page("pages/shots.py", title="Shot Analysis")
preparation_page = st.Page("pages/substitutions.py", title="Substitutions")
pg = st.navigation([main_page, seasonal_page, shots_page, preparation_page])

matches, selected_match = sidebar_select_match()
st.session_state.matches = matches
st.session_state.selected_match = selected_match

pg.run()
