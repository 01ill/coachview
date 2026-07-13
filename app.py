import streamlit as st
import os
import pathlib
import json

from settings import IMPECT_DIR, FREIBURG_ID
from data_helpers.kpi_helper import load_kpi_def, load_player_kpi, kpi_statistics, kpi_long_df, get_player_kpi
from data_helpers.squad_helper import merge_squadnames_matches
from data_helpers.player_helper import merge_playernames
import pandas as pd
import numpy as np

matches_path = os.path.join(IMPECT_DIR, 'matches', 'matches_743.json')
with open(matches_path) as f:
    data = json.load(f)
    #save it in dataframe
df_matches = pd.json_normalize(data)
df_matches = merge_squadnames_matches(df_matches)
df_matches_scf = df_matches.loc[(df_matches["homeSquadId"] == FREIBURG_ID) | (df_matches["awaySquadId"] == FREIBURG_ID)]

players_path = os.path.join(IMPECT_DIR, 'players', 'players_743.json')
with open(players_path) as f:
    data = json.load(f)
    #save it in dataframe
df_players = pd.json_normalize(data)

def sidebar_select_match():
    matches = df_matches_scf.reset_index(drop=True)
    selected_match = st.sidebar.selectbox("Match", options=matches.index, format_func=lambda i: f"{matches.loc[i, 'id']} - {matches.loc[i, "homeSquadName"]} - {matches.loc[i, "awaySquadName"]}")
    print(selected_match)
    return matches, matches.loc[selected_match]

st.header("First Test")

matches, selected_match = sidebar_select_match()
col1, col2, col3 = st.columns(3)
with col1:
    passes_clicked = st.sidebar.button("Summary", use_container_width=True)
with col2:
    summary_clicked = st.sidebar.button("Preparation", use_container_width=True)
with col3:
    summary_clicked = st.sidebar.button("Half-Time Analysis", use_container_width=True)

#st.write(selected_match)

# now get the lineup associated with the match
lineup_path = os.path.join(IMPECT_DIR, 'lineups', f'lineups_{selected_match.id}.json')
with open(lineup_path) as f:
    data = json.load(f)
    #save it in dataframe
df_lineup = pd.json_normalize(data)

# merge lineups with player names
name_lookup = df_players.set_index("id")["commonname"].to_dict()
home_players = df_lineup.loc[0, "squadHome.players"]
away_players = df_lineup.loc[0, "squadAway.players"]
df_lineup.at[0, "squadHome.players"] = pd.DataFrame(df_lineup.loc[0, "squadHome.players"]).merge(
    df_players[["id", "commonname"]], on="id", how="left"
)
df_lineup.at[0, "squadAway.players"] = pd.DataFrame(df_lineup.loc[0, "squadAway.players"]).merge(
    df_players[["id", "commonname"]], on="id", how="left"
)
#st.dataframe(df_lineup["squadAway.players"])

df_kpi_def = load_kpi_def()
#st.write(df_kpi_def)

def select_kpis():
    kpi_def = df_kpi_def.reset_index(drop=True)
    selected_kpis = st.multiselect("KPI Selector", kpi_def.index, format_func=lambda i: f"{kpi_def.loc[i, "details.label"]} - {kpi_def.loc[i, "details.definition"]}")
    return kpi_def, kpi_def.loc[selected_kpis]

kpi_def, selected_kpis = select_kpis()

# print(kpi_def, selected_kpis)


df_kpi_all = load_player_kpi(df_matches.id)
long_df = kpi_long_df(df_matches, df_kpi_all)
kpi_range = kpi_statistics(long_df)
all_match_ids = df_kpi_all["matchId"].tolist()

players_df = get_player_kpi(df_matches, long_df, df_kpi_all)
players_df = merge_playernames(players_df)
copy1 = players_df.copy()
print(players_df)
print(copy1)
kpi_columns_all = [c for c in players_df.columns if c.startswith("kpi_")]
n_matches = len(all_match_ids)
# player_kpi sometimes returns just a None value instead of list with None
for col in kpi_columns_all:
    players_df[col] = players_df[col].apply(
        lambda lst: [-10] * n_matches if not isinstance(lst, list)
        else [float(x) if pd.notnull(x) else -10 for x in lst]
    )

selected_kpi_ids = selected_kpis["id"].tolist()
selected_cols = [f"kpi_{int(k)}" for k in selected_kpi_ids]
selected_cols = [c for c in selected_cols if c in players_df.columns]

label_map = dict(zip(
    [f"kpi_{int(k)}" for k in selected_kpi_ids],
    selected_kpis["details.label"]
))

match_row = df_kpi_all[df_kpi_all["matchId"] == selected_match.id].iloc[0]

home_ids = [p["id"] for p in match_row["squadHome.players"]]
away_ids = [p["id"] for p in match_row["squadAway.players"]]

squad_ids = set(home_ids) | set(away_ids)
display_df = players_df[players_df["playerId"].isin(squad_ids)].copy()
squad_side = {pid: "home" for pid in home_ids}
squad_side.update({pid: "away" for pid in away_ids})
display_cols = ["playerId", "playerName"] + selected_cols
display_df = players_df[display_cols]
display_df["matchSide"] = display_df["playerId"].map(squad_side)

def pad(lo, hi, frac=0.05):
    span = hi - lo
    p = span * frac if span else 1.0
    return lo - p, hi + p

column_config = {}
for col in selected_cols:
    y_min, y_max = pad(*kpi_range.loc[col, ["min", "max"]])
    column_config[col] = st.column_config.LineChartColumn(
        label=label_map[col],
        width="medium",
        y_min=-10, y_max=y_max,
        color="auto"
    )

col_home, col_away = st.columns(2)
with col_home:
    st.subheader("Home")
    st.dataframe(
        display_df[display_df["matchSide"] == "home"].drop(columns="matchSide"),
        column_config=column_config,
        hide_index=True,
    )

with col_away:
    st.subheader("Away")
    st.dataframe(
        display_df[display_df["matchSide"] == "away"].drop(columns="matchSide"),
        column_config=column_config,
        hide_index=True,
    )

from mplsoccer import PyPizza
avg_cols = [f"avg_{col}" for col in selected_cols]
playerId = 1294
values = copy1[["avg_kpi_0"]]["avg_kpi_0"][0]
percentiles_df = players_df[avg_cols].rank(pct=True) * 100
player_idx = players_df.index[players_df["playerId"] == playerId][0]
player_name = players_df[players_df["playerId"] == playerId].playerName.iloc[0]
values = percentiles_df.loc[player_idx].fillna(0).round(0).astype(int).tolist()
labels = selected_kpis["details.label"].tolist()
# instantiate PyPizza class
baker = PyPizza(
    params=labels,                  # list of parameters
    straight_line_color="#000000",  # color for straight lines
    straight_line_lw=1,             # linewidth for straight lines
    last_circle_lw=1,               # linewidth of last circle
    other_circle_lw=1,              # linewidth for other circles
    other_circle_ls="-."            # linestyle for other circles
)

# plot pizza
fig, ax = baker.make_pizza(
    values,              # list of values
    figsize=(8, 8),      # adjust figsize according to your need
    param_location=110,  # where the parameters will be added
    kwargs_slices=dict(
        facecolor="cornflowerblue", edgecolor="#000000",
        zorder=2, linewidth=1
    ),                   # values to be used when plotting slices
    kwargs_params=dict(
        color="#000000", fontsize=12,
        va="center"
    ),                   # values to be used when adding parameter
    kwargs_values=dict(
        color="#000000", fontsize=12,
        zorder=3,
        bbox=dict(
            edgecolor="#000000", facecolor="cornflowerblue",
            boxstyle="round,pad=0.2", lw=1
        )
    )                    # values to be used when adding parameter-values
)

# add title
fig.text(
    0.515, 0.97, f"{player_name}", size=18,
    ha="center", color="#000000"
)

# add subtitle
fig.text(
    0.515, 0.942,
    "Percentile Rank vs League Forwards | Season 2023-24",
    size=15,
    ha="center", color="#000000"
)
st.pyplot(fig)