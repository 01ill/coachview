"""
LLM notice:
- the functions where LLM use is indicated are predominantly written by LLM
- all other code is mostly written by hand and details/bugs were fixed/helped with by LLM
"""

import streamlit as st
from settings import FREIBURG_ID, PLOTLY_CLUSTER
from highlight_text import fig_text
import pandas as pd

from data_helpers.kpi_helper import build_kpi_catalog, selected_kpi_columns, EXTRA_KPI_INCLUDED, EXTRA_KPI_LABELS
from data_helpers.player_helper import load_players, player_pipeline, compute_scores
from data_helpers.squad_helper import load_lineups
from data_helpers.cluster_helper import prepare_dataset, cluster_silhuette, cluster_squad, bench_options, plot_cluster_plotly, label_clusters

if "selected_start_player" not in st.session_state:
    st.session_state.selected_start_player = None
if "selected_substitute_player" not in st.session_state:
    st.session_state.selected_substitute_player = None

st.header("Match Preparation")

matches = st.session_state.matches
selected_match = st.session_state.selected_match

# def load_extra_kpi_definitions() -> pd.DataFrame:
#     rows = []
#     for feature in EXTRA_KPI_INCLUDED:
#         rows.append(
#             {
#                 "id": f"extra_{feature}",
#                 "details.label": EXTRA_KPI_LABELS.get(feature, feature.replace("_", " ").title()),
#                 "details.definition": f"Extra player-match feature: {feature}",
#             }
#         )
#     return pd.DataFrame(rows)

# df_kpi_def = load_kpi_def()
# df_extra_kpi_def = load_extra_kpi_definitions()
# df_kpi_def = pd.concat([df_kpi_def, df_extra_kpi_def], ignore_index=True)
kpi_catalog = build_kpi_catalog()
DEFAULT_KPI_LABELS = [
    "Goals",
    "Successful Passes",
    "Bypassed Opponents",
    "Ball Win Added Teammates",
    "Ball Win Removed Opponents",
    "Ball Win Number",
    "Assists",
    "Shot-based xG",
    "Successful Passes",
    "Won Ground Duels",
    "Won Aerial Duels",
    "Dribbles",
    "Total Shots Blocked",
    "Opponent Box Actions per 90"
]
def select_kpis():
    default_idx = kpi_catalog.index[kpi_catalog["label"].isin(DEFAULT_KPI_LABELS)].tolist()
    selected = st.multiselect(
        "KPI Selector",
        kpi_catalog.index,
        default=default_idx,
        format_func=lambda i: kpi_catalog.loc[i, "picker_label"],
    )
    return kpi_catalog.loc[selected].copy()
selected_kpis = select_kpis()
labels = selected_kpis["label"].tolist()

players_df, long_df, kpi_range = player_pipeline(matches)
selected_kpi_ids = selected_kpis["id"].tolist()
# selected_cols = [f"kpi90_{int(k)}" for k in selected_kpi_ids]
# selected_cols = [c for c in selected_cols if c in players_df.columns]
#avg_cols = [f"avg_{col}_pct" for col in selected_cols]
# label_map = dict(zip(
#     [f"kpi90_{int(k)}" for k in selected_kpi_ids],
#     selected_kpis["details.label"]
# ))
# now find the players of the specific match
freiburg_is_home = selected_match.homeSquadId == FREIBURG_ID
freiburg_players = load_players(FREIBURG_ID)
# this finds all players which played for freiburg this season
# gives wrong entries when a player has made a mid season transfer
# TODO: match kpi event date with date of fixture and then only pick kpi from previous matches
freiburg_player_ids = long_df[long_df["squadId"] == FREIBURG_ID]["id"].unique()
lineup_df = load_lineups(selected_match.id)
freiburg_starter_ids = [player["playerId"] for player in lineup_df.loc[0, "squadHome.startingPositions" if freiburg_is_home else "squadAway.startingPositions"]]
players_df = players_df[players_df["playerId"].isin(freiburg_player_ids)]
players_df["isStarter"] = players_df["playerId"].isin(freiburg_starter_ids)
selected_raw_cols = [c for c in selected_kpis["raw_col"].tolist() if c in players_df.columns]
# merging the extra KPIs was mostly helped by LLM
pct_cols = [c for c in selected_kpis["pct_col"].tolist() if c in players_df.columns]
label_map = selected_kpis.set_index("raw_col")["label"].to_dict()
players_df = compute_scores(players_df, pct_cols)

display_cols = ["playerName", "score", "playerPosition", "playerLine", "isStarter", "playerId"] + selected_raw_cols
display_df = players_df[display_cols]
start_df = display_df[display_df["isStarter"] == True].drop(columns="isStarter").reset_index(drop=True)
substitute_df = display_df[display_df["isStarter"] == False].drop(columns="isStarter").reset_index(drop=True)

# def pad(lo, hi, frac=0.05):
#     span = hi - lo
#     p = span * frac if span else 1.0
#     return lo - p, hi + p

column_config = {}
# for col in selected_cols:
#     y_min, y_max = pad(*kpi_range.loc[col, ["min", "max"]])
#     column_config[col] = st.column_config.LineChartColumn(
#         label=label_map[col],
#         width="medium",
#         y_min=-10, y_max=y_max,
#         color="auto"
#     )

col_starters, col_substitutes = st.columns(2)
with col_starters:
    st.subheader("Starting 11")
    styled = start_df[["playerName", "playerPosition", "score"]].style.background_gradient(
        subset=["score"],
        cmap="RdYlGn",
        vmin=0,
        vmax=100
    ).format({'score': '{:.1f}'})
    start_player = st.dataframe(
        styled,
        column_config=column_config,
        hide_index=True,
        on_select="rerun",
        selection_mode="single-row",
        key="start_table"
    )
    if start_player.selection and start_player.selection["rows"]:
        row_idx = start_player.selection["rows"][0]
        st.session_state.selected_start_player = start_df.iloc[row_idx]["playerId"]
    else:
        st.session_state.selected_start_player = None

with col_substitutes:
    st.subheader("Substitutes")
    styled = substitute_df[["playerName", "playerPosition", "score"]].style.background_gradient(
        subset=["score"],
        cmap="RdYlGn",
        vmin=0,
        vmax=100
    ).format({'score': '{:.1f}'})
    substitute_player = st.dataframe(
        styled,
        column_config=column_config,
        hide_index=True,
        on_select="rerun",
        selection_mode="single-row",
        key="substitute_table"
    )
    if substitute_player.selection and substitute_player.selection["rows"]:
        row_idx = substitute_player.selection["rows"][0]
        st.session_state.selected_substitute_player = substitute_df.iloc[row_idx]["playerId"]
    else:
        st.session_state.selected_substitute_player = None

from mplsoccer import PyPizza
# playerId = 68285
playerId1 = st.session_state.selected_start_player
playerId2 = st.session_state.selected_substitute_player

if not pct_cols:
    st.info("Pick at least one KPI to compare players")
elif playerId1 is None or playerId2 is None:
    st.info("Select two players to compare")
else:
    player_idx1 = players_df.index[players_df["playerId"] == playerId1][0]
    player_idx2 = players_df.index[players_df["playerId"] == playerId2][0]
    player_name1 = players_df[players_df["playerId"] == playerId1].playerName.iloc[0]
    player_name2 = players_df[players_df["playerId"] == playerId2].playerName.iloc[0]
    player_pos1 = players_df[players_df["playerId"] == playerId1].playerPosition.iloc[0]
    player_pos2 = players_df[players_df["playerId"] == playerId2].playerPosition.iloc[0]
    values1 = players_df.loc[players_df["playerId"] == playerId1, pct_cols].round(1).iloc[0].tolist()
    values2 = players_df.loc[players_df["playerId"] == playerId2, pct_cols].round(1).iloc[0].tolist()
    labels = selected_kpis["label"].tolist()    
    # instantiate PyPizza class
    baker = PyPizza(
        params=labels,                  # list of parameters
        background_color="#ebebe9",
        straight_line_color="#000000",  # color for straight lines
        straight_line_lw=1,             # linewidth for straight lines
        last_circle_lw=1,               # linewidth of last circle
        other_circle_lw=1,              # linewidth for other circles
        other_circle_ls="-.",            # linestyle for other circles
        last_circle_color="#222222"
    )

    # plot pizza
    fig, ax = baker.make_pizza(
        values1,
        compare_values=values2,
        figsize=(9, 9),      # adjust figsize according to your need
        #param_location=110,  # where the parameters will be added
        kwargs_slices=dict(
            facecolor="cornflowerblue", edgecolor="#222222",
            zorder=2, linewidth=1
        ),                   # values to be used when plotting slices
        kwargs_compare=dict(
            facecolor="#ff9300", edgecolor="#222222", zorder=2, linewidth=1
        ),
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
        ),                    # values to be used when adding parameter-values
        kwargs_compare_values=dict(
            color="#000000", fontsize=12, zorder=3,
            bbox=dict(edgecolor="#000000", facecolor="#ff9300", boxstyle="round,pad=0.2", lw=1)
        )
    )

    fig_text(
        0.515, 0.99, f"<{player_name1}> vs <{player_name2}>", size=18, fig=fig,
        highlight_textprops=[{"color": "#1a78cf"}, {"color": "#ee8900"}],
        ha="center", color="#000000"
    )

    # add subtitle
    fig.text(
        0.515, 0.942,
        f"Percentile Rank | Season 2023-24",
        size=15,
        ha="center", color="#000000"
    )
    st.pyplot(fig)

st.subheader("Squad Clustering")

target_player_id = st.session_state.selected_start_player
if target_player_id is None:
    st.info("Select a starting player above first.")
elif len(pct_cols) < 2:
    st.info("Select two or more KPIs to generate the clustering")
else:
    exclude_gk = st.checkbox("Exclude goalkeepers", value=True)
    exclude_positions = ["GOALKEEPER"] if exclude_gk else None
    cluster_df, X = prepare_dataset(players_df, freiburg_player_ids, pct_cols, exclude_positions)
    best_k, _ = cluster_silhuette(X)
    n_clusters = st.slider(f"Number of clusters. Best silhuette score: {best_k}", 2, min(8, len(cluster_df) - 1), best_k)

    cluster_df = cluster_squad(cluster_df, X, n_clusters)
    cluster_labels = label_clusters(cluster_df, pct_cols, label_map)
    bench_ids = set(freiburg_player_ids) - set(freiburg_starter_ids)
    options = bench_options(cluster_df, X, target_player_id, bench_only_ids=bench_ids)
    only_benched = st.checkbox("Show only benched players", value=True)
    if PLOTLY_CLUSTER:
        fig = plot_cluster_plotly(cluster_df, cluster_labels, target_player_id, options, 3, only_benched)
        st.plotly_chart(fig)
    # else:
    #     fig = plot_cluster(cluster_df, cluster_labels, target_player_id, options)
    #     st.pyplot(fig)
    #     plt.close(fig)

    target_name = players_df.loc[players_df["playerId"] == target_player_id, "playerName"].iloc[0]
    target_archetype = cluster_labels[cluster_df.loc[cluster_df["playerId"] == target_player_id, "cluster"].iloc[0]]
    target_position = cluster_df.loc[cluster_df["playerId"] == target_player_id, "playerLine"].iloc[0]
    st.write(f"Bench options for **{target_name}** ({target_position}, {target_archetype} archetype): same archetype, same line, or close distance:")
    styled = options[["playerName", "playerLine", "distance", "score"]].rename(columns={"playerName": "Player", "playerPosition": "Position",
                                        "distance": "Distance (lower = closer)", "score": "Score"}
    ).style.background_gradient(
        subset=["Score"], cmap="RdYlGn", vmin=0, vmax=100
    ).background_gradient(
        subset=["Distance (lower = closer)"], cmap="RdYlGn_r"
    ).format({'Score': '{:.1f}', 'Distance (lower = closer)': "{:.2f}"})

    st.dataframe(styled, hide_index=True)
