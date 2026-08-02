import streamlit as st

from shot_analysis.loader import load_match
from shot_analysis.utils import get_match_score, get_opponent, get_opponent_id, summary_metrics
from shot_analysis.plots.shots_map import plot_shots, plot_team_shots, plot_xg_timeline
from shot_analysis.possessions import get_top_possessions
from shot_analysis.plots.possessions_map import plot_possession_chain

selected_match = st.session_state.selected_match

match_id = selected_match.id
event_dataset, metadata, df, raw_events = load_match(match_id)
team_goals, opponent_goals, opponent = get_match_score(df)


st.markdown(
    f"""
    <h1 style="font-size:32px;">
        SC Freiburg {team_goals}–{opponent_goals} {opponent}
    </h1>
    """,
    unsafe_allow_html=True,
)


opponent_name=get_opponent(df)
freiburg_num_shots, freiburg_xg, freiburg_on_target, opponent_num_shots, opponent_xg, opponent_on_target= summary_metrics(df, opponent_name)
col1, col2 = st.columns(2)
col3, col4 = st.columns(2)
col5, col6 = st.columns(2)

# Shots
col1.metric("Total Shots", freiburg_num_shots)
col2.metric("Total Shots", opponent_num_shots)

# xG
col3.metric("xG", f"{freiburg_xg:.2f}")
col4.metric("xG", f"{opponent_xg:.2f}")

# Shots on Target
col5.metric("Shots on Target", freiburg_on_target)
col6.metric("Shots on Target", opponent_on_target)

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "All Shots",
    "SC Freiburg Shots",
    f"{opponent_name} Shots",
    "Shots xG Timeline",
    "Freiburg Possession Chains",
    "Opponent Possession Chains"
])

with tab1:
    st.pyplot(plot_shots(df))

with tab2:
    st.pyplot(plot_team_shots(df, "SC Freiburg"))

with tab3:
    st.pyplot(plot_team_shots(df, opponent_name))

with tab4:
    st.pyplot(plot_xg_timeline(df))
#used LLM for this
with tab5:
    top=get_top_possessions(raw_events, df, 34)
    if not top:
        st.warning("No qualifying possessions found for this match.")
    else:
        labels = [
            f"{i+1}. xG: {p['xg']:.2f}"
            for i, p in enumerate(top)
        ]

        selected = st.radio(
            "Choose possession",
            range(len(top)),
            format_func=lambda i: labels[i],
            horizontal=True,
            key="freiburg_possession"
        )

        possession = top[selected]

        st.metric("xG", f"{possession['xg']:.2f}")

        fig = plot_possession_chain(possession)
        st.pyplot(fig)

with tab6:
    opponent_id = get_opponent_id(df)
    opponent_id=int(opponent_id)
    top=get_top_possessions(raw_events, df, opponent_id)
    if not top:
        st.warning("No qualifying possessions found for this match.")
    else:
        labels = [
            f"{i+1}. xG: {p['xg']:.2f}"
            for i, p in enumerate(top)
        ]

        selected = st.radio(
            "Choose possession",
            range(len(top)),
            format_func=lambda i: labels[i],
            horizontal=True,
            key="opponent_possession"
        )

        possession = top[selected]

        st.metric("xG", f"{possession['xg']:.2f}")

        fig = plot_possession_chain(possession)
        st.pyplot(fig)
    

st.write(f"{len(df)} events loaded.")