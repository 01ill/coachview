import streamlit as st

from shot_analysis.shot_clustering import cluster_shots, compute_cluster_statistics
from shot_analysis.plots.cluster_map import plot_cluster_map

st.title("Season Shot Profiles")

@st.cache_data
def get_cluster_analysis():

    cluster_df, model, scaler = cluster_shots(34)

    cluster_stats = compute_cluster_statistics(
        cluster_df,
        model,
        scaler,
    )

    return cluster_df, cluster_stats, model, scaler

cluster_df, cluster_stats, model, scaler = get_cluster_analysis()

st.pyplot(
    plot_cluster_map(
        cluster_df,
        cluster_stats,
        model,
        scaler,
    )
)

st.dataframe(
    cluster_stats,
    hide_index=True,
)

st.subheader("Shot Profile Insights")

for _, row in cluster_stats.iterrows():

    st.markdown(f"""
### {row['profile']}

- **Shots:** {row['shots']} ({row['shot_percentage']:.1f}%)
- **Average xG:** {row['average_xg']:.3f}
- **Goal Rate:** {row['goal_percentage']:.1f}%
- **Saved:** {row['saved_percentage']:.1f}%
- **Off Target:** {row['off_target_percentage']:.1f}%
""")