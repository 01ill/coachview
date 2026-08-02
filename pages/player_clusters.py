from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------
APP_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = APP_DIR / "outputs"


# ---------------------------------------------------------------------
# Styling
# ---------------------------------------------------------------------
st.markdown(
    """
    <style>
        .block-container {
            padding-top: 1.4rem;
            padding-bottom: 2rem;
        }

        div[data-testid="stMetric"] {
            background: rgba(120, 120, 120, 0.08);
            border: 1px solid rgba(120, 120, 120, 0.20);
            border-radius: 12px;
            padding: 12px;
        }

        .cluster-card {
            border: 1px solid rgba(120, 120, 120, 0.25);
            border-radius: 14px;
            padding: 16px;
            margin-bottom: 12px;
            background: rgba(120, 120, 120, 0.06);
        }

        .small-note {
            opacity: 0.75;
            font-size: 0.90rem;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------
@st.cache_data
def load_data() -> tuple[pd.DataFrame, pd.DataFrame, list[str], dict[int, str]]:
    """Load the clustered player-match data and supporting metadata."""
    clusters_path = OUTPUT_DIR / "player_match_clusters.csv"
    summary_path = OUTPUT_DIR / "cluster_summary.csv"
    features_path = OUTPUT_DIR / "cluster_features.json"
    names_path = OUTPUT_DIR / "cluster_name_map.json"

    required_files = [
        clusters_path,
        summary_path,
        features_path,
        names_path,
    ]

    missing_files = [str(path) for path in required_files if not path.exists()]
    if missing_files:
        raise FileNotFoundError(
            "Missing required files:\n" + "\n".join(missing_files)
        )

    clustered = pd.read_csv(clusters_path)
    summary = pd.read_csv(summary_path)

    with open(features_path, "r", encoding="utf-8") as file:
        cluster_features = json.load(file)

    with open(names_path, "r", encoding="utf-8") as file:
        raw_name_map = json.load(file)

    cluster_name_map = {int(key): value for key, value in raw_name_map.items()}

    clustered["cluster"] = pd.to_numeric(
        clustered["cluster"],
        errors="coerce",
    ).astype("Int64")

    clustered["cluster_name"] = clustered["cluster"].map(cluster_name_map).fillna(
        clustered.get("cluster_name", "Unknown cluster")
    )

    return clustered, summary, cluster_features, cluster_name_map


try:
    df, cluster_summary, cluster_features, cluster_name_map = load_data()
except Exception as error:
    st.error(f"Could not load the project files: {error}")
    st.stop()


# ---------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------
def available_columns(columns: list[str]) -> list[str]:
    """Return only columns that exist in the clustered dataframe."""
    return [column for column in columns if column in df.columns]


def humanize_feature(feature: str) -> str:
    """Convert a snake_case feature name into a readable label."""
    return (
        feature.replace("_per90", " per 90")
        .replace("_pxt_", "_pxT_")
        .replace("_", " ")
        .title()
        .replace("Pxt", "pxT")
    )


def cluster_profile_table(data: pd.DataFrame) -> pd.DataFrame:
    """Calculate mean feature values for each selected cluster."""
    selected_features = available_columns(cluster_features)
    return (
        data.groupby(["cluster", "cluster_name"])[selected_features]
        .mean()
        .reset_index()
    )


def standardized_cluster_profile(data: pd.DataFrame) -> pd.DataFrame:
    """Calculate z-scored cluster profiles for heatmaps and radar charts."""
    selected_features = available_columns(cluster_features)
    feature_data = data[selected_features].copy()

    std = feature_data.std(ddof=0).replace(0, np.nan)
    standardized = (feature_data - feature_data.mean()) / std
    standardized = standardized.fillna(0)

    standardized["cluster"] = data["cluster"].to_numpy()
    standardized["cluster_name"] = data["cluster_name"].to_numpy()

    return (
        standardized.groupby(["cluster", "cluster_name"])[selected_features]
        .mean()
        .reset_index()
    )


def radar_figure(
    profile_row: pd.Series,
    features: list[str],
    title: str,
) -> go.Figure:
    """Create one radar chart from a standardized cluster profile row."""
    labels = [humanize_feature(feature) for feature in features]
    values = [float(profile_row[feature]) for feature in features]

    labels_closed = labels + [labels[0]]
    values_closed = values + [values[0]]

    figure = go.Figure()
    figure.add_trace(
        go.Scatterpolar(
            r=values_closed,
            theta=labels_closed,
            fill="toself",
            name=title,
        )
    )
    figure.update_layout(
        title=title,
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[-1.5, 2.0],
            )
        ),
        showlegend=False,
        height=520,
        margin=dict(l=40, r=40, t=70, b=40),
    )
    return figure


def top_cluster_features(
    standardized_profiles: pd.DataFrame,
    cluster_id: int,
    n: int = 5,
) -> tuple[pd.Series, pd.Series]:
    """Return the strongest and weakest standardized features."""
    row = standardized_profiles.loc[
        standardized_profiles["cluster"] == cluster_id,
        available_columns(cluster_features),
    ]

    if row.empty:
        return pd.Series(dtype=float), pd.Series(dtype=float)

    sorted_values = row.iloc[0].sort_values(ascending=False)
    return sorted_values.head(n), sorted_values.tail(n)


all_players = sorted(df["player_name"].dropna().unique().tolist())


# ---------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------
st.title("SC Freiburg Player Performance Clusters")
st.write(
    "Interactive analysis of match-level player performance archetypes "
    "identified with K-Means clustering."
)


# ---------------------------------------------------------------------
# Filtered data download (used on Cluster Explorer and Player Explorer)
# ---------------------------------------------------------------------
download_columns = available_columns(
    [
        "match_id",
        "player_id",
        "player_name",
        "player_position",
        "minutes",
        "cluster",
        "cluster_name",
        "pca_1",
        "pca_2",
        "silhouette_value",
        "distance_to_centroid",
    ]
    + cluster_features
)

csv_bytes = df[download_columns].to_csv(index=False).encode("utf-8")


# ---------------------------------------------------------------------
# Navigation
# ---------------------------------------------------------------------
tab_overview, tab_clusters, tab_players = st.tabs(
    [
        "Overview",
        "Cluster Explorer",
        "Player Explorer",
    ]
)


# ---------------------------------------------------------------------
# Overview tab
# ---------------------------------------------------------------------
with tab_overview:
    metric_1, metric_2, metric_3, metric_4 = st.columns(4)

    metric_1.metric(
        "Player-match samples",
        f"{len(df):,}",
    )
    metric_2.metric(
        "Unique players",
        f"{df['player_id'].nunique():,}",
    )
    metric_3.metric(
        "Matches",
        f"{df['match_id'].nunique():,}",
    )
    metric_4.metric(
        "Performance clusters",
        f"{df['cluster'].nunique():,}",
    )

    st.subheader("Cluster distribution")

    distribution = (
        df.groupby(["cluster", "cluster_name"])
        .size()
        .reset_index(name="count")
    )

    distribution["percentage"] = (
        distribution["count"] / distribution["count"].sum() * 100
    )

    distribution_figure = px.bar(
        distribution,
        x="cluster_name",
        y="count",
        text="percentage",
        labels={
            "cluster_name": "Performance cluster",
            "count": "Player-match performances",
        },
    )
    distribution_figure.update_traces(
        texttemplate="%{text:.1f}%",
        textposition="outside",
    )
    distribution_figure.update_layout(
        showlegend=False,
        xaxis_tickangle=-15,
        height=430,
    )
    st.plotly_chart(distribution_figure, use_container_width=True)

    st.subheader("PCA map of player-match performances")

    hover_columns = available_columns(
        [
            "player_name",
            "player_position",
            "match_id",
            "minutes",
            "silhouette_value",
        ]
    )

    pca_figure = px.scatter(
        df,
        x="pca_1",
        y="pca_2",
        color="cluster_name",
        hover_name="player_name",
        hover_data=hover_columns,
        labels={
            "pca_1": "Principal Component 1",
            "pca_2": "Principal Component 2",
            "cluster_name": "Cluster",
        },
        opacity=0.75,
    )
    cluster_colors = {trace.name: trace.marker.color for trace in pca_figure.data}
    centroids = df.groupby("cluster_name")[["pca_1", "pca_2"]].mean().reset_index()

    for _, centroid_row in centroids.iterrows():
        pca_figure.add_trace(
            go.Scatter(
                x=[centroid_row["pca_1"]],
                y=[centroid_row["pca_2"]],
                mode="markers",
                marker=dict(
                    symbol="circle",
                    size=32,
                    color=cluster_colors.get(centroid_row["cluster_name"], "white"),
                    line=dict(width=2, color="white"),
                ),
                name=f"{centroid_row['cluster_name']} centroid",
                legendgroup=centroid_row["cluster_name"],
                showlegend=False,
                hovertemplate=f"<b>{centroid_row['cluster_name']} centroid</b><extra></extra>",
            )
        )

    pca_figure.add_trace(
        go.Scatter(
            x=[None],
            y=[None],
            mode="markers",
            marker=dict(symbol="circle", size=18, color="white", line=dict(width=1, color="black")),
            name="Cluster centroid",
            showlegend=True,
        )
    )

    pca_figure.update_layout(height=650)
    st.plotly_chart(pca_figure, use_container_width=True)

    st.caption(
        "PCA is used for two-dimensional visualization. "
        "The final K-Means model was fitted on the full standardized feature space."
    )

    with st.expander("What each cluster means"):
        cluster_insights = {
            "Defensive Performance": {
                "signals": (
                    "Above-average interceptions, ground duels, clearances, "
                    "blocks, and pressure actions, with below-average shots "
                    "and final third involvement."
                ),
                "analyst_use": (
                    "Use this group to benchmark defenders and defensive "
                    "midfielders against peers doing the same job, scout "
                    "like-for-like replacements, and flag players who defend "
                    "far above or below the group's norm."
                ),
            },
            "High-Impact Attacking Performance": {
                "signals": (
                    "Well above-average shots, possession threat (pxT), final "
                    "third actions, and opponent box involvement, with lower "
                    "defensive activity."
                ),
                "analyst_use": (
                    "Use this group to identify your most dangerous attacking "
                    "contributors, compare forwards and creative players on "
                    "output rather than reputation, and track whether a "
                    "player's attacking threat is rising or fading over time."
                ),
            },
            "Support Performance": {
                "signals": (
                    "Values close to or slightly below average across most "
                    "metrics, with a leaning toward wide and deeper support "
                    "roles rather than a standout attacking or defensive "
                    "signature."
                ),
                "analyst_use": (
                    "Use this group to spot squad players who provide balance "
                    "and availability rather than standout numbers, useful for "
                    "assessing squad depth and rotation options."
                ),
            },
        }

        cluster_order = (
            df[["cluster", "cluster_name", "cluster_description"]]
            .drop_duplicates()
            .sort_values("cluster")
        )

        cluster_cards = st.columns(len(cluster_order))

        for card, (_, cluster_row) in zip(cluster_cards, cluster_order.iterrows()):
            name = cluster_row["cluster_name"]
            insight = cluster_insights.get(name, {})
            signals = insight.get("signals", cluster_row["cluster_description"])
            analyst_use = insight.get(
                "analyst_use",
                "Use this group to compare players who share this performance profile.",
            )

            with card:
                st.markdown(
                    f"""
                    <div class="cluster-card">
                        <h4>{name}</h4>
                        <p><strong>What it means:</strong> {signals}</p>
                        <p><strong>How analysts can use it:</strong> {analyst_use}</p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )


# ---------------------------------------------------------------------
# Cluster Explorer tab
# ---------------------------------------------------------------------
with tab_clusters:
    st.subheader("Cluster feature heatmap")

    heatmap_profiles = standardized_cluster_profile(df)
    heatmap_features = available_columns(cluster_features)

    heatmap_data = heatmap_profiles.set_index("cluster_name")[heatmap_features]
    heatmap_data.columns = [
        humanize_feature(feature) for feature in heatmap_data.columns
    ]

    heatmap_figure = px.imshow(
        heatmap_data,
        labels=dict(x="Feature", y="Cluster", color="Z-score"),
        color_continuous_scale="RdBu_r",
        zmin=-2,
        zmax=2,
        aspect="auto",
        text_auto=".1f",
    )
    heatmap_figure.update_layout(
        height=380,
        xaxis_tickangle=-35,
        margin=dict(l=10, r=10, t=30, b=10),
    )
    st.plotly_chart(heatmap_figure, use_container_width=True)

    with st.expander("View cluster feature table"):
        st.dataframe(
            heatmap_data.style.format("{:.2f}").background_gradient(
                cmap="RdBu_r", vmin=-2, vmax=2
            ),
            use_container_width=True,
        )

    st.subheader("Explore one performance cluster")

    cluster_options = (
        df[["cluster", "cluster_name"]]
        .drop_duplicates()
        .sort_values("cluster")
    )

    selected_cluster_name = st.selectbox(
        "Select a cluster",
        options=cluster_options["cluster_name"].tolist(),
    )

    selected_cluster_id = int(
        cluster_options.loc[
            cluster_options["cluster_name"] == selected_cluster_name,
            "cluster",
        ].iloc[0]
    )

    selected_cluster_df = df[
        df["cluster"] == selected_cluster_id
    ].copy()

    if selected_cluster_df.empty:
        st.warning("The active sidebar filters remove all rows from this cluster.")
    else:
        summary_row = cluster_summary[
            cluster_summary["cluster"] == selected_cluster_id
        ]

        card_1, card_2, card_3, card_4 = st.columns(4)
        card_1.metric("Samples", len(selected_cluster_df))
        card_2.metric(
            "Players",
            selected_cluster_df["player_id"].nunique(),
        )
        card_3.metric(
            "Mean silhouette",
            f"{selected_cluster_df['silhouette_value'].mean():.3f}",
        )
        card_4.metric(
            "Median minutes",
            f"{selected_cluster_df['minutes'].median():.1f}",
        )

        description = (
            selected_cluster_df["cluster_description"].dropna().iloc[0]
            if "cluster_description" in selected_cluster_df.columns
            and selected_cluster_df["cluster_description"].notna().any()
            else "No description is available."
        )

        st.markdown(
            f"""
            <div class="cluster-card">
                <h3>{selected_cluster_name}</h3>
                <p>{description}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        standardized_profiles = standardized_cluster_profile(df)
        strongest, weakest = top_cluster_features(
            standardized_profiles,
            selected_cluster_id,
        )

        profile_left, profile_right = st.columns([1.25, 1])

        with profile_left:
            profile_row = standardized_profiles[
                standardized_profiles["cluster"] == selected_cluster_id
            ].iloc[0]

            radar_features = (
                pd.concat([strongest, weakest])
                .abs()
                .sort_values(ascending=False)
                .head(10)
                .index.tolist()
            )

            st.plotly_chart(
                radar_figure(
                    profile_row=profile_row,
                    features=radar_features,
                    title=f"{selected_cluster_name}: standardized profile",
                ),
                use_container_width=True,
            )

        with profile_right:
            st.markdown("#### Strongest features")
            strongest_table = strongest.rename("z_score").reset_index()
            strongest_table.columns = ["feature", "z_score"]
            strongest_table["feature"] = strongest_table["feature"].map(
                humanize_feature
            )
            st.dataframe(
                strongest_table.style.format({"z_score": "{:.2f}"}),
                use_container_width=True,
                hide_index=True,
            )

            st.markdown("#### Weakest features")
            weakest_table = weakest.rename("z_score").reset_index()
            weakest_table.columns = ["feature", "z_score"]
            weakest_table["feature"] = weakest_table["feature"].map(
                humanize_feature
            )
            st.dataframe(
                weakest_table.style.format({"z_score": "{:.2f}"}),
                use_container_width=True,
                hide_index=True,
            )

        st.subheader("Position composition")

        position_distribution = (
            selected_cluster_df["player_position"]
            .value_counts()
            .rename_axis("position")
            .reset_index(name="count")
        )

        position_figure = px.bar(
            position_distribution,
            x="position",
            y="count",
            labels={
                "position": "Player position",
                "count": "Player-match performances",
            },
        )
        position_figure.update_layout(
            showlegend=False,
            xaxis_tickangle=-25,
            height=430,
        )
        st.plotly_chart(position_figure, use_container_width=True)

        st.subheader("Most representative performances")

        representative_columns = available_columns(
            [
                "player_name",
                "player_position",
                "match_id",
                "minutes",
                "distance_to_centroid",
                "silhouette_value",
                "shots_per90",
                "total_pxt_per90",
                "passes_per90",
                "loose_ball_regains_per90",
            ]
        )

        representative_df = selected_cluster_df.sort_values(
            "distance_to_centroid"
        )[representative_columns].head(15)

        st.dataframe(
            representative_df,
            use_container_width=True,
            hide_index=True,
        )

    st.markdown("---")
    st.subheader("Download filtered data")
    st.download_button(
        label="Download filtered player-match data as CSV",
        data=csv_bytes,
        file_name="filtered_player_match_clusters.csv",
        mime="text/csv",
        key="download_clusters_tab",
    )


# ---------------------------------------------------------------------
# Player Explorer tab
# ---------------------------------------------------------------------
with tab_players:
    st.subheader("Player performance history")

    selected_player = st.selectbox(
        "Select a player",
        options=all_players,
    )

    player_df = df[
        df["player_name"] == selected_player
    ].sort_values("match_id")

    if player_df.empty:
        st.warning("No performances remain for this player under the current filters.")
    else:
        p1, p2, p3, p4 = st.columns(4)
        p1.metric("Matches", len(player_df))
        p2.metric("Clusters represented", player_df["cluster"].nunique())
        p3.metric("Mean minutes", f"{player_df['minutes'].mean():.1f}")
        with p4:
            st.markdown(
                f"""
                <div data-testid="stMetric">
                    <div data-testid="stMetricLabel">Most common cluster</div>
                    <div style="font-size: 1.35rem; font-weight: 600; line-height: 1.3;">
                        {player_df["cluster_name"].mode().iloc[0]}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.subheader("Cluster distribution")

        player_distribution = (
            player_df["cluster_name"]
            .value_counts()
            .rename_axis("cluster_name")
            .reset_index(name="count")
        )

        player_distribution_figure = px.pie(
            player_distribution,
            names="cluster_name",
            values="count",
            hole=0.45,
        )
        player_distribution_figure.update_layout(height=450)
        st.plotly_chart(
            player_distribution_figure,
            use_container_width=True,
        )

        st.subheader("Player-match table")

        player_table_columns = available_columns(
            [
                "match_id",
                "cluster_name",
                "player_position",
                "minutes",
                "silhouette_value",
                "distance_to_centroid",
                "passes_per90",
                "shots_per90",
                "total_pxt_per90",
                "final_third_actions_per90",
                "interceptions_per90",
                "loose_ball_regains_per90",
            ]
        )

        st.dataframe(
            player_df[player_table_columns],
            use_container_width=True,
            hide_index=True,
        )

    st.markdown("---")
    st.subheader("Download filtered data")
    st.download_button(
        label="Download filtered player-match data as CSV",
        data=csv_bytes,
        file_name="filtered_player_match_clusters.csv",
        mime="text/csv",
        key="download_players_tab",
    )
