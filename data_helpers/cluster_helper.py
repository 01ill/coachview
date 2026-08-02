import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.pipeline import Pipeline
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from settings import RANDOM_STATE
import plotly.express as px
import plotly.graph_objects as go

LINE_MARKERS = {"GK": "s", "DEF": "o", "MID": "^", "FWD": "D"}
# cluster method changes only the assignment to a cluster
# the visualization depends on PCA/TSNE
CLUSTER_METHOD = AgglomerativeClustering

def prepare_dataset(players_df, squad_ids, feature_cols, exclude_positions=None):
    cluster_df = players_df[players_df["playerId"].isin(squad_ids)].copy()
    if exclude_positions:
        cluster_df = cluster_df[~cluster_df["playerPosition"].isin(exclude_positions)]
    cluster_df = cluster_df.dropna(subset=feature_cols, thresh=max(1, len(feature_cols) // 2)).reset_index(drop=True)

    X = cluster_df[feature_cols].apply(pd.to_numeric, errors="coerce")
    X = X.fillna(X.mean())
    X = StandardScaler().fit_transform(X)
    return cluster_df, X

def cluster_silhuette(X, k_min=2, k_max=8):
    """
    Find the best K for clustering
    https://scikit-learn.org/stable/auto_examples/cluster/plot_kmeans_silhouette_analysis.html
    """
    k_max = min(k_max, len(X) - 1)
    silhuette_avg = {}
    for n_clusters in range(k_min, k_max + 1):
        args = {"n_clusters": n_clusters}
        if CLUSTER_METHOD is KMeans:
            args["random_state"] = RANDOM_STATE
        cluster_labels = CLUSTER_METHOD(**args).fit_predict(X)
        silhuette_avg[n_clusters] = silhouette_score(X, cluster_labels)
    return max(silhuette_avg, key=silhuette_avg.get), silhuette_avg

def cluster_squad(cluster_df, X, n_clusters):
    # https://scikit-learn.org/stable/auto_examples/cluster/plot_kmeans_digits.html
    args = {"n_clusters": min(n_clusters, len(cluster_df) - 1)}
    pca = PCA(n_components=2, random_state=RANDOM_STATE)
    # like in the soccermatics course. But leads to weird results so just use PCA
    # tsne = TSNE(random_state=RANDOM_STATE, perplexity=3)
    # pipeline = Pipeline([('pca', pca), ('tsne', tsne)])
    comps = pca.fit_transform(X)
    if CLUSTER_METHOD is KMeans:
        args["random_state"] = RANDOM_STATE
        # args["init"] = pca.components_
        args["n_init"] = 10
    cluster = CLUSTER_METHOD(**args)
    labels = cluster.fit_predict(comps)
    return cluster_df.assign(cluster=labels.astype(str), pca_x=comps[:, 0], pca_y=comps[:, 1])

def map_kpi_labels(avg_col, label_map):
    """
    Replace the KPI ID with the correct label
    LLM used
    """
    base = avg_col[len("avg_"):] if avg_col.startswith("avg_") else avg_col
    base = base[:-len("_pct_pos")] if base.endswith("_pct_pos") else base
    base = base[:-len("_pct")] if base.endswith("_pct") else base
    return label_map.get(base, base)

def label_clusters(cluster_df, feature_cols, label_map, top_n=1):
    """
    LLM used
    """
    overall_mean = cluster_df[feature_cols].mean()  # calculate mean or each selected kpi
    labels = {}
    for cid, group in cluster_df.groupby("cluster"):  # go through all clusters
        cluster_mean = group[feature_cols].mean()  # calculate mean for all kpis of this cluster
        # find strength of this cluster (diff stores difference for each kpi)
        diff = (cluster_mean - overall_mean).sort_values(ascending=False)
        # top_n determines how many kpis are included
        labels[cid] = ", ".join(map_kpi_labels(c, label_map) for c in diff.head(top_n).index)
    return labels

def bench_options(cluster_df, X, target_player_id, bench_only_ids=None, top_n=3):
    target_idx = cluster_df.index[cluster_df["playerId"] == target_player_id]
    if len(target_idx) == 0:
        return pd.DataFrame()
    target_idx = target_idx[0]
    target_cluster = cluster_df.loc[target_idx, "cluster"]
    target_line = cluster_df.loc[target_idx, "playerLine"]
    target_vec = X[target_idx]
    options = cluster_df[(cluster_df["playerId"] != target_player_id)].copy()
    if bench_only_ids is not None:
        options = options[options["playerId"].isin(bench_only_ids)]
    options["distance"] = options.index.map(lambda i: round(np.linalg.norm(X[i] - target_vec), 2))
    options["sameLine"] = options["playerLine"] == target_line
    options["sameCluster"] = options["cluster"] == target_cluster
    same = options[options["sameLine"] | options["sameCluster"]]
    remainder = options[~options.index.isin(same.index)].sort_values("distance")
    # filter out players with bad distance
    if not same.empty:
        distance_mean = same["distance"].mean()
        remainder = remainder[remainder["distance"] < distance_mean]
    lucky_loser = remainder.head(top_n)
    return pd.concat([same, lucky_loser]).sort_values("distance")

# def plot_cluster(cluster_df, cluster_labels, highlight_player_id=None, bench_options=None, n_closest=3):
#     fig, ax = plt.subplots()
#     palette = plt.cm.tab10.colors
#     unique_clusters = sorted(cluster_df["cluster"].unique(), key=int)
#     color_map = {cid: palette[i % len(palette)] for i, cid in enumerate(unique_clusters)}

#     for line, marker in LINE_MARKERS.items():
#         sub = cluster_df[cluster_df["playerLine"] == line]
#         if sub.empty:
#             continue
#         ax.scatter(sub["pca_x"], sub["pca_y"], c=sub["cluster"].map(color_map),
#                    marker=marker, s=100, edgecolors="black", linewidths=0.6)

#     if highlight_player_id is not None:
#         row = cluster_df.loc[cluster_df["playerId"] == highlight_player_id]
#         if not row.empty:
#             ax.scatter(row["pca_x"], row["pca_y"], facecolors="none", edgecolors="black",
#                         s=400, linewidths=2.5, zorder=5)
#             ax.annotate(row["playerName"].iloc[0], (row["pca_x"].iloc[0], row["pca_y"].iloc[0]),
#                         textcoords="offset points", xytext=(10, 6), fontweight="bold")

#     if bench_options is not None and not bench_options.empty:
#         best_options = bench_options.head(n_closest)
#         labeled_rows = cluster_df[cluster_df["playerId"].isin(best_options["playerId"])]

#         for _, player in labeled_rows.iterrows():
#             ax.annotate(player["playerName"], (player["pca_x"], player["pca_y"]),
#                         textcoords="offset points", xytext=(8, 6), fontsize=8, zorder=6)

#     handles = [mlines.Line2D([], [], color=color_map[cid], marker="o", linestyle="",
#                              markersize=9, label=cluster_labels[cid]) for cid in unique_clusters]
#     ax.legend(handles=handles, title="Archetype", loc="upper left", bbox_to_anchor=(1.02, 1), fontsize=8)

#     #ax.set_xlabel(x_label)
#     #ax.set_ylabel(y_label)
#     ax.set_title("Squad Clustering")
#     fig.tight_layout()
#     return fig


def plot_cluster_plotly(cluster_df, cluster_labels, highlight_player_id=None, bench_options=None, n_closest=3, only_benched=False):
    """
    LLM used for translating the pyplot figure to plotly (to have hover)
    """
    plot_df = cluster_df.assign(archetype=cluster_df["cluster"].map(cluster_labels))
    if only_benched:
        plot_df = plot_df[(~plot_df["isStarter"]) | (plot_df["playerId"] == highlight_player_id)]
    fig = px.scatter(
        plot_df,
        x="pca_x", y="pca_y",
        color="archetype",
        symbol="playerLine",
        hover_name="playerName",
        hover_data={"playerPosition": True, "score": ":.1f", "pca_x": False, "pca_y": False},
        labels={"pca_x": "PCA1", "pca_y": "PCA2"}
    )
    fig.update_traces(marker=dict(size=12, line=dict(width=1, color="black")))

    if highlight_player_id is not None:
        row = cluster_df.loc[cluster_df["playerId"] == highlight_player_id]
        if not row.empty:
            fig.add_trace(go.Scatter(
                x=row["pca_x"], y=row["pca_y"], mode="markers+text",
                marker=dict(size=22, color="rgba(0,0,0,0)", line=dict(width=3, color="black")),
                text=row["playerName"], textposition="top center",
                textfont=dict(size=13, color="black"),
                showlegend=False, hoverinfo="skip"
            ))
    if bench_options is not None and not bench_options.empty:
        best_options = bench_options.head(n_closest)
        labeled_rows = (cluster_df[cluster_df["playerId"].isin(best_options["playerId"])])

        fig.add_trace(go.Scatter(
            x=labeled_rows["pca_x"], y=labeled_rows["pca_y"], mode="markers", name="Recommended Substitutes",
            marker=dict(size=20, color="rgba(0,0,0,0)", line=dict(width=2.5, color="crimson", dash="dot")),
            hoverinfo="skip",
        ))

    fig.update_layout(title="Squad Clustering", legend_title="Archetype")
    #fig.tight_layout()
    return fig