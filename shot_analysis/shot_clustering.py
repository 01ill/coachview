from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import pandas as pd

from .loader import load_all_shots


#Through the elbow method, I found the optimal k to be 5, the classification is based on the coordinates of the shot
def cluster_shots(squad_id, n_clusters=5):

    cluster_df = load_all_shots(squad_id)

    cluster_df = cluster_df.dropna(
        subset=["coordinates_x", "coordinates_y"]
    ).copy()

    X = cluster_df[
        ["coordinates_x", "coordinates_y"]
    ].values

    scaler = StandardScaler()

    X_scaled = scaler.fit_transform(X)

    model = KMeans(
        n_clusters=n_clusters,
        random_state=2147,
        n_init="auto"
    )

    cluster_df["cluster"] = model.fit_predict(X_scaled)

    return cluster_df, model, scaler

def assign_profile_name(x, y):

    if x >= 108:
        depth = "Close Range"
    elif x >= 100:
        depth = "Penalty Area"
    else:
        depth = "Edge of Box"

    if y < 30:
        side = "Left"
    elif y < 40:
        side = "Left-Central"
    elif y < 50:
        side = "Right-Central"
    else:
        side = "Right"

    return f"{depth} - {side}"


def compute_cluster_statistics(cluster_df, model, scaler):
#used LLM
    stats = (
        cluster_df
        .groupby("cluster")
        .agg(
            shots=("cluster", "size"),
            goals=("result", lambda x: (x == "GOAL").sum()),
            saved=("result", lambda x: (x == "SAVED").sum()),
            off_target=("result", lambda x: (x == "OFF_TARGET").sum()),
            average_xg=("xg", "mean"),
        )
        .reset_index()
    )

    stats["shot_percentage"] = (
        stats["shots"] / len(cluster_df) * 100
    )

    stats["goal_percentage"] = (
        stats["goals"] / stats["shots"] * 100
    )

    stats["saved_percentage"] = (
        stats["saved"] / stats["shots"] * 100
    )

    stats["off_target_percentage"] = (
        stats["off_target"] / stats["shots"] * 100
    )

    centroids = scaler.inverse_transform(model.cluster_centers_)

    centroid_df = pd.DataFrame(
        centroids,
        columns=["centroid_x", "centroid_y"]
    )

    centroid_df["cluster"] = centroid_df.index

    stats = stats.merge(
        centroid_df,
        on="cluster",
        how="left"
    )

    stats["profile"] = stats.apply(
    lambda row: assign_profile_name(
        row["centroid_x"],
        row["centroid_y"]
    ),
    axis=1
    )

    stats = stats.round({
    "centroid_x": 1,
    "centroid_y": 1,
    "shot_percentage": 1,
    "average_xg": 3,
    "goal_percentage": 1,
    "saved_percentage": 1,
    "off_target_percentage": 1,
    })

    return stats

