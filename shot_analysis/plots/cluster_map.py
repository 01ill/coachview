from mplsoccer import Pitch
import matplotlib.pyplot as plt


def plot_cluster_map(cluster_df, cluster_stats, model, scaler):
#partial LLM use
    pitch = Pitch(
        pitch_type="statsbomb",
        pitch_color="white",
        line_color="black"
    )

    fig, ax = pitch.draw(figsize=(10, 7))

    # centroid coordinates
    centroids = scaler.inverse_transform(model.cluster_centers_)

    cmap = plt.get_cmap("tab10")

    for cluster in sorted(cluster_df["cluster"].unique()):

        shots = cluster_df[cluster_df["cluster"] == cluster]

        color = cmap(cluster)

        pitch.scatter(
            shots["coordinates_x"],
            shots["coordinates_y"],
            color=color,
            s=45,
            edgecolors="black",
            linewidth=0.4,
            alpha=0.75,
            ax=ax,
        )

    # draw centroids
    pitch.scatter(
        centroids[:, 0],
        centroids[:, 1],
        marker="X",
        s=250,
        color="black",
        ax=ax,
        zorder=5,
    )

    for i, (x, y) in enumerate(centroids):

        ax.text(
            x,
            y + 2,
            f"P{i+1}",
            fontsize=11,
            fontweight="bold",
            ha="center",
            color="black",
        )

    legend = []

    for _, row in cluster_stats.sort_values("cluster").iterrows():

        legend.append(
            plt.Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                markerfacecolor=cmap(int(row.cluster)),
                markeredgecolor="black",
                markersize=10,
                label=(
                    f"P{int(row.cluster)+1} | "
                    f"{row.profile}\n"
                    f"xG {row.average_xg:.2f} | "
                    f"Goals {row.goal_percentage:.1f}% | "
                    f"Shots {row.shot_percentage:.1f}%"
                ),
            )
        )

    ax.legend(
        handles=legend,
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        frameon=False,
        fontsize=9,
        labelspacing=1.5,
    )

    plt.tight_layout()

    return fig