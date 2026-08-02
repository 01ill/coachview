from mplsoccer import Pitch
import matplotlib.pyplot as plt
from mplsoccer import VerticalPitch
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


def plot_xg_timeline(shots, home_team="SC Freiburg"):
    #Used LLM to fix and improve the function
    # Remove shots without an xG value
    shots = shots.dropna(subset=["xg"])

    # Positive xG for home team, negative for away team
    shots["plot_xg"] = shots["xg"]
    shots.loc[shots["team_name"] != home_team, "plot_xg"] *= -1

    result_colors = {
        "GOAL": "gold",
        "SAVED": "royalblue",
        "OFF_TARGET": "red",
    }

    fig, ax = plt.subplots(figsize=(14, 6))

    for _, shot in shots.iterrows():

        color = result_colors.get(shot["result"], "gray")

        # Stem
        ax.vlines(
            shot["match_minute"],
            0,
            shot["plot_xg"],
            color=color,
            linewidth=2,
            alpha=0.8,
        )

        # Shot marker
        ax.scatter(
            shot["match_minute"],
            shot["plot_xg"],
            s=90,
            color=color,
            edgecolor="black",
            linewidth=0.8,
            zorder=3,
        )

    # Zero line
    ax.axhline(0, color="black", linewidth=1)

    # Axis labels
    ax.set_xlabel("Match Minute")
    ax.set_ylabel("Expected Goals (xG)")

    # Team labels
    max_xg = shots["xg"].max()

    away_team = (
        shots.loc[shots["team_name"] != home_team, "team_name"]
        .iloc[0]
    )

    ax.text(
        1,
        max_xg * 1.08,
        home_team,
        fontsize=12,
        fontweight="bold",
    )

    ax.text(
        1,
        -max_xg * 1.15,
        away_team,
        fontsize=12,
        fontweight="bold",
    )

    # Legend
    legend_elements = [
        Line2D(
            [0], [0],
            marker='o',
            color='w',
            label=result,
            markerfacecolor=color,
            markeredgecolor='black',
            markersize=8,
        )
        for result, color in result_colors.items()
    ]

    ax.legend(
        handles=legend_elements,
        title="Shot Result",
        loc="upper right",
    )

    # Clean up
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.set_ylim(
        -max_xg * 1.3,
        max_xg * 1.3,
    )

    plt.tight_layout()

    return fig

def plot_team_shots(df, team_name):

    shots = df[
        (df["event_type"] == "SHOT") &
        (df["team_name"] == team_name)
    ]

    result_styles = {
        "GOAL": {
            "color": "gold",
            "label": "Goal"
        },
        "SAVED": {
            "color": "royalblue",
            "label": "Saved"
        },
        "OFF_TARGET": {
            "color": "red",
            "label": "Off Target"
        }
    }

    pitch = VerticalPitch(line_color="black", half=True)

    fig, ax = pitch.grid(
        grid_height=0.9,
        title_height=0.06,
        axis=False,
        endnote_height=0.04,
        title_space=0,
        endnote_space=0,
    )

    for result, style in result_styles.items():

        subset = shots[shots["result"] == result]

        if subset.empty:
            continue

        pitch.scatter(
            subset["coordinates_x"],
            subset["coordinates_y"],
            s=500,
            color=style["color"],
            edgecolors="black",
            alpha=1,
            label=style["label"],
            ax=ax["pitch"],
        )

    ax["pitch"].set_xlim(10, 70)
    ax["pitch"].set_ylim(80, 125)

    ax["pitch"].legend(loc="lower right")

    fig.suptitle(
        f"{team_name} Shots",
        fontsize=24,
    )

    return fig



def plot_shots(df):
    all_shots = df[df["event_type"] == "SHOT"]

    pitch = Pitch(line_color="black")
    fig, ax = pitch.draw(figsize=(10, 7))

    pitchLengthX = 120
    pitchWidthY = 80

    for _, shot in all_shots.iterrows():

        x = shot["coordinates_x"]
        y = shot["coordinates_y"]

        goal = shot["result"] == "GOAL"

        if shot["team_name"] == "SC Freiburg":
            color = "red"

            if goal:
                circle = plt.Circle((x, y), 2, color=color)
            else:
                circle = plt.Circle((x, y), 2, color=color, alpha=0.2)

        else:
            color = "blue"

            if goal:
                circle = plt.Circle(
                    (pitchLengthX - x, pitchWidthY - y),
                    2,
                    color=color,
                )
            else:
                circle = plt.Circle(
                    (pitchLengthX - x, pitchWidthY - y),
                    2,
                    color=color,
                    alpha=0.2,
                )

        ax.add_patch(circle)

    teams = df["team_name"].dropna().unique().tolist()
    opponent = [t for t in teams if t != "SC Freiburg"][0]

    fig.suptitle(
        f"SC Freiburg (red) vs {opponent} (blue) shots",
        fontsize=20,
    )

    return fig