from mplsoccer import Pitch
import matplotlib.pyplot as plt
import streamlit as st
from matplotlib.lines import Line2D
#Used LLM 
ACTION_STYLE = {
    "PASS": {
        "color": "royalblue",
        "width": 2,
        "size": 40,
    },
    "DRIBBLE": {
        "color": "forestgreen",
        "width": 2,
        "size": 40,
    },
    "SHOT": {
        "color": "red",
        "width": 3,
        "size": 80,
    },
    "GOAL": {
        "color": "gold",
        "width": 3,
        "size": 100,
    },
}

def plot_possession_chain(possession):
    pitch = Pitch(
        pitch_type="statsbomb",
        pitch_color="white",
        line_color="black"
    )
    
    fig, ax = pitch.draw(figsize=(10, 7))

    for event in possession["events"]:

        action = event["actionType"]

        if action not in ACTION_STYLE:
            continue

        start = event.get("start")
        end = event.get("end")

        if start is None or end is None:
            continue

        start = start.get("coordinates")   # or adjCoordinates if you switch back
        end = end.get("coordinates")

        if start is None or end is None:
            continue

        x1 = 60 - start["x"]      # attack left -> right
        y1 = start["y"] + 40

        x2 = 60 - end["x"]
        y2 = end["y"] + 40

        style = ACTION_STYLE[action].copy()

        if (action == "SHOT" and event["id"] == possession["goal_shot_id"]):
            style["color"] = "gold"
            style["size"] = 100

        pitch.arrows(
            x1,
            y1,
            x2,
            y2,
            ax=ax,
            color=style["color"],
            width=style["width"],
            headwidth=4,
            headlength=5,
            zorder=2,
        )

        pitch.scatter(
            x1,
            y1,
            ax=ax,
            s=style["size"],
            color=style["color"],
            edgecolors="black",
            linewidth=1,
            zorder=3,
        )
    
    legend = [
        Line2D([0], [0], color="royalblue", lw=2, label="Pass"),
        Line2D([0], [0], color="forestgreen", lw=2, label="Dribble"),
        Line2D([0], [0], color="red", lw=2, label="Shot"),
    ]
    if possession["goal"]:
        st.success("⚽ Goal")
    else:
        st.info("❌ No goal")
    ax.legend(handles=legend, loc="upper left")
    return fig