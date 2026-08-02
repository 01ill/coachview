from kloppy import impect
from kloppy.domain import Provider
from pathlib import Path
from .utils import add_match_minutes, add_xg
import pandas as pd
import json
import streamlit as st

EVENTS_DIR = Path("open-data/data/events")

TEAM_NAMES = {
    27: "FC Köln",
    29: "Borussia Dortmund",
    30: "VfL Wolfsburg",
    31: "TSG 1899 Hoffenheim",
    32: "Borussia Mönchengladbach",
    33: "FC Bayern München",
    34: "SC Freiburg",
    35: "Eintracht Frankfurt",
    36: "SV Darmstadt 98",
    37: "RasenBallsport Leipzig",
    38: "SV Werder Bremen",
    39: "FC Augsburg",
    41: "Bayer 04 Leverkusen",
    42: "FSV Mainz 05",
    46: "VfB Stuttgart",
    416: "VfL Bochum",
    432: "FC Heidenheim 1846",
    446: "FC Union Berlin",
}


def load_match(match_id):

    event_json_path = EVENTS_DIR / f"events_{match_id}.json"

    with open(event_json_path, "r") as f:
        raw_events = json.load(f)

    event_dataset = (
        impect.load_open_data(match_id=match_id)
        .transform(to_coordinate_system=Provider.STATSBOMB)
    )

    metadata = event_dataset.metadata

    df = event_dataset.to_df(engine="pandas")

    df["team_name"] = df["team_id"].map(
        lambda x: TEAM_NAMES.get(int(x)) if pd.notna(x) else None
    )

    df = df.dropna(subset=["coordinates_x", "coordinates_y"], how="all")

    df = add_match_minutes(df)
    df = add_xg(df, match_id)

    return event_dataset, metadata, df, raw_events

@st.cache_data
def get_match_ids(squad_id):

    match_ids = []

    for file in EVENTS_DIR.glob("events_*.json"):

        match_id = int(file.stem.split("_")[1])

        event_dataset = (
            impect.load_open_data(match_id=match_id)
            .transform(to_coordinate_system=Provider.STATSBOMB)
        )

        df = event_dataset.to_df(engine="pandas")

        if str(squad_id) in df["team_id"].unique():
            match_ids.append(match_id)

    return sorted(match_ids)

@st.cache_data
def load_all_shots(squad_id):

    match_ids = get_match_ids(squad_id)
    all_shots = []

    for match_id in match_ids:

        _, _, df, _ = load_match(match_id)

        shots = df[(df["team_id"] == str(squad_id)) & (df["event_type"] == "SHOT")].copy()

        shots["match_id"] = match_id
        all_shots.append(shots)

    return pd.concat(all_shots, ignore_index=True)
