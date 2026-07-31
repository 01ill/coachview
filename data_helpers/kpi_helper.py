from settings import IMPECT_DIR, PCT_BY_LINE
import json
import os
import pathlib
import pandas as pd
from numpy import nan
import streamlit as st


@st.cache_data
def load_event_kpi(match_id) -> pd.DataFrame:
    event_kpi_path = os.path.join(IMPECT_DIR, 'events_kpis', f'events_kpis_{match_id}.json')
    with open(event_kpi_path) as f:
        data = json.load(f)
        #save it in dataframe
    df_events_kpi = pd.json_normalize(data)
    return df_events_kpi

@st.cache_data
def load_kpi_def() -> pd.DataFrame:
    """
    Reads the definitions for the included KPIs
    """
    kpi_def_path = os.path.join(IMPECT_DIR, 'kpi_definitions.json')
    with open(kpi_def_path) as f:
        return pd.json_normalize(json.load(f)).loc[lambda df: df["details.label"].notnull()]

@st.cache_data
def load_player_kpi(match_id: list = None) -> pd.DataFrame:
    """
    Reads the player KPIs for specific match(es)
    """
    if match_id is None:
        kpi_dir = pathlib.Path(IMPECT_DIR) / "player_kpis"
        match_id = sorted(
            int(p.stem.removeprefix("player_kpis_"))
            for p in kpi_dir.glob("player_kpis_*.json")
        )
    df_kpi_all = pd.DataFrame()
    for id in match_id:
        path = os.path.join(IMPECT_DIR, 'player_kpis', f'player_kpis_{id}.json')
        with open(path) as f:
            data = json.load(f)
        df_kpi_all = pd.concat([df_kpi_all, pd.json_normalize(data)])
    return df_kpi_all

def melt_side(df, players_col: str, id_col: str, side: str) -> pd.DataFrame:
    return (
        df[["matchId", id_col, players_col]]
            .rename(columns={id_col: "squadId", players_col: "players"})
            .assign(side=side)
            .explode("players", ignore_index=True)  # make each player a row
            .pipe(lambda frame: pd.concat([frame.drop(columns="players"), pd.json_normalize(frame["players"])], axis=1))  # normalize player data to columns
            .explode("kpis").reset_index(drop=True)  # each player+kpi becomes a row
            .pipe(lambda frame: pd.concat([frame.drop(columns="kpis"), pd.json_normalize(frame["kpis"])], axis=1))
    )

def merge_match_stints(df: pd.DataFrame) -> pd.DataFrame:
    """
    we might have multiple entries for players in the kpi files
    there is one entry for each player on each position
    so if the position changes, there will be a new entry
    """
    return (
        df.groupby(["id", "matchId", "kpiId"], sort=False, as_index=False)
            .agg(  # aggregate each entry
                matchday=("matchday", "first"),
                squadId=("squadId", "first"),
                side=("side", "first"),
                position=("position", lambda s: s.mode().iat[0] if not s.mode().empty else s.iat[0]),
                playDuration=("playDuration", "sum"),
                value=("value", "sum")
            )
    )

def kpi_long_df(matches, kpi_all=None):
    if kpi_all is None:
        kpi_all = load_player_kpi(matches.id)
    matchday_map = matches.set_index("id")["matchDay.index"]
    df = pd.concat([
        melt_side(kpi_all, "squadHome.players", "squadHome.id", "home"),
        melt_side(kpi_all, "squadAway.players", "squadAway.id", "away")
    ], ignore_index=True)
    df = df.assign(matchday=df["matchId"].map(matchday_map)).pipe(merge_match_stints)
    kpi_ids = df["kpiId"].dropna().unique()
    played_matches = df.sort_values("playDuration", ascending=False).drop_duplicates(subset=["id", "matchId"])[["id", "matchId", "matchday", "position", "squadId", "side", "playDuration"]]
    grid = played_matches.merge(pd.DataFrame({"kpiId": kpi_ids}), how="cross")
    df = grid.merge(df[["id", "matchId", "kpiId", "value"]], on=["id", "matchId", "kpiId"], how="left")
    df["value"] = df["value"].fillna(0)
    df["value90"] = df["value"] * (5400 / df["playDuration"])
    return df

def kpi_statistics(df: pd.DataFrame):
    """
    Calculate the max/min/percentiles so that we can display it correctly in graphs
    and normalise accordingly
    The DataFrame resulting from kpi_long_df() has to be passed
    """
    # we have the exploded view where each player in each match has one entry for each kpi
    # group by kpiId. Then we can easily use min/max
    # rename to keep parity with player df (i.e. not use just id but kpi_id)
    kpi_range = df.groupby("kpiId")["value90"].agg(["min", "max"]).rename(index=lambda k: f"kpi90_{int(k)}")
    return kpi_range

def kpi_percentiles(playerStats: pd.DataFrame) -> pd.DataFrame:
    avg90_cols = [c for c in playerStats.columns if c.startswith("avg_kpi90_")]
    percentiles = playerStats[avg90_cols].rank(pct=True) * 100
    percentiles.columns = [f"{c}_pct" for c in avg90_cols]
    if PCT_BY_LINE:
        percentiles_position = playerStats.groupby("playerLine")[avg90_cols].rank(pct=True) * 100
    else:
        percentiles_position = playerStats.groupby("playerPosition")[avg90_cols].rank(pct=True) * 100
    percentiles_position.columns = [f"{c}_pct_pos" for c in avg90_cols]
    playerStats = pd.concat([playerStats, percentiles, percentiles_position], axis=1)
    return playerStats

def _fix_nan(cell, count: int):
    return cell if isinstance(cell, list) else [nan] * count

def get_player_kpi(matches: pd.DataFrame, df=None, kpi_all=None):
    if kpi_all is None:
        kpi_all = load_player_kpi(matches.id)
    if df is None:
        df = kpi_long_df(matches, kpi_all)
    # show for each matchday all kpis of the player
    pivot = df.pivot_table(index=["id", "kpiId"], columns="matchday", values="value", aggfunc="first")  # value is value of kpi
    pivot = pivot.apply(list, axis=1)  # convert all columns to list -> each row: playerId, kpiId, values
    pivot = pivot.unstack("kpiId")  # now each kpi has own column
    pivot.columns = [f"kpi_{int(c)}" for c in pivot.columns]  # rename so each kpiId has kip_ in front
    pivot = pivot.apply(lambda col: col.apply(lambda cell: _fix_nan(cell, len(matches))))
    kpi_pivot = df.pivot_table(index="id", columns="kpiId", values="value", aggfunc="mean")  # calculate average for each kpi for each player
    kpi_pivot.columns = [f"avg_kpi_{int(c)}" for c in kpi_pivot.columns]
    
    pivot90 = df.pivot_table(index=["id", "kpiId"], columns="matchday", values="value90", aggfunc="first")  # value is value of kpi
    pivot90 = pivot90.apply(list, axis=1)  # convert all columns to list -> each row: playerId, kpiId, values
    pivot90 = pivot90.unstack("kpiId")  # now each kpi has own column
    pivot90.columns = [f"kpi90_{int(c)}" for c in pivot90.columns]  # rename so each kpiId has kip_ in front
    pivot90 = pivot90.apply(lambda col: col.apply(lambda cell: _fix_nan(cell, len(matches))))
    kpi_pivot90 = df.pivot_table(index="id", columns="kpiId", values="value90", aggfunc="mean")  # calculate average for each kpi for each player
    kpi_pivot90.columns = [f"avg_kpi90_{int(c)}" for c in kpi_pivot90.columns]
    pivot = (pivot.join(kpi_pivot, on="id")
        .join(pivot90, on="id")
        .join(kpi_pivot90, on="id")
        .reset_index()
        .rename(columns={"id": "playerId"}))
    return pivot
