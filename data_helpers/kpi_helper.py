from settings import IMPECT_DIR
import json
import os
import pandas as pd

def load_event_kpi(match_id=None) -> pd.DataFrame:
    event_kpi_path = os.path.join(IMPECT_DIR, 'events_kpis', f'events_kpis_{match_id}.json')
    with open(event_kpi_path) as f:
        data = json.load(f)
        #save it in dataframe
    df_events_kpi = pd.json_normalize(data)
    return df_events_kpi


def load_kpi_def() -> pd.DataFrame:
    """
    Reads the definitions for the included KPIs
    """
    kpi_def_path = os.path.join(IMPECT_DIR, 'kpi_definitions.json')
    with open(kpi_def_path) as f:
        data = json.load(f)
        #save it in dataframe
    df_kpi_def = pd.json_normalize(data)
    df_kpi_def = df_kpi_def[df_kpi_def["details.label"].notnull()]
    return df_kpi_def

def load_player_kpi(match_id=None) -> pd.DataFrame:
    """
    Reads the player KPIs for specific match(es)
    """
    df_kpi_all = pd.DataFrame()
    for id in match_id:
        path = os.path.join(IMPECT_DIR, 'player_kpis', f'player_kpis_{id}.json')
        with open(path) as f:
            data = json.load(f)
        df_kpi_all = pd.concat([df_kpi_all, pd.json_normalize(data)])
    return df_kpi_all

def melt_side(df, players_col: str, id_col: str, side: str) -> pd.DataFrame:
    d = df[["matchId", id_col, players_col]].rename(columns={id_col: "squadId", players_col: "players"})
    d["side"] = side
    d = d.explode("players").reset_index(drop=True)
    d = pd.concat([d.drop(columns="players"), pd.json_normalize(d["players"])], axis=1)
    d = d.explode("kpis").reset_index(drop=True)
    return pd.concat([d.drop(columns="kpis"), pd.json_normalize(d["kpis"])], axis=1)

def kpi_long_df(matches, kpi_all=None):
    if kpi_all is None:
        kpi_all = load_player_kpi(matches.id)
    matchday_map = matches.set_index("id")["matchDay.index"]
    df = pd.concat([
        melt_side(kpi_all, "squadHome.players", "squadHome.id", "home"),
        melt_side(kpi_all, "squadAway.players", "squadAway.id", "away")
    ], ignore_index=True)
    df["matchday"] = df["matchId"].map(matchday_map)
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
    kpi_range = df.groupby("kpiId")["value"].agg(["min", "max"]).rename(index=lambda k: f"kpi_{int(k)}")
    return kpi_range

def get_player_kpi(matches: pd.DataFrame, df=None, kpi_all=None):
    if kpi_all is None:
        kpi_all = load_player_kpi(matches.id)
    if df is None:
        df = kpi_long_df(matches, kpi_all)
    all_matchdays = sorted(matches["matchDay.index"].unique())  # just a list of all matchday indices (from 0-33)
    # show for each matchday all kpis of the player
    pivot = df.pivot_table(index=["id", "kpiId"], columns="matchday", values="value", aggfunc="first")#.reindex(columns=all_matchdays)  # value is value of kpi
    pivot = pivot.apply(list, axis=1)  # convert all columns to list -> each row: playerId, kpiId, values
    pivot = pivot.unstack("kpiId")  # now each kpi has own column
    pivot.columns = [f"kpi_{int(c)}" for c in pivot.columns]  # rename so each kpiId has kip_ in front
    kpi_pivot = df.pivot_table(index="id", columns="kpiId", values="value", aggfunc="mean")  # calculate average for each kpi for each player
    kpi_pivot.columns = [f"avg_kpi_{int(c)}" for c in kpi_pivot.columns]
    pivot = pivot.join(kpi_pivot, on="id").reset_index().rename(columns={"id": "playerId"})
    return pivot

def generate_radar_df(df: pd.DataFrame, kpi_list: list):
    pass