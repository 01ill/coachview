from settings import IMPECT_DIR, PCT_BY_LINE, EXTRA_DATA_DIR
import json
import os
import pathlib
import pandas as pd
import numpy as np
from numpy import nan
import streamlit as st

EXTRA_KPI_INCLUDED = [
    "between_lines_receptions_per90",
    "wide_availability_per90",
    "box_availability_per90",
    "final_third_actions_per90",
    "opponent_box_actions_per90",
    "interceptions_per90",
    "loose_ball_regains_per90",
    "clearances_per90",
    "high_pressure_actions_per90",
]
EXTRA_KPI_LABELS = {
    "between_lines_receptions_per90": "Between Lines Receptions per 90",
    "wide_availability_per90": "Wide Availability per 90",
    "box_availability_per90": "Box Availability per 90",
    "final_third_actions_per90": "Final Third Actions per 90",
    "opponent_box_actions_per90": "Opponent Box Actions per 90",
    "interceptions_per90": "Interceptions per 90",
    "loose_ball_regains_per90": "Loose Ball Regains per 90",
    "clearances_per90": "Clearances per 90",
    "high_pressure_actions_per90": "High Pressure Actions per 90",
}


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

def load_extra_kpi() -> pd.DataFrame:
    return pd.read_csv(os.path.join(EXTRA_DATA_DIR, "player_match_features.csv"))

def load_extra_kpi_definitions() -> pd.DataFrame:
    rows = []
    for feature in EXTRA_KPI_INCLUDED:
        rows.append(
            {
                "id": f"extra_{feature}",
                "details.label": EXTRA_KPI_LABELS.get(feature, feature.replace("_", " ").title()),
                "details.definition": f"Extra player-match feature: {feature}",
            }
        )
    return pd.DataFrame(rows)

def _normalize_extra_feature_name(name: str) -> str:
    out = []
    for ch in str(name).lower():
        out.append(ch if ch.isalnum() else "_")
    normalized = "".join(out)
    while "__" in normalized:
        normalized = normalized.replace("__", "_")
    return normalized.strip("_")


def _collapse_extra_match_rows(extra_kpi: pd.DataFrame) -> pd.DataFrame:
    """
    Collapse multiple stint rows per (player_id, match_id) into one row.
    - additive/count-like columns: sum
    - avg/rate-like columns: minute-weighted mean
    - max_* columns: max
    - *_per90 columns: recomputed from summed base column when available
    """
    id_cols = {"match_id", "player_id"}
    meta_cols = {"player_name", "player_position", "position_side"}
    exclude = id_cols | meta_cols | {"first_event_minute", "last_event_minute"}

    numeric_cols = [c for c in extra_kpi.columns if c not in exclude]
    work = extra_kpi.copy()
    for c in numeric_cols:
        work[c] = pd.to_numeric(work[c], errors="coerce")

    minutes_col = "minutes"
    has_minutes = minutes_col in work.columns

    weighted_cols = [
        c for c in numeric_cols
        if (
            c.startswith("average_")
            or c.endswith("_rate")
            or c in {"average_x", "average_y", "average_distance_to_goal"}
        )
    ]
    max_cols = [c for c in numeric_cols if c.startswith("max_")]
    per90_cols = [c for c in numeric_cols if c.endswith("_per90")]

    protected = set(weighted_cols) | set(max_cols) | set(per90_cols)
    sum_cols = [c for c in numeric_cols if c not in protected]

    def _agg_one(group: pd.DataFrame) -> pd.Series:
        out = {}

        # sum-like metrics
        for c in sum_cols:
            out[c] = group[c].sum(min_count=1)

        # max-like metrics
        for c in max_cols:
            out[c] = group[c].max()

        # weighted averages / rates
        for c in weighted_cols:
            if has_minutes and group[minutes_col].notna().any() and group[minutes_col].sum() > 0:
                out[c] = np.average(
                    group[c].fillna(0.0),
                    weights=group[minutes_col].fillna(0.0),
                )
            else:
                out[c] = group[c].mean()

        # per90: recompute from summed base where possible, else weighted mean
        for c in per90_cols:
            base = c[:-6]  # strip "_per90"
            if base in out and has_minutes and pd.notna(out.get(minutes_col)) and out[minutes_col] and out[minutes_col] > 0:
                out[c] = (out[base] / out[minutes_col]) * 90.0
            elif has_minutes and group[minutes_col].notna().any() and group[minutes_col].sum() > 0:
                out[c] = np.average(
                    group[c].fillna(0.0),
                    weights=group[minutes_col].fillna(0.0),
                )
            else:
                out[c] = group[c].mean()

        return pd.Series(out)

    collapsed = (
        work.groupby(["player_id", "match_id"], as_index=False)
        .apply(_agg_one, include_groups=False)
        .reset_index()
        .drop(columns=["level_2"], errors="ignore")
    )

    return collapsed


def merge_extra_kpi(
    playerStats: pd.DataFrame,
    matches: pd.DataFrame,
    extra_kpi: pd.DataFrame | None = None,
) -> pd.DataFrame:
    if extra_kpi is None:
        extra_kpi = load_extra_kpi()

    required = {"match_id", "player_id"}
    if not required.issubset(extra_kpi.columns):
        return playerStats

    extra = _collapse_extra_match_rows(extra_kpi)

    feature_blacklist = {"match_id", "player_id"}
    feature_cols = [c for c in extra.columns if c not in feature_blacklist]
    if not feature_cols:
        return playerStats

    ordered_match_ids = matches["id"].dropna().astype(int).tolist()
    if not ordered_match_ids:
        ordered_match_ids = sorted(extra["match_id"].dropna().astype(int).unique().tolist())

    base_index = pd.Index(playerStats["playerId"].astype(int).unique(), name="playerId")
    out_cols: dict[str, pd.Series] = {}

    for feature in feature_cols:
        kpi_name = f"kpi_{_normalize_extra_feature_name(feature)}"
        pivot = extra.pivot_table(index="player_id", columns="match_id", values=feature, aggfunc="mean")
        pivot = pivot.reindex(columns=ordered_match_ids)

        out_cols[kpi_name] = pivot.apply(
            lambda row: [float(v) if pd.notnull(v) else 0.0 for v in row.tolist()],
            axis=1,
        )
        out_cols[f"{kpi_name}_avg"] = pivot.mean(axis=1, skipna=True)

    out = pd.DataFrame(out_cols).reindex(base_index)

    avg_cols = [c for c in out.columns if c.endswith("_avg")]
    pct = out[avg_cols].rank(pct=True) * 100
    pct.columns = [f"{c[:-4]}_pct" for c in avg_cols]

    meta = (
        playerStats[["playerId", "playerLine", "playerPosition"]]
        .drop_duplicates("playerId")
        .set_index("playerId")
    )
    out = out.join(meta, how="left")

    group_col = "playerLine" if PCT_BY_LINE else "playerPosition"
    pct_pos = out.groupby(group_col)[avg_cols].rank(pct=True) * 100
    pct_pos.columns = [f"{c[:-4]}_pct_pos" for c in avg_cols]

    out = out.join(pct).join(pct_pos).drop(columns=["playerLine", "playerPosition"])

    merged = playerStats.join(out, on="playerId", how="left")

    raw_cols = [
        c for c in merged.columns
        if c.startswith("kpi_") and not c.endswith(("_avg", "_pct", "_pct_pos"))
    ]
    n_matches = len(ordered_match_ids)
    for c in raw_cols:
        merged[c] = merged[c].apply(lambda v: v if isinstance(v, list) else [0.0] * n_matches)

    stat_cols = [c for c in merged.columns if c.endswith(("_avg", "_pct", "_pct_pos"))]
    merged[stat_cols] = merged[stat_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)

    return merged

EXTRA_KPI_START_ID = 10000

EXTRA_KPI_ID_TO_FEATURE = {
    EXTRA_KPI_START_ID + i: feature
    for i, feature in enumerate(EXTRA_KPI_INCLUDED)
}

EXTRA_KPI_FEATURE_TO_ID = {
    feature: kpi_id
    for kpi_id, feature in EXTRA_KPI_ID_TO_FEATURE.items()
}

def build_kpi_catalog() -> pd.DataFrame:
    core = (
        load_kpi_def()[["id", "details.label", "details.definition"]]
        .rename(columns={"details.label": "label", "details.definition": "definition"})
        .copy()
    )
    core["source"] = "core"
    core["feature_name"] = None
    core["is_extra"] = False
    core["raw_col"] = core["id"].map(lambda k: f"kpi90_{int(k)}")
    core["avg_col"] = core["id"].map(lambda k: f"avg_kpi90_{int(k)}")
    core["pct_col"] = core["id"].map(lambda k: f"avg_kpi90_{int(k)}_pct")
    core["pct_pos_col"] = core["id"].map(lambda k: f"avg_kpi90_{int(k)}_pct_pos")
    core["select_col"] = core["pct_col"]

    extra = pd.DataFrame(
        {
            "id": list(EXTRA_KPI_ID_TO_FEATURE.keys()),
            "feature_name": list(EXTRA_KPI_ID_TO_FEATURE.values()),
        }
    )
    extra["label"] = extra["feature_name"].map(
        lambda name: EXTRA_KPI_LABELS.get(name, name.replace("_", " ").title())
    )
    extra["definition"] = extra["feature_name"].map(
        lambda name: f"Extra player-match feature: {name}"
    )
    extra["source"] = "extra"
    extra["is_extra"] = True
    extra["raw_col"] = extra["feature_name"].map(
        lambda name: f"kpi_{_normalize_extra_feature_name(name)}"
    )
    extra["avg_col"] = extra["raw_col"] + "_avg"
    extra["pct_col"] = extra["raw_col"] + "_pct"
    extra["pct_pos_col"] = extra["raw_col"] + "_pct_pos"
    extra["select_col"] = extra["pct_col"]

    catalog = pd.concat([core, extra], ignore_index=True)
    catalog["picker_label"] = catalog["label"] + " - " + catalog["definition"]
    return catalog

def selected_kpi_columns(selected_kpis: pd.DataFrame) -> tuple[list[str], list[str]]:
    value_cols = selected_kpis["select_col"].tolist()
    labels = selected_kpis.set_index("select_col")["label"].to_dict()
    return value_cols, labels