import pandas as pd
import json

def get_match_title(df):
    teams = df["team_name"].dropna().unique().tolist()

    freiburg = "SC Freiburg"
    opponent = next(team for team in teams if team != freiburg)

    return f"{freiburg} vs {opponent}"

def get_opponent(df):
    teams = df["team_name"].dropna().unique().tolist()

    freiburg = "SC Freiburg"
    opponent = next(team for team in teams if team != freiburg)

    return opponent

def get_opponent_id(df):
    freiburg = "SC Freiburg"

    opponent = (
        df.loc[df["team_name"] != freiburg, ["team_name", "team_id"]]
        .drop_duplicates()
        .iloc[0]
    )

    return opponent["team_id"]

def add_match_minutes(df):
    df = df.copy()

    period_offsets = {
        1: 0,
        2: 45,
        3: 90,
        4: 105,
    }

    df["match_minute"] = (
        df["timestamp"].dt.total_seconds() / 60
        + df["period_id"].map(period_offsets).fillna(0)
    )

    return df

def add_xg(df, match_id):

    with open(f"./open-data/data/events_kpis/events_kpis_{match_id}.json", "r") as f:
        data = json.load(f)

    kpis = pd.DataFrame(data)

    shots_xg = (kpis[kpis["kpiId"] == 82].rename(columns={"eventId": "event_id","value": "xg",}))

    df = df.copy()
    shots = (
            df[df["event_type"] == "SHOT"]
            .copy()
            .sort_values("match_minute")
        )

    shots["event_id"] = shots["event_id"].astype("int64")

    shots = shots.merge(shots_xg[["event_id", "xg"]],on="event_id",how="left")

    return shots


def get_match_score(df, team="SC Freiburg"):

    teams = df["team_name"].dropna().unique().tolist()
    opponent = next(t for t in teams if t != team)

    team_goals = ((df["event_type"] == "SHOT")& (df["team_name"] == team) & (df["result"] == "GOAL")).sum()

    opponent_goals = ((df["event_type"] == "SHOT") & (df["team_name"] == opponent) & (df["result"] == "GOAL")).sum()

    own_goals = df[(df["event_type"] == "SHOT") & (df["result"] == "OWN GOAL")]

    for _, row in own_goals.iterrows():
        if row["team_name"] == team:
            opponent_goals += 1
        else:
            team_goals += 1

    return team_goals, opponent_goals, opponent


def summary_metrics(df, opponent_name):

    freiburg_shots = df[(df["team_name"] == "SC Freiburg") & (df["event_type"] == "SHOT")]


    opponent_shots = df[(df["team_name"] == opponent_name) & (df["event_type"] == "SHOT")]

    freiburg_num_shots = len(freiburg_shots)
    opponent_num_shots = len(opponent_shots)

    freiburg_xg = freiburg_shots["xg"].sum()
    opponent_xg = opponent_shots["xg"].sum()

    freiburg_on_target = freiburg_shots["result"].isin(["GOAL", "SAVED"]).sum()
    opponent_on_target = opponent_shots["result"].isin(["GOAL", "SAVED"]).sum()

    return freiburg_num_shots, freiburg_xg, freiburg_on_target, opponent_num_shots, opponent_xg, opponent_on_target