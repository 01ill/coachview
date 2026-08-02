from collections import defaultdict
import streamlit as st
from collections import Counter

def build_possessions(raw_events):
    possessions = defaultdict(list)

    for event in raw_events:
        possessions[event["sequenceIndex"]].append(event)

    return possessions


def get_shots(raw_events):
    return [
        event for event in raw_events
        if event["actionType"]=="SHOT"
    ]


def build_xg_lookup(df):
    return (
        df.set_index("event_id")["xg"]
        .to_dict()
    )

def get_top_possessions( raw_events, df, team_id, n=3, open_play_only=True, min_events=4):
    possessions = build_possessions(raw_events)
    shots = get_shots(raw_events)


    xg_lookup = build_xg_lookup(df)

    top_possessions = []

    
    for shot in shots:

        # keep shots from the selected team
        if shot["currentAttackingSquadId"] != team_id:
            continue

        # ignore set pieces if requested
        if open_play_only and shot["phase"] == "SET_PIECE":
            continue

        events = possessions[shot["sequenceIndex"]]

        # we ignore trivial possessions
        if len(events) < min_events:
            continue

        shot_id = shot["id"]
        xg = xg_lookup.get(shot_id)

        # incase xG is missing
        if xg is None:
            continue
        goal = False
        goal_shot_id = None
        for i, event in enumerate(events):
            if event["id"] == shot_id:

                for next_event in events[i + 1:]:

                    if next_event["actionType"] == "SHOT":
                        break

                    if next_event["actionType"] == "GOAL":
                        goal = True
                        break

                break

        top_possessions.append({
            "sequence_index": shot["sequenceIndex"],
            "shot": shot,
            "events": events,
            "xg": xg,
            "goal": goal,
            "goal_shot_id": goal_shot_id,
        })

    top_possessions.sort(
        key=lambda possession: possession["xg"],
        reverse=True,
    )

    return top_possessions[:n]