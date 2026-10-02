"""
Transform: reshape raw tables into analysis-ready marts.

Fighters are kept as red/blue, with red-minus-blue differences.

Corner bias: ufcstats lists the winner first, so "red" is not the real corner
and red wins ~63% of fights. Differences therefore lean towards the winner's
side. Before using these columns as model features, randomise or swap the
sides so the label is not leaked by position.
"""

import re

import numpy as np
import pandas as pd

from src.scrapers.historical_scraper import _seconds

# Men's divisions lightest to heaviest, then women's, then the irregular ones.
WEIGHT_CLASS_ORDER = [
    "Flyweight", "Bantamweight", "Featherweight", "Lightweight", "Welterweight",
    "Middleweight", "Light Heavyweight", "Heavyweight", "Super Heavyweight",
    "Women's Strawweight", "Women's Flyweight", "Women's Bantamweight",
    "Women's Featherweight",
    "Catch Weight", "Open Weight", "Unknown",
]

WEIGHT_CLASS_LBS = {
    "Strawweight": 115, "Flyweight": 125, "Bantamweight": 135, "Featherweight": 145,
    "Lightweight": 155, "Welterweight": 170, "Middleweight": 185,
    "Light Heavyweight": 205, "Heavyweight": 265,
}

METHOD_GROUPS = {
    "KO/TKO": "KO/TKO",
    "TKO - Doctor's Stoppage": "KO/TKO",
    "Submission": "SUB",
    "Decision - Unanimous": "U-DEC",
    "Decision - Split": "S-DEC",
    "Decision - Majority": "M-DEC",
    "DQ": "DQ",
}
METHOD_GROUP_ORDER = ["KO/TKO", "SUB", "U-DEC", "S-DEC", "M-DEC", "DQ", "Other"]
FINISH_METHODS = {"KO/TKO", "SUB"}

FIGHTER_ATTRIBUTES = ["name", "height_in", "reach_in", "stance", "dob"]

# Raw per-fighter counts carried from fight_statistics.
STAT_COUNTS = [
    "knockdowns", "sig_str_landed", "sig_str_attempted", "total_str_landed",
    "total_str_attempted", "td_landed", "td_attempted", "submission_attempts",
    "reversals", "control_time_seconds",
    "head_landed", "body_landed", "leg_landed",
    "distance_landed", "clinch_landed", "ground_landed",
]

# Columns that also get a red-minus-blue difference.
DIFF_COLUMNS = [
    "height_in", "reach_in", "age",
    "knockdowns", "sig_str_landed", "sig_str_acc", "sig_str_per_min",
    "td_landed", "td_acc", "td_per_15", "submission_attempts",
    "control_time_seconds", "control_share",
]


def round_lengths_minutes(time_format: str | None) -> list[int]:
    """'3 Rnd (5-5-5)' -> [5, 5, 5]; 'No Time Limit' -> []."""
    match = re.search(r"\(([\d-]+)\)", time_format or "")
    return [int(minutes) for minutes in match.group(1).split("-")] if match else []


def fight_seconds(round_: int | None, time: str | None, time_format: str | None) -> float:
    """Elapsed fight time: completed rounds plus the clock in the final round."""
    clock = _seconds(time)
    if pd.isna(round_) or clock is None:
        return np.nan
    completed = int(round_) - 1
    lengths = round_lengths_minutes(time_format)
    if completed > len(lengths):
        return np.nan
    return float(sum(lengths[:completed]) * 60 + clock)


def _fighter_frame(fighters: pd.DataFrame, fighter_ids: pd.Series, prefix: str) -> pd.DataFrame:
    """Attributes of the given fighter ids, columns prefixed, aligned to fighter_ids."""
    attributes = fighters.set_index("id")[FIGHTER_ATTRIBUTES]
    frame = attributes.reindex(fighter_ids.to_numpy()).set_axis(fighter_ids.index)
    return frame.add_prefix(prefix)


def _ratio(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    """numerator / denominator, NaN where the denominator is 0 or missing."""
    return numerator / denominator.where(denominator > 0)


def _corner_stats(stats: pd.DataFrame, corner: str, seconds: pd.Series) -> pd.DataFrame:
    """Counts and rates for one corner, indexed by fight_id, columns prefixed."""
    frame = stats.loc[stats["corner"] == corner].set_index("fight_id")[STAT_COUNTS]
    frame = frame.reindex(seconds.index)
    minutes = seconds / 60
    sig_landed = frame["sig_str_landed"]
    frame["sig_str_acc"] = _ratio(sig_landed, frame["sig_str_attempted"])
    frame["sig_str_per_min"] = _ratio(sig_landed, minutes)
    frame["td_acc"] = _ratio(frame["td_landed"], frame["td_attempted"])
    frame["td_per_15"] = _ratio(frame["td_landed"] * 15, minutes)
    frame["control_share"] = _ratio(frame["control_time_seconds"], seconds)
    for part in ("head", "body", "leg", "distance", "clinch", "ground"):
        frame[f"{part}_share"] = _ratio(frame[f"{part}_landed"], sig_landed)
    return frame.add_prefix(f"{corner}_")


def build_fights_mart(raw: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """One row per fight on a completed event: outcome, both fighters and their fight stats."""
    events = raw["events"].rename(columns={"id": "event_id", "name": "event_name"})
    fights = raw["fights"].rename(columns={"id": "fight_id"})
    mart = fights.merge(events, on="event_id", how="inner")

    mart["year"] = mart["event_date"].dt.year
    mart["decade"] = (mart["year"] // 10 * 10).astype("Int64")

    weight_class = mart["weight_class"].fillna("Unknown").replace("", "Unknown")
    mart["weight_class"] = pd.Categorical(weight_class, categories=WEIGHT_CLASS_ORDER, ordered=True)
    mart["gender"] = np.where(weight_class.str.startswith("Women's"), "Women", "Men")
    mart["weight_class_lbs"] = weight_class.str.removeprefix("Women's ").map(WEIGHT_CLASS_LBS)
    mart["is_title_bout"] = mart["is_title_bout"].astype(bool)

    method_group = mart["method"].map(METHOD_GROUPS).fillna("Other")
    mart["method_group"] = pd.Categorical(method_group, categories=METHOD_GROUP_ORDER, ordered=True)
    mart["is_finish"] = method_group.isin(FINISH_METHODS)
    mart["scheduled_rounds"] = mart["time_format"].map(lambda tf: len(round_lengths_minutes(tf)) or np.nan)
    mart["fight_seconds"] = [
        fight_seconds(r, t, tf) for r, t, tf in zip(mart["round"], mart["time"], mart["time_format"])
    ]

    decided = mart["winner_side"].isin(["red", "blue"])
    mart["outcome"] = np.where(decided, "win", mart["winner_side"])
    mart["winner_id"] = (
        mart["red_fighter_id"].where(mart["winner_side"] == "red", mart["blue_fighter_id"])
        .where(decided).astype("Int64")
    )

    fighters = raw["fighters"]
    seconds = mart.set_index("fight_id")["fight_seconds"]
    mart = pd.concat(
        [mart, _fighter_frame(fighters, mart["red_fighter_id"], "red_"),
         _fighter_frame(fighters, mart["blue_fighter_id"], "blue_")],
        axis=1,
    )
    for corner in ("red", "blue"):
        mart[f"{corner}_age"] = (mart["event_date"] - mart.pop(f"{corner}_dob")).dt.days / 365.25
        mart[f"{corner}_stance"] = mart[f"{corner}_stance"].replace("", np.nan)
        mart = mart.join(_corner_stats(raw["fight_statistics"], corner, seconds), on="fight_id")

    mart["winner_name"] = mart["red_name"].where(mart["winner_side"] == "red", mart["blue_name"]).where(decided)
    for column in DIFF_COLUMNS:
        mart[f"{column}_diff"] = mart[f"red_{column}"] - mart[f"blue_{column}"]
    mart["stance_matchup"] = [
        " vs ".join(sorted([r, b])) if isinstance(r, str) and isinstance(b, str) else np.nan
        for r, b in zip(mart["red_stance"], mart["blue_stance"])
    ]

    fight_columns = [
        "fight_id", "event_id", "event_name", "event_date", "year", "decade", "location",
        "bout_order", "weight_class", "gender", "weight_class_lbs", "is_title_bout",
        "outcome", "winner_side", "winner_id", "winner_name",
        "method", "method_details", "method_group", "is_finish",
        "round", "time", "time_format", "scheduled_rounds", "fight_seconds", "referee",
        "red_fighter_id", "blue_fighter_id",
    ]
    corner_columns = [c for c in mart.columns if c.startswith(("red_", "blue_")) and c not in fight_columns]
    diff_columns = [f"{column}_diff" for column in DIFF_COLUMNS] + ["stance_matchup"]
    return mart[fight_columns + corner_columns + diff_columns].sort_values(
        ["event_date", "bout_order"], ignore_index=True
    )


RESULT_CODES = {"draw": "D", "nc": "NC"}


def build_fighter_fights(mart: pd.DataFrame) -> pd.DataFrame:
    """One row per fighter per fight, from that fighter's point of view.

    Each red_*/blue_* pair becomes fighter_* (own) and opponent_* (the other
    corner), so a fighter's career can be read without caring which corner
    they were listed in.
    """
    shared = [c for c in mart.columns if not c.startswith(("red_", "blue_")) and not c.endswith("_diff")]
    shared.remove("stance_matchup")
    suffixes = [c.removeprefix("red_") for c in mart.columns if c.startswith("red_")]

    views = []
    for own, other in (("red", "blue"), ("blue", "red")):
        view = mart[shared].copy()
        for suffix in suffixes:
            fighter_col, opponent_col = (
                ("fighter_id", "opponent_id") if suffix == "fighter_id"
                else (f"fighter_{suffix}", f"opponent_{suffix}")
            )
            view[fighter_col] = mart[f"{own}_{suffix}"]
            view[opponent_col] = mart[f"{other}_{suffix}"]
        view["corner"] = own
        view["result"] = np.select(
            [mart["winner_side"] == own, mart["winner_side"] == other],
            ["W", "L"],
            default=mart["winner_side"].map(RESULT_CODES).fillna("NC"),
        )
        views.append(view)
    return pd.concat(views, ignore_index=True).sort_values(
        ["event_date", "bout_order", "corner"], ascending=[True, True, False], ignore_index=True
    )
