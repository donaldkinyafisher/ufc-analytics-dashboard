"""Training and prediction inputs built from the scraped database.

One row per fight: both fighters' attributes (height and reach in inches,
age, stance), their career stats from fights *before* this one, and the
fight's weight class, sex and title status.

Career stats follow ufcstats' definitions but are recomputed from stored
fight statistics so they never include the fight being predicted or later
ones (fighter_career_stats is today's snapshot and would leak results).
Upcoming fights run through the same code, so their stats cover each
fighter's full stored history.
"""

import numpy as np
import pandas as pd

from src.analytics.transform import build_fighter_fights, build_fights_mart
from src.ml.features import (
    CORNER_NUMERIC,
    CORNERS,
    LABEL_MAPPING,
    TARGET_COLUMN,
    swap_corners,
)

CAREER_STATS = [
    "ufc_fights", "win_rate", "slpm", "str_acc", "sapm", "str_def",
    "td_avg", "td_acc", "td_def", "sub_avg",
]
FIGHTER_ATTRIBUTES = ["height_in", "reach_in", "age", "stance"]

# Identifiers carried alongside the features (never used as features).
ID_COLUMNS = [
    "fight_id", "event_id", "event_date", "bout_order", TARGET_COLUMN,
    "red_fighter_id", "blue_fighter_id", "red_name", "blue_name",
]


def _ratio(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    """numerator / denominator, NaN where the denominator is 0."""
    return numerator / denominator.where(denominator > 0)


def pre_fight_stats(fighter_fights: pd.DataFrame) -> pd.DataFrame:
    """Each fighter's career stats going into each fight.

    Sums every fight on a strictly earlier date, so fights on the same card
    (early tournaments) never count towards each other. Fights without a
    result (upcoming) or without statistics (early era) add nothing to the
    rates; fights with a result still count towards ufc_fights and win_rate.
    Returns one row per (fight_id, fighter_id).
    """
    ff = fighter_fights
    fought = ff[TARGET_COLUMN].notna()
    has_stats = fought & ff["fighter_sig_str_landed"].notna() & ff["fight_seconds"].notna()

    def counted(column: str) -> pd.Series:
        return pd.to_numeric(ff[column], errors="coerce").where(has_stats).fillna(0)

    per_fight = pd.DataFrame({
        "fighter_id": ff["fighter_id"],
        "event_date": ff["event_date"],
        "fights": fought.astype(int),
        "wins": (fought & (ff["result"] == "W")).astype(int),
        "minutes": (ff["fight_seconds"] / 60).where(has_stats).fillna(0),
        "sig_landed": counted("fighter_sig_str_landed"),
        "sig_attempted": counted("fighter_sig_str_attempted"),
        "sig_absorbed": counted("opponent_sig_str_landed"),
        "opp_sig_attempted": counted("opponent_sig_str_attempted"),
        "td_landed": counted("fighter_td_landed"),
        "td_attempted": counted("fighter_td_attempted"),
        "opp_td_landed": counted("opponent_td_landed"),
        "opp_td_attempted": counted("opponent_td_attempted"),
        "sub_attempts": counted("fighter_submission_attempts"),
    })
    by_date = per_fight.groupby(["fighter_id", "event_date"]).sum().sort_index()
    # Running total minus the day itself: everything strictly before.
    before = by_date.groupby(level="fighter_id").cumsum() - by_date

    stats = pd.DataFrame(index=before.index)
    stats["ufc_fights"] = before["fights"]
    stats["win_rate"] = _ratio(before["wins"], before["fights"])
    stats["slpm"] = _ratio(before["sig_landed"], before["minutes"])
    stats["str_acc"] = _ratio(before["sig_landed"], before["sig_attempted"])
    stats["sapm"] = _ratio(before["sig_absorbed"], before["minutes"])
    stats["str_def"] = 1 - _ratio(before["sig_absorbed"], before["opp_sig_attempted"])
    stats["td_avg"] = _ratio(before["td_landed"] * 15, before["minutes"])
    stats["td_acc"] = _ratio(before["td_landed"], before["td_attempted"])
    stats["td_def"] = 1 - _ratio(before["opp_td_landed"], before["opp_td_attempted"])
    stats["sub_avg"] = _ratio(before["sub_attempts"] * 15, before["minutes"])

    keys = pd.MultiIndex.from_frame(ff[["fighter_id", "event_date"]])
    result = stats.reindex(keys).set_axis(pd.MultiIndex.from_frame(ff[["fight_id", "fighter_id"]]))
    return result[CAREER_STATS]


def fight_features(raw: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """One row per fight in raw: identifiers plus every model feature.

    Pass tables from load_raw_tables(include_upcoming=True) to also get rows
    for upcoming fights (winner_side is null for those).
    """
    mart = build_fights_mart(raw)
    stats = pre_fight_stats(build_fighter_fights(mart))

    features = pd.DataFrame({
        "fight_id": mart["fight_id"],
        "event_id": mart["event_id"],
        "event_date": mart["event_date"],
        "bout_order": mart["bout_order"],
        TARGET_COLUMN: mart[TARGET_COLUMN],
        "weight_class_lbs": mart["weight_class_lbs"],
        "is_title_bout": mart["is_title_bout"].astype(int),
        "sex": mart["gender"],
    })
    for corner in CORNERS:
        features[f"{corner}_fighter_id"] = mart[f"{corner}_fighter_id"]
        features[f"{corner}_name"] = mart[f"{corner}_name"]
        for attribute in FIGHTER_ATTRIBUTES:
            features[f"{corner}_{attribute}"] = mart[f"{corner}_{attribute}"]
        keys = pd.MultiIndex.from_arrays([mart["fight_id"], mart[f"{corner}_fighter_id"]])
        corner_stats = stats.reindex(keys).set_axis(mart.index)
        for stat in CAREER_STATS:
            features[f"{corner}_{stat}"] = corner_stats[stat]
    for feature in CORNER_NUMERIC:
        features[f"{feature}_diff"] = features[f"red_{feature}"] - features[f"blue_{feature}"]
    return features


def training_set(
    features: pd.DataFrame, test_size: float = 0.2, random_state: int = 42
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """(X_train, X_test, y_train, y_test) from fight_features rows.

    Only fights with a red or blue winner are kept. Corners are swapped in a
    random half of the fights, since ufcstats lists the winner first and the
    label would otherwise be mostly "red". The split is chronological: the
    newest test_size share of fights is the test set, as in real use. Fights
    on one date never straddle the split. y is 1 when red wins.
    """
    decided = features[features[TARGET_COLUMN].isin(LABEL_MAPPING)]
    decided = decided.sort_values(["event_date", "fight_id"], ignore_index=True)

    swap = np.random.default_rng(random_state).random(len(decided)) < 0.5
    decided = pd.concat([decided[~swap], swap_corners(decided[swap])]).sort_index()

    cutoff = decided["event_date"].iloc[int(len(decided) * (1 - test_size))]
    is_test = decided["event_date"] >= cutoff
    labels = decided[TARGET_COLUMN].map(LABEL_MAPPING).astype(int)
    return decided[~is_test], decided[is_test], labels[~is_test], labels[is_test]
