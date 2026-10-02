import pandas as pd
import pytest

from src.analytics.transform import (
    build_fighter_fights,
    build_fights_mart,
    fight_seconds,
    round_lengths_minutes,
)


@pytest.mark.parametrize(
    ("time_format", "expected"),
    [
        ("3 Rnd (5-5-5)", [5, 5, 5]),
        ("1 Rnd + OT (12-3)", [12, 3]),
        ("No Time Limit", []),
        (None, []),
    ],
)
def test_round_lengths_minutes(time_format, expected):
    assert round_lengths_minutes(time_format) == expected


@pytest.mark.parametrize(
    ("round_", "time", "time_format", "expected"),
    [
        (1, "2:13", "3 Rnd (5-5-5)", 133),
        (3, "5:00", "3 Rnd (5-5-5)", 900),
        (2, "1:00", "1 Rnd + OT (12-3)", 780),
        (1, "20:00", "No Time Limit", 1200),
        (2, "1:00", "No Time Limit", None),
        (None, "1:00", "3 Rnd (5-5-5)", None),
        (1, "", "3 Rnd (5-5-5)", None),
    ],
)
def test_fight_seconds(round_, time, time_format, expected):
    result = fight_seconds(round_, time, time_format)
    assert pd.isna(result) if expected is None else result == expected


@pytest.fixture
def rows(raw):
    """Mart rows in date/bout order: ufc1, title, split, draw."""
    return build_fights_mart(raw).to_dict("records")


def test_extract_parses_dates_and_skips_upcoming(raw):
    assert len(raw["events"]) == 2
    assert pd.api.types.is_datetime64_any_dtype(raw["events"]["event_date"])
    assert pd.api.types.is_datetime64_any_dtype(raw["fighters"]["dob"])


def test_fights_mart_one_row_per_completed_fight(raw):
    mart = build_fights_mart(raw)
    assert len(mart) == 4
    assert mart["fight_id"].is_unique
    assert list(mart["event_name"]) == ["UFC 1", "UFC 300", "UFC 300", "UFC 300"]


def test_fights_mart_keeps_corners_and_winner(rows):
    _, title, _, draw = rows
    assert (title["red_name"], title["blue_name"], title["winner_name"]) == ("Alpha", "Bravo", "Bravo")
    assert title["winner_id"] == title["blue_fighter_id"]
    assert title["height_in_diff"] == 2 and title["reach_in_diff"] == 2
    assert title["age_diff"] == pytest.approx(5, abs=0.01)
    assert title["stance_matchup"] == "Orthodox vs Southpaw"
    assert draw["outcome"] == "draw"
    assert pd.isna(draw["winner_id"]) and pd.isna(draw["winner_name"])


def test_fights_mart_fight_columns(rows):
    ufc1, title, split, draw = rows
    assert (ufc1["method_group"], ufc1["is_finish"], ufc1["fight_seconds"]) == ("SUB", True, 26)
    assert pd.isna(ufc1["scheduled_rounds"]) and ufc1["decade"] == 1990
    assert (title["method_group"], title["fight_seconds"], title["gender"]) == ("KO/TKO", 390, "Women")
    assert title["weight_class_lbs"] == 125 and title["year"] == 2024
    assert (split["method_group"], split["is_finish"], split["fight_seconds"]) == ("S-DEC", False, 900)
    assert split["scheduled_rounds"] == 3 and split["gender"] == "Men"
    assert draw["weight_class"] == "Unknown" and draw["method_group"] == "M-DEC"
    assert pd.isna(split["blue_age"]) and pd.isna(split["blue_stance"])


def test_fights_mart_stat_rates_and_diffs(rows):
    _, title, _, _ = rows
    assert title["red_sig_str_acc"] == 0.5
    assert title["red_sig_str_per_min"] == pytest.approx(2.0)
    assert title["blue_sig_str_per_min"] == pytest.approx(4.0)
    assert title["red_td_acc"] == 0.25
    assert pd.isna(title["blue_td_acc"])  # no attempts, not 0%
    assert title["red_td_per_15"] == pytest.approx(15 / 6.5)
    assert title["red_control_share"] == pytest.approx(0.1)
    assert title["blue_head_share"] == pytest.approx(20 / 26)
    assert title["sig_str_landed_diff"] == -13
    assert title["knockdowns_diff"] == -1
    assert title["sig_str_per_min_diff"] == pytest.approx(-2.0)


def test_fights_mart_missing_stats_stay_missing(rows):
    ufc1, _, split, _ = rows
    assert pd.isna(ufc1["red_sig_str_landed"]) and pd.isna(ufc1["red_sig_str_acc"])
    # No fight_statistics rows at all for this fight.
    assert pd.isna(split["red_knockdowns"]) and pd.isna(split["sig_str_landed_diff"])


def test_fights_mart_categories_are_ordered(raw):
    mart = build_fights_mart(raw)
    assert mart["weight_class"].cat.ordered
    assert list(mart["method_group"].cat.categories)[:2] == ["KO/TKO", "SUB"]


def test_fighter_fights_two_rows_per_fight_from_each_side(raw):
    mart = build_fights_mart(raw)
    fighter_fights = build_fighter_fights(mart)
    assert len(fighter_fights) == 2 * len(mart)
    title = fighter_fights[fighter_fights["is_title_bout"]].set_index("fighter_name")
    assert title.loc["Alpha", "opponent_name"] == "Bravo"
    assert (title.loc["Alpha", "result"], title.loc["Bravo", "result"]) == ("L", "W")
    assert title.loc["Bravo", "fighter_sig_str_landed"] == 26
    assert title.loc["Bravo", "opponent_sig_str_landed"] == 13
    assert title.loc["Bravo", "fighter_id"] == title.loc["Alpha", "opponent_id"]
    assert not any(c.startswith(("red_", "blue_")) or c.endswith("_diff") for c in fighter_fights.columns)


def test_fighter_fights_results(raw):
    fighter_fights = build_fighter_fights(build_fights_mart(raw))
    results = fighter_fights.groupby("fighter_name")["result"].apply(lambda r: "".join(sorted(r)))
    assert results.to_dict() == {"Alpha": "DLW", "Bravo": "WW", "Charlie": "DLL"}
