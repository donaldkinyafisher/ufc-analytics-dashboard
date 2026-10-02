"""Parser tests against rendered ufcstats pages saved by src.scrapers.save_samples."""

from datetime import date
from pathlib import Path

import pytest

from src.scrapers.historical_scraper import (
    _inches,
    _landed_attempted,
    _parse_record,
    _pct,
    _seconds,
    parse_completed_events,
    parse_event_details,
    parse_fight_details,
    parse_fighter_details,
    ufcstats_id,
)

FIXTURES = Path(__file__).parent / "fixtures"


def fixture(name: str) -> str:
    return (FIXTURES / name).read_text(encoding="utf-8")


# --- value helpers ---------------------------------------------------------

@pytest.mark.parametrize(
    ("text", "expected"),
    [("181 of 305", (181, 305)), ("0 of 0", (0, 0)), ("---", (None, None)), ("", (None, None))],
)
def test_landed_attempted(text, expected):
    assert _landed_attempted(text) == expected


@pytest.mark.parametrize(("text", "expected"), [("2:13", 133), ("0:00", 0), ("--", None)])
def test_seconds(text, expected):
    assert _seconds(text) == expected


@pytest.mark.parametrize(
    ("text", "expected"), [("5' 5\"", 65.0), ("6' 0\"", 72.0), ('65"', 65.0), ("--", None)]
)
def test_inches(text, expected):
    assert _inches(text) == expected


@pytest.mark.parametrize(("text", "expected"), [("57%", 0.57), ("---", None)])
def test_pct(text, expected):
    assert _pct(text) == expected


def test_record_with_no_contests():
    assert _parse_record("Record: 27-1-0 (1 NC)") == {
        "wins": 27, "losses": 1, "draws": 0, "no_contests": 1,
    }


def test_ufcstats_id():
    assert ufcstats_id("http://ufcstats.com/fight-details/568ec6af4008355a") == "568ec6af4008355a"
    assert ufcstats_id(None) is None


# --- completed events ------------------------------------------------------

def test_parse_completed_events():
    events = parse_completed_events(fixture("completed_events.html"))

    assert len(events) > 10
    ufc331 = next(e for e in events if e["ufcstats_id"] == "8a0a35e7c74bebcc")
    assert ufc331["name"] == "UFC 331: Van vs. Pantoja 2"
    assert ufc331["event_date"] == date(2026, 9, 19)
    assert ufc331["location"] == "Los Angeles, California, USA"
    # Newest first.
    dates = [e["event_date"] for e in events]
    assert dates == sorted(dates, reverse=True)


# --- event details ---------------------------------------------------------

def test_parse_event_details():
    url = "http://ufcstats.com/event-details/8a0a35e7c74bebcc"
    event = parse_event_details(fixture("event_ufc331.html"), url)

    assert event["ufcstats_id"] == "8a0a35e7c74bebcc"
    assert event["name"] == "UFC 331: Van vs. Pantoja 2"
    assert event["event_date"] == date(2026, 9, 19)
    assert len(event["fights"]) == 12
    main_event = event["fights"][0]
    assert main_event["bout_order"] == 1
    assert main_event["ufcstats_id"] == "568ec6af4008355a"
    assert [f["name"] for f in main_event["fighters"]] == ["Joshua Van", "Alexandre Pantoja"]
    assert main_event["fighters"][0]["ufcstats_id"] == "17e97649403ba428"
    assert main_event["weight_class"] == "Flyweight"
    assert main_event["bout_type"] == "UFC Flyweight Title Bout"
    assert main_event["is_title_bout"] is True


def test_parse_upcoming_event_card():
    url = "http://ufcstats.com/event-details/ad3fdba28a7540cf"
    event = parse_event_details(fixture("event_upcoming.html"), url)

    assert event["name"] == "UFC 332: Silva vs. Wang"
    assert event["event_date"] == date(2026, 10, 3)
    assert len(event["fights"]) == 14
    main_event = event["fights"][0]
    assert main_event["bout_order"] == 1
    assert main_event["fighters"] == [
        {"corner": "red", "name": "Natalia Silva", "ufcstats_id": "262d32ebda89efc4",
         "profile_url": "http://ufcstats.com/fighter-details/262d32ebda89efc4"},
        {"corner": "blue", "name": "Wang Cong", "ufcstats_id": "2997e7fe3c9d3d4a",
         "profile_url": "http://ufcstats.com/fighter-details/2997e7fe3c9d3d4a"},
    ]
    assert main_event["weight_class"] == "Women's Flyweight"
    assert main_event["bout_type"] == "UFC Women's Flyweight Title Bout"
    assert main_event["is_title_bout"] is True
    co_main = event["fights"][1]
    assert (co_main["weight_class"], co_main["bout_type"], co_main["is_title_bout"]) == (
        "Bantamweight", "Bantamweight Bout", False,
    )
    assert all(len(fight["fighters"]) == 2 for fight in event["fights"])


# --- fight details ---------------------------------------------------------

def test_parse_fight_title_decision():
    fight = parse_fight_details(fixture("fight_title_decision.html"))

    assert fight["event_ufcstats_id"] == "8a0a35e7c74bebcc"
    assert fight["weight_class"] == "Flyweight"
    assert fight["is_title_bout"] is True
    assert fight["method"] == "Decision - Unanimous"
    assert fight["round"] == 5
    assert fight["time"] == "5:00"
    assert fight["referee"] == "Herb Dean"
    assert fight["winner_side"] == "red"

    red, blue = fight["fighters"]
    assert (red["name"], red["result"], red["ufcstats_id"]) == ("Joshua Van", "W", "17e97649403ba428")
    assert (blue["name"], blue["result"]) == ("Alexandre Pantoja", "L")
    assert red["nickname"] == "The Fearless"

    red_stats, blue_stats = fight["stats"]["red"], fight["stats"]["blue"]
    assert red_stats["knockdowns"] == 1
    assert (red_stats["sig_str_landed"], red_stats["sig_str_attempted"]) == (181, 305)
    assert (red_stats["total_str_landed"], red_stats["total_str_attempted"]) == (259, 392)
    assert (blue_stats["td_landed"], blue_stats["td_attempted"]) == (5, 18)
    assert red_stats["control_time_seconds"] == 133
    assert blue_stats["control_time_seconds"] == 551
    assert (red_stats["head_landed"], red_stats["head_attempted"]) == (158, 278)
    assert (blue_stats["leg_landed"], blue_stats["leg_attempted"]) == (26, 34)
    assert (red_stats["ground_landed"], red_stats["ground_attempted"]) == (4, 6)


def test_parse_fight_early_era_missing_values():
    fight = parse_fight_details(fixture("fight_early_era.html"))

    assert fight["weight_class"] is None
    assert fight["time_format"] == "No Time Limit"
    assert fight["method"] == "KO/TKO"
    assert fight["winner_side"] == "red"
    assert fight["stats"]["red"]["control_time_seconds"] is None
    assert fight["stats"]["red"]["sig_str_landed"] == 4


def test_parse_fight_no_contest():
    fight = parse_fight_details(fixture("fight_no_contest.html"))

    assert fight["method"] == "Overturned"
    assert fight["winner_side"] == "nc"
    assert [f["result"] for f in fight["fighters"]] == ["NC", "NC"]
    assert fight["weight_class"] == "Light Heavyweight"


def test_parse_fight_draw():
    fight = parse_fight_details(fixture("fight_draw.html"))

    assert fight["method"] == "Decision - Majority"
    assert fight["winner_side"] == "draw"
    assert [f["result"] for f in fight["fighters"]] == ["D", "D"]


# --- fighter details -------------------------------------------------------

def test_parse_fighter_complete():
    fighter = parse_fighter_details(fixture("fighter_complete.html"))

    assert fighter["name"] == "Joshua Van"
    assert fighter["nickname"] == "The Fearless"
    assert (fighter["wins"], fighter["losses"], fighter["draws"]) == (18, 2, 0)
    assert fighter["height_in"] == 65.0
    assert fighter["reach_in"] == 65.0
    assert fighter["weight_lbs"] == 125.0
    assert fighter["stance"] == "Orthodox"
    assert fighter["dob"] == date(2001, 10, 10)
    assert fighter["career_stats"]["strikes_landed_per_minute"] == 8.26
    assert fighter["career_stats"]["striking_accuracy"] == 0.57
    assert fighter["career_stats"]["takedown_defense"] == 0.75


def test_parse_fighter_sparse():
    fighter = parse_fighter_details(fixture("fighter_sparse.html"))

    assert fighter["name"] == "Patrick Smith"
    assert fighter["reach_in"] is None
    assert fighter["nickname"] is None
    assert fighter["height_in"] == 74.0
