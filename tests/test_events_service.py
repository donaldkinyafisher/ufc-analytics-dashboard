"""Upcoming-events sync: event rows, scraped_at and fight cards."""

import asyncio
from datetime import date, datetime

import pytest
from sqlalchemy import select

from src.api.v1.models import Event, Fight, Fighter, FightStatistic
from src.api.v1.services import events_services
from src.api.v1.services.events_services import store_upcoming_card, sync_upcoming_events
from src.scrapers.historical_scraper import parse_event_details
from tests.helpers import FakeBrowser, count, fixture

TODAY = date(2026, 10, 2)
FIRST_SYNC = datetime(2026, 10, 1, 9, 0)
SECOND_SYNC = datetime(2026, 10, 2, 9, 0)


def scraped(name="UFC 332: Silva vs. Wang", raw_date="October 03, 2026", url="http://ufcstats.com/event-details/a"):
    return {"event_name": name, "date": raw_date, "location": "Las Vegas, Nevada, USA", "source_url": url}


def test_sync_sets_scraped_at_on_inserted_events(db):
    result = sync_upcoming_events(db, [scraped()], TODAY, synced_at=FIRST_SYNC)

    event = db.scalar(select(Event))
    assert result.inserted == 1
    assert event.status == "upcoming"
    assert event.scraped_at == FIRST_SYNC


def test_resync_refreshes_scraped_at_without_counting_an_update(db):
    sync_upcoming_events(db, [scraped()], TODAY, synced_at=FIRST_SYNC)
    result = sync_upcoming_events(db, [scraped()], TODAY, synced_at=SECOND_SYNC)

    assert (result.inserted, result.updated) == (0, 0)
    assert db.scalar(select(Event)).scraped_at == SECOND_SYNC


def test_expired_event_is_completed_and_scraped_at_cleared(db):
    db.add(Event(name="UFC 331", event_date=date(2026, 9, 19), status="upcoming", scraped_at=FIRST_SYNC))
    db.flush()

    result = sync_upcoming_events(db, [], TODAY, synced_at=SECOND_SYNC)

    event = db.scalar(select(Event))
    db.refresh(event)
    assert result.completed == 1
    assert event.status == "completed"
    assert event.scraped_at is None


def test_malformed_events_get_no_scraped_at(db):
    result = sync_upcoming_events(db, [scraped(raw_date="TBD")], TODAY, synced_at=FIRST_SYNC)

    assert result.malformed == 1
    assert db.scalar(select(Event)) is None


# --- upcoming cards ----------------------------------------------------------

UPCOMING_URL = "http://ufcstats.com/event-details/ad3fdba28a7540cf"


def upcoming_card() -> list[dict]:
    return parse_event_details(fixture("event_upcoming.html"), UPCOMING_URL)["fights"]


@pytest.fixture
def upcoming(db):
    event = Event(name="UFC 332: Silva vs. Wang", event_date=date(2026, 10, 3),
                  status="upcoming", source_url=UPCOMING_URL)
    db.add(event)
    db.flush()
    return event


def test_store_card_inserts_fights_and_fighter_stubs(db, upcoming):
    fights = store_upcoming_card(db, upcoming, upcoming_card())

    assert len(fights) == count(db, Fight) == 14
    assert count(db, FightStatistic) == 0
    assert upcoming.ufcstats_id == "ad3fdba28a7540cf"
    main_event = db.scalar(select(Fight).where(Fight.bout_order == 1))
    assert (main_event.red_fighter.name, main_event.blue_fighter.name) == ("Natalia Silva", "Wang Cong")
    assert main_event.bout_type == "UFC Women's Flyweight Title Bout"
    assert main_event.is_title_bout is True
    assert main_event.winner_side is None
    # New fighters are stubs, left for the next ingestion run to fill in.
    assert main_event.red_fighter.last_synced_at is None


def test_store_card_matches_existing_fighters_by_ufcstats_id(db, upcoming):
    silva = Fighter(name="Natalia Silva", ufcstats_id="262d32ebda89efc4", height_in=64.0, reach_in=65.0)
    db.add(silva)
    db.flush()

    store_upcoming_card(db, upcoming, upcoming_card())

    main_event = db.scalar(select(Fight).where(Fight.bout_order == 1))
    assert main_event.red_fighter_id == silva.id
    assert silva.height_in == 64.0
    assert count(db, Fighter) == 28


def test_store_card_is_idempotent(db, upcoming):
    store_upcoming_card(db, upcoming, upcoming_card())
    store_upcoming_card(db, upcoming, upcoming_card())

    assert count(db, Fight) == 14
    assert count(db, Fighter) == 28


def test_bout_dropped_from_card_is_deleted(db, upcoming):
    card = upcoming_card()
    store_upcoming_card(db, upcoming, card)

    store_upcoming_card(db, upcoming, card[:-1])

    assert count(db, Fight) == 13
    assert db.scalar(select(Fight).where(Fight.ufcstats_id == card[-1]["ufcstats_id"])) is None


def test_empty_card_keeps_stored_fights(db, upcoming):
    store_upcoming_card(db, upcoming, upcoming_card())

    store_upcoming_card(db, upcoming, [])

    assert count(db, Fight) == 14


def test_malformed_bouts_are_skipped(db, upcoming):
    card = upcoming_card()
    card[0]["fighters"] = card[0]["fighters"][:1]
    card[1]["fighters"][1]["ufcstats_id"] = None

    fights = store_upcoming_card(db, upcoming, card)

    assert len(fights) == 12


def test_completed_event_card_is_not_overwritten(db, upcoming):
    upcoming.status = "completed"

    with pytest.raises(ValueError, match="not upcoming"):
        store_upcoming_card(db, upcoming, upcoming_card())


def test_sync_cards_keeps_going_when_one_page_fails(db, upcoming, monkeypatch):
    broken = Event(name="UFC 999", event_date=date(2030, 1, 1), status="upcoming",
                   source_url="http://ufcstats.com/event-details/broken")
    db.add(broken)
    db.flush()

    async def fake_fetch_event(browser, url):
        if url == broken.source_url:
            raise TimeoutError("page did not render")
        return parse_event_details(fixture("event_upcoming.html"), url)

    monkeypatch.setattr(events_services, "UFCStatsBrowser", FakeBrowser)
    monkeypatch.setattr(events_services, "fetch_event", fake_fetch_event)
    result = asyncio.run(events_services.sync_upcoming_cards(db, [upcoming, broken]))

    assert result.fights == 14
    assert result.failed_events == ["UFC 999"]
