"""Ingestion service tests: in-memory SQLite fed with parsed fixture pages."""

import asyncio
import copy
import json
from datetime import date

import pytest
from sqlalchemy import func, select

from src.api.v1.models import Event, Fight, Fighter, FightStatistic, ScrapeJob
from src.api.v1.services import ingestion_services as svc
from src.scrapers.historical_scraper import parse_fight_details, parse_fighter_details
from tests.helpers import (
    BASE,
    EVENT_URL,
    SMITH_URL,
    VAN_URL,
    count,
    event_data,
    fixture,
)

# --- DB functions ----------------------------------------------------------

def test_ingest_event_stores_event_fights_fighters_and_stats(db):
    event, fights = svc.ingest_event(db, event_data())
    db.commit()

    assert event.status == "completed"
    assert event.scraped_at is not None
    assert event.ufcstats_id == "8a0a35e7c74bebcc"
    assert len(fights) == 3
    assert count(db, Fighter) == 6
    assert count(db, FightStatistic) == 6

    title = db.scalar(select(Fight).where(Fight.ufcstats_id == "568ec6af4008355a"))
    assert title.bout_order == 1
    assert title.winner_side == "red"
    assert title.red_fighter.name == "Joshua Van"
    assert title.is_title_bout is True
    assert title.referee == "Herb Dean"
    assert title.method_details.startswith("Sal D'amato")

    red = next(s for s in title.statistics if s.corner == "red")
    assert red.result == "W"
    assert (red.sig_str_landed, red.sig_str_attempted) == (181, 305)
    assert red.control_time_seconds == 133

    winner_sides = {f.ufcstats_id: f.winner_side for f in db.scalars(select(Fight))}
    assert winner_sides["5ab286735ccee803"] == "nc"
    assert winner_sides["c258996570faeec9"] == "draw"


def test_ingest_event_is_idempotent(db):
    svc.ingest_event(db, event_data())
    db.commit()
    svc.ingest_event(db, event_data())
    db.commit()

    assert count(db, Event) == 1
    assert count(db, Fight) == 3
    assert count(db, Fighter) == 6
    assert count(db, FightStatistic) == 6


def test_upcoming_event_is_promoted_not_duplicated(db):
    db.add(Event(name="UFC 331: Van vs. Pantoja 2", source_url=EVENT_URL, status="upcoming"))
    db.commit()

    svc.ingest_event(db, event_data())
    db.commit()

    events = db.scalars(select(Event)).all()
    assert len(events) == 1
    assert events[0].status == "completed"
    assert events[0].ufcstats_id == "8a0a35e7c74bebcc"


def test_fighter_appearing_twice_on_one_card(db):
    """Early tournaments: the same fighter fights more than once per event."""
    first = parse_fight_details(fixture("fight_early_era.html"), f"{BASE}/fight-details/aaa")
    second = copy.deepcopy(first)
    second["ufcstats_id"] = "bbb"
    second["fighters"][1].update(name="Other Fighter", ufcstats_id="0000", profile_url=f"{BASE}/fighter-details/0000")
    data = {"ufcstats_id": "ufc2", "name": "UFC 2", "event_date": date(1994, 3, 11),
            "source_url": f"{BASE}/event-details/ufc2", "fights": [first, second]}

    svc.ingest_event(db, data)
    db.commit()

    assert count(db, Fight) == 2
    assert count(db, Fighter) == 3  # Royce Gracie stored once


def test_apply_fighter_profile_updates_stub_and_career_stats(db):
    svc.ingest_event(db, event_data())
    db.commit()

    fighter = svc.apply_fighter_profile(db, parse_fighter_details(fixture("fighter_complete.html"), VAN_URL))
    db.commit()

    assert count(db, Fighter) == 6  # updated the stub, did not add a fighter
    assert (fighter.wins, fighter.losses, fighter.draws) == (18, 2, 0)
    assert fighter.height_in == 65.0
    assert fighter.dob == date(2001, 10, 10)
    assert fighter.last_synced_at is not None
    assert fighter.career_stats.strikes_landed_per_minute == 8.26


def test_apply_sparse_profile_stores_missing_as_none(db):
    fighter = svc.apply_fighter_profile(db, parse_fighter_details(fixture("fighter_sparse.html"), SMITH_URL))
    db.commit()

    assert fighter.name == "Patrick Smith"
    assert fighter.reach_in is None
    assert fighter.height_in == 74.0


# --- job runner ------------------------------------------------------------

class FakeBrowser:
    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc_info):
        return None


OTHER_EVENT_URL = f"{BASE}/event-details/1111111111111111"
LISTED = [
    {"ufcstats_id": "1111111111111111", "name": "Newer Event", "event_date": date(2026, 9, 26),
     "location": "Las Vegas", "source_url": OTHER_EVENT_URL, "is_next": False},
    {"ufcstats_id": "8a0a35e7c74bebcc", "name": "UFC 331: Van vs. Pantoja 2", "event_date": date(2026, 9, 19),
     "location": "Los Angeles, California, USA", "source_url": EVENT_URL, "is_next": False},
]


@pytest.fixture
def fake_crawl(monkeypatch):
    """Patch the scraper so UFC 331 loads from fixtures and the newer event fails."""
    state = {"fail": {OTHER_EVENT_URL}, "fetched_events": []}

    async def fetch_completed_events(browser, base_url, since=None, until=None):
        return [e for e in LISTED if (since is None or e["event_date"] >= since)]

    async def fetch_event_with_fights(browser, url):
        state["fetched_events"].append(url)
        if url in state["fail"]:
            raise TimeoutError("page did not render")
        if url == OTHER_EVENT_URL:
            data = copy.deepcopy(event_data())
            data.update(ufcstats_id="1111111111111111", name="Newer Event", source_url=url,
                        event_date=date(2026, 9, 26), fights=[])
            return data
        return event_data()

    async def fetch_fighter(browser, url):
        return parse_fighter_details(fixture("fighter_complete.html"), url) | {
            "ufcstats_id": url.rsplit("/", 1)[-1], "name": None,
        }

    monkeypatch.setattr(svc, "UFCStatsBrowser", FakeBrowser)
    monkeypatch.setattr(svc, "fetch_completed_events", fetch_completed_events)
    monkeypatch.setattr(svc, "fetch_event_with_fights", fetch_event_with_fights)
    monkeypatch.setattr(svc, "fetch_fighter", fetch_fighter)
    return state


def run_job(session_factory, **job_kwargs) -> ScrapeJob:
    with session_factory() as db:
        job = svc.create_job(db, **job_kwargs)
        db.commit()
        job_id = job.id
    asyncio.run(svc.run_ingestion_job(job_id, session_factory=session_factory))
    with session_factory() as db:
        job = db.get(ScrapeJob, job_id)
        db.expunge(job)
        return job


def test_job_records_page_failure_and_still_succeeds(session_factory, fake_crawl):
    job = run_job(session_factory, mode="backfill")

    assert job.status == "succeeded"
    assert (job.events_total, job.events_done) == (2, 2)
    assert job.fights_upserted == 3
    assert job.fighters_upserted == 6
    assert job.errors == 1
    assert json.loads(job.error_log)[0]["url"] == OTHER_EVENT_URL
    assert job.finished_at is not None

    with session_factory() as db:
        # Failed event was rolled back; UFC 331 is stored and profiles synced.
        assert db.scalar(select(Event).where(Event.ufcstats_id == "1111111111111111")) is None
        assert db.scalar(select(Event).where(Event.ufcstats_id == "8a0a35e7c74bebcc")).scraped_at is not None
        assert db.scalar(select(func.count()).select_from(Fighter).where(Fighter.last_synced_at.is_(None))) == 0


def test_incremental_job_only_fetches_missing_events(session_factory, fake_crawl):
    run_job(session_factory, mode="backfill", refresh_fighters=False)

    fake_crawl["fail"].clear()
    fake_crawl["fetched_events"].clear()
    job = run_job(session_factory, mode="incremental", refresh_fighters=False)

    assert job.status == "succeeded"
    assert job.events_total == 1
    assert fake_crawl["fetched_events"] == [OTHER_EVENT_URL]
    with session_factory() as db:
        assert db.scalar(select(Event).where(Event.ufcstats_id == "1111111111111111")).scraped_at is not None


def test_backfill_since_limits_window(session_factory, fake_crawl):
    fake_crawl["fail"].clear()
    job = run_job(session_factory, mode="backfill", since=date(2026, 9, 20), refresh_fighters=False)

    assert job.events_total == 1
    assert fake_crawl["fetched_events"] == [OTHER_EVENT_URL]


def test_events_are_ingested_oldest_first(session_factory, fake_crawl):
    fake_crawl["fail"].clear()
    run_job(session_factory, mode="backfill", refresh_fighters=False)

    assert fake_crawl["fetched_events"] == [EVENT_URL, OTHER_EVENT_URL]


def test_listing_failure_marks_job_failed(session_factory, fake_crawl, monkeypatch):
    async def broken_listing(*args, **kwargs):
        raise TimeoutError("listing did not render")

    monkeypatch.setattr(svc, "fetch_completed_events", broken_listing)
    job = run_job(session_factory, mode="backfill")

    assert job.status == "failed"
    assert "listing did not render" in job.error_log


def test_incremental_never_extends_backwards(session_factory, fake_crawl):
    """Only events on/after the oldest stored event are picked up."""
    fake_crawl["fail"].clear()
    run_job(session_factory, mode="backfill", since=date(2026, 9, 20), refresh_fighters=False)
    fake_crawl["fetched_events"].clear()

    job = run_job(session_factory, mode="incremental", refresh_fighters=False)

    assert job.status == "succeeded"
    assert job.events_total == 0  # UFC 331 (older) is left for a backfill
    assert fake_crawl["fetched_events"] == []


def test_incremental_on_empty_db_asks_for_backfill(session_factory, fake_crawl):
    job = run_job(session_factory, mode="incremental")

    assert job.status == "failed"
    assert "run a backfill first" in job.error_log
    assert fake_crawl["fetched_events"] == []


def test_fail_stale_jobs(db):
    svc.create_job(db, mode="incremental").status = "running"
    db.commit()

    assert svc.fail_stale_jobs(db) == 1
    db.commit()
    assert svc.get_active_job(db) is None
