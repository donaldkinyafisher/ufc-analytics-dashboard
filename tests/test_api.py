"""API route tests against an in-memory database seeded from fixture pages."""

import pytest
from fastapi.testclient import TestClient

from src.api.v1.database import get_db
from src.api.v1.main import app
from src.api.v1.models import Event
from src.api.v1.routers import events_router, ingestion_router
from src.api.v1.services import events_services
from src.api.v1.services import ingestion_services as svc
from src.scrapers.historical_scraper import parse_event_details, parse_fighter_details
from tests.helpers import VAN_URL, FakeBrowser, event_data, fixture


@pytest.fixture
def client(session_factory, monkeypatch):
    """Client on a fresh DB; background ingestion is replaced by a recorder."""
    started: list[int] = []

    async def fake_run(job_id: int) -> None:
        started.append(job_id)

    def override_get_db():
        with session_factory() as db:
            yield db

    monkeypatch.setattr(ingestion_router, "run_ingestion_job", fake_run)
    app.dependency_overrides[get_db] = override_get_db
    # Not used as a context manager, so the lifespan (which touches the real DB) doesn't run.
    test_client = TestClient(app)
    test_client.started_jobs = started
    yield test_client
    app.dependency_overrides.clear()


@pytest.fixture
def seeded(session_factory):
    """UFC 331 with three fights and Joshua Van's profile."""
    with session_factory() as db:
        event, fights = svc.ingest_event(db, event_data())
        svc.apply_fighter_profile(db, parse_fighter_details(fixture("fighter_complete.html"), VAN_URL))
        db.add(Event(name="UFC 999: Future", status="upcoming"))
        db.commit()
        return {
            "event_id": event.id,
            "title_fight_id": fights[0].id,
            "van_id": fights[0].red_fighter_id,
        }


# --- ingestion jobs --------------------------------------------------------

def test_create_job_returns_202_with_location(client):
    response = client.post("/api/v1/ingestion/jobs", json={"mode": "backfill", "since": "2025-09-28"})

    assert response.status_code == 202
    job = response.json()
    assert job["status"] == "queued"
    assert job["mode"] == "backfill"
    assert job["since"] == "2025-09-28"
    assert job["error_log"] == []
    assert response.headers["location"].endswith(f"/api/v1/ingestion/jobs/{job['id']}")
    assert client.started_jobs == [job["id"]]


def test_second_job_conflicts_while_one_is_active(client):
    first = client.post("/api/v1/ingestion/jobs", json={"mode": "incremental"})
    second = client.post("/api/v1/ingestion/jobs", json={"mode": "incremental"})

    assert first.status_code == 202
    assert second.status_code == 409
    assert f"job {first.json()['id']}" in second.json()["detail"]


@pytest.mark.parametrize(
    "payload",
    [
        {"mode": "incremental", "since": "2025-01-01"},
        {"mode": "backfill", "since": "2026-01-01", "until": "2025-01-01"},
        {"mode": "backfill", "since": "2999-01-01"},
        {"mode": "rebuild"},
    ],
)
def test_invalid_job_requests_are_rejected(client, payload):
    assert client.post("/api/v1/ingestion/jobs", json=payload).status_code == 422


def test_get_and_list_jobs(client, session_factory):
    with session_factory() as db:
        job = svc.create_job(db, mode="backfill")
        job.status = "succeeded"
        job.errors = 1
        job.error_log = '[{"url": "http://ufcstats.com/event-details/x", "error": "TimeoutError: slow"}]'
        db.commit()
        job_id = job.id

    detail = client.get(f"/api/v1/ingestion/jobs/{job_id}")
    assert detail.status_code == 200
    assert detail.json()["error_log"][0]["error"] == "TimeoutError: slow"

    listing = client.get("/api/v1/ingestion/jobs")
    assert [j["id"] for j in listing.json()] == [job_id]

    assert client.get("/api/v1/ingestion/jobs/999").status_code == 404


# --- events ----------------------------------------------------------------

def test_list_events_filters_by_status_and_date(client, seeded):
    completed = client.get("/api/v1/events", params={"status": "completed"}).json()
    assert [e["name"] for e in completed] == ["UFC 331: Van vs. Pantoja 2"]
    assert completed[0]["scraped_at"] is not None

    assert len(client.get("/api/v1/events").json()) == 2
    assert client.get("/api/v1/events", params={"since": "2026-09-20"}).json() == []


def test_get_event_lists_fights_in_card_order(client, seeded):
    response = client.get(f"/api/v1/events/{seeded['event_id']}")

    assert response.status_code == 200
    fights = response.json()["fights"]
    assert [f["bout_order"] for f in fights] == [1, 2, 3]
    assert fights[0]["red_fighter"]["name"] == "Joshua Van"
    assert fights[0]["event"]["name"] == "UFC 331: Van vs. Pantoja 2"
    assert client.get("/api/v1/events/999").status_code == 404


def test_upcoming_route_still_resolves(client, seeded):
    # /upcoming must not be captured by /{event_id}.
    assert client.get("/api/v1/events/upcoming").status_code == 200


# --- fights ----------------------------------------------------------------

def test_sync_stamps_scraped_at_on_upcoming_events(client, monkeypatch):
    async def fake_scrape(headless=True):
        return [{"event_name": "UFC 999: Future", "date": "January 01, 2099",
                 "location": "Las Vegas, Nevada, USA",
                 "source_url": "http://ufcstats.com/event-details/f"}]

    async def fake_fetch_event(browser, url):
        return parse_event_details(fixture("event_upcoming.html"), url)

    monkeypatch.setattr(events_router, "scrape_upcoming_events", fake_scrape)
    monkeypatch.setattr(events_services, "UFCStatsBrowser", FakeBrowser)
    monkeypatch.setattr(events_services, "fetch_event", fake_fetch_event)
    synced = client.post("/api/v1/events/upcoming/sync")

    assert synced.status_code == 200
    assert (synced.json()["fights"], synced.json()["card_failures"]) == (14, [])
    upcoming = client.get("/api/v1/events/upcoming").json()
    assert [e["name"] for e in upcoming] == ["UFC 999: Future"]
    assert upcoming[0]["scraped_at"] is not None
    card = client.get(f"/api/v1/events/{upcoming[0]['id']}").json()["fights"]
    assert len(card) == 14
    assert card[0]["bout_type"] == "UFC Women's Flyweight Title Bout"


def test_get_fight_includes_statistics(client, seeded):
    response = client.get(f"/api/v1/fights/{seeded['title_fight_id']}")

    assert response.status_code == 200
    fight = response.json()
    assert fight["winner_side"] == "red"
    assert fight["method_details"].startswith("Sal D'amato")
    red = next(s for s in fight["statistics"] if s["corner"] == "red")
    assert (red["sig_str_landed"], red["sig_str_attempted"]) == (181, 305)
    assert red["result"] == "W"
    assert client.get("/api/v1/fights/999").status_code == 404


# --- fighters --------------------------------------------------------------

def test_search_fighters_by_name(client, seeded):
    results = client.get("/api/v1/fighters", params={"name": "van"}).json()
    assert [f["name"] for f in results] == ["Joshua Van"]


def test_get_fighter_profile_with_career_stats(client, seeded):
    fighter = client.get(f"/api/v1/fighters/{seeded['van_id']}").json()

    assert (fighter["wins"], fighter["losses"]) == (18, 2)
    assert fighter["height_in"] == 65.0
    assert fighter["dob"] == "2001-10-10"
    assert fighter["career_stats"]["strikes_landed_per_minute"] == 8.26
    assert client.get("/api/v1/fighters/999").status_code == 404


def test_get_fighter_fights(client, seeded):
    fights = client.get(f"/api/v1/fighters/{seeded['van_id']}/fights").json()

    assert len(fights) == 1
    assert fights[0]["blue_fighter"]["name"] == "Alexandre Pantoja"
