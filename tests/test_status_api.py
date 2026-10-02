"""GET /api/v1/status: stored data, running jobs and the setup gate."""

import json
from datetime import UTC, date, datetime

import pytest
from fastapi.testclient import TestClient

from src.api.v1.database import get_db
from src.api.v1.main import app
from src.api.v1.models import Event, Fight, Fighter, ScrapeJob, TrainingJob
from src.ml import utils as ml_utils


@pytest.fixture
def client(session_factory):
    def override_get_db():
        with session_factory() as db:
            yield db

    app.dependency_overrides[get_db] = override_get_db
    yield TestClient(app)
    app.dependency_overrides.clear()


@pytest.fixture(autouse=True)
def models_dir(tmp_path, monkeypatch):
    """No trained models unless a test adds a model file here."""
    monkeypatch.setattr(ml_utils, "MODELS_DIR", tmp_path)
    monkeypatch.setattr(ml_utils, "MODEL_METRICS_PATH", tmp_path / "metrics.json")
    return tmp_path


SCRAPED = datetime(2026, 10, 1, tzinfo=UTC)


def add_events(db) -> None:
    """Two ingested events (three fights), one unscraped and one upcoming."""
    alpha, bravo, charlie = Fighter(name="Alpha"), Fighter(name="Bravo"), Fighter(name="Charlie")
    old = Event(name="UFC 1", event_date=date(2021, 1, 1), status="completed", scraped_at=SCRAPED)
    new = Event(name="UFC 2", event_date=date(2025, 6, 1), status="completed", scraped_at=SCRAPED)
    unscraped = Event(name="UFC 3", event_date=date(2025, 7, 1), status="completed")
    upcoming = Event(name="UFC 4", event_date=date(2030, 1, 1), status="upcoming", scraped_at=SCRAPED)
    db.add_all([alpha, bravo, charlie, old, new, unscraped, upcoming])
    for event, red, blue in ((old, alpha, bravo), (new, alpha, bravo), (new, alpha, charlie),
                             (upcoming, alpha, bravo)):
        db.add(Fight(event=event, red_fighter=red, blue_fighter=blue))


def add_job(db, status: str) -> ScrapeJob:
    job = ScrapeJob(mode="backfill", status=status)
    db.add(job)
    db.flush()
    return job


def test_empty_database_needs_setup(client):
    status = client.get("/api/v1/status").json()

    assert status == {
        "events_ingested": 0,
        "fights": 0,
        "fighters": 0,
        "oldest_event_date": None,
        "newest_event_date": None,
        "active_ingestion_job": None,
        "last_ingestion_job": None,
        "trained_models": 0,
        "active_training_job": None,
        "needs_setup": True,
    }


def test_ingested_data_opens_the_app(client, session_factory):
    with session_factory() as db:
        add_events(db)
        job = add_job(db, "succeeded")
        db.commit()
        job_id = job.id

    status = client.get("/api/v1/status").json()

    assert status["events_ingested"] == 2  # unscraped and upcoming events don't count
    assert status["fights"] == 3
    assert status["fighters"] == 3
    assert status["oldest_event_date"] == "2021-01-01"
    assert status["newest_event_date"] == "2025-06-01"
    assert status["last_ingestion_job"]["id"] == job_id
    assert status["active_ingestion_job"] is None
    assert status["needs_setup"] is False


def test_first_backfill_still_running_keeps_setup(client, session_factory):
    with session_factory() as db:
        add_events(db)  # events committed so far by the running job
        job = add_job(db, "running")
        db.commit()
        job_id = job.id

    status = client.get("/api/v1/status").json()

    assert status["events_ingested"] == 2
    assert status["active_ingestion_job"]["id"] == job_id
    assert status["needs_setup"] is True


def test_later_job_running_does_not_reopen_setup(client, session_factory):
    with session_factory() as db:
        add_events(db)
        add_job(db, "succeeded")
        add_job(db, "running")
        db.commit()

    assert client.get("/api/v1/status").json()["needs_setup"] is False


def test_failed_job_with_nothing_ingested_needs_setup(client, session_factory):
    with session_factory() as db:
        add_job(db, "failed")
        db.commit()

    status = client.get("/api/v1/status").json()

    assert status["last_ingestion_job"]["status"] == "failed"
    assert status["needs_setup"] is True


def test_failed_first_backfill_with_partial_data_keeps_setup(client, session_factory):
    with session_factory() as db:
        add_events(db)  # stored before the job failed
        add_job(db, "failed")
        db.commit()

    status = client.get("/api/v1/status").json()

    assert status["events_ingested"] == 2
    assert status["needs_setup"] is True


def test_trained_models_and_active_training_job(client, session_factory, models_dir):
    (models_dir / f"xgboost{ml_utils.MODEL_FILE_SUFFIX}").touch()
    (models_dir / f"svm{ml_utils.MODEL_FILE_SUFFIX}").touch()
    with session_factory() as db:
        job = TrainingJob(models=json.dumps(["knn"]), status="running")
        db.add(job)
        db.commit()
        job_id = job.id

    status = client.get("/api/v1/status").json()

    assert status["trained_models"] == 2
    assert status["active_training_job"]["id"] == job_id
    assert status["active_training_job"]["models"] == ["knn"]
