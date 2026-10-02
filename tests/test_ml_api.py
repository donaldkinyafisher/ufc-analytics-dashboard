"""Model listing and training-job routes, with training itself faked."""

import json

import pytest
from fastapi.testclient import TestClient

from src.api.v1.database import get_db
from src.api.v1.main import app
from src.api.v1.models import TrainingJob
from src.api.v1.routers import ml_router
from src.api.v1.services import ml_services
from src.ml import training
from src.ml import utils as ml_utils


@pytest.fixture
def client(session_factory, monkeypatch):
    """Client on a fresh DB; background training is replaced by a recorder."""
    started: list[int] = []

    def override_get_db():
        with session_factory() as db:
            yield db

    monkeypatch.setattr(ml_router, "run_training_job", started.append)
    app.dependency_overrides[get_db] = override_get_db
    test_client = TestClient(app)
    test_client.started_jobs = started
    yield test_client
    app.dependency_overrides.clear()


@pytest.fixture
def artifacts(tmp_path, monkeypatch):
    """Empty model and metrics locations; returns a helper that adds a trained model."""
    metrics_path = tmp_path / "metrics.json"
    monkeypatch.setattr(ml_utils, "MODELS_DIR", tmp_path)
    monkeypatch.setattr(ml_utils, "MODEL_METRICS_PATH", metrics_path)

    def add(name: str, f1_score: float, trained: bool = True) -> None:
        if trained:
            (tmp_path / f"{name}.joblib").touch()
        metrics = json.loads(metrics_path.read_text()) if metrics_path.exists() else {}
        metrics[name] = {"accuracy": 0.6, "precision": 0.6, "recall": 0.6, "f1_score": f1_score,
                         "train_size": 7000, "test_size": 1750, "test_from": "2023-05-20",
                         "trained_at": "2026-10-02T12:00:00+00:00", "classification_report": {}}
        metrics_path.write_text(json.dumps(metrics))

    return add


def fake_results(models: list[str]) -> dict:
    return {
        name: {"accuracy": 0.61, "precision": 0.6, "recall": 0.62, "f1_score": 0.61,
               "train_size": 7000, "test_size": 1750, "test_from": "2023-05-20",
               "trained_at": "2026-10-02T12:00:00+00:00", "classification_report": {"red": {}}}
        for name in models
    }


# --- models ------------------------------------------------------------------

def test_list_models_marks_the_best_trained_model(client, artifacts):
    artifacts("xgboost", 0.60)
    artifacts("svm", 0.65)
    artifacts("knn", 0.90, trained=False)  # stale metrics without a model file

    models = {m["name"]: m for m in client.get("/api/v1/models").json()}

    assert list(models) == ml_utils.MODEL_NAMES
    assert [name for name, m in models.items() if m["is_best"]] == ["svm"]
    assert models["svm"]["metrics"]["f1_score"] == 0.65
    assert models["svm"]["metrics"]["test_from"] == "2023-05-20"
    assert models["knn"] == {"name": "knn", "trained": False, "is_best": False, "metrics": None}
    assert ml_services.best_model_name() == "svm"


def test_list_models_with_nothing_trained(client, artifacts):
    models = client.get("/api/v1/models").json()

    assert not any(m["trained"] or m["is_best"] for m in models)
    assert ml_services.best_model_name() is None


# --- training jobs -------------------------------------------------------------

def test_create_job_returns_202_with_location(client):
    response = client.post("/api/v1/models/training-jobs", json={"models": ["xgboost", "svm"]})

    assert response.status_code == 202
    job = response.json()
    assert (job["status"], job["models"], job["tune"], job["metrics"]) == ("queued", ["xgboost", "svm"], False, None)
    assert response.headers["location"].endswith(f"/api/v1/models/training-jobs/{job['id']}")
    assert client.started_jobs == [job["id"]]


def test_empty_body_trains_every_model(client):
    job = client.post("/api/v1/models/training-jobs", json={}).json()

    assert job["models"] == ml_utils.MODEL_NAMES


def test_second_job_conflicts_while_one_is_active(client):
    first = client.post("/api/v1/models/training-jobs", json={})
    second = client.post("/api/v1/models/training-jobs", json={})

    assert second.status_code == 409
    assert f"job {first.json()['id']}" in second.json()["detail"]


@pytest.mark.parametrize("payload", [{"models": ["transformer"]}, {"models": []}, {"tune": "maybe"}])
def test_invalid_job_requests_are_rejected(client, payload):
    assert client.post("/api/v1/models/training-jobs", json=payload).status_code == 422


def test_get_and_list_jobs(client, session_factory):
    created = [client.post("/api/v1/models/training-jobs", json={}).json()]
    with session_factory() as db:
        db.get(TrainingJob, created[0]["id"]).status = "failed"
        db.commit()
    created.append(client.post("/api/v1/models/training-jobs", json={"models": ["knn"]}).json())

    listed = client.get("/api/v1/models/training-jobs").json()
    assert [job["id"] for job in listed] == [created[1]["id"], created[0]["id"]]
    assert client.get(f"/api/v1/models/training-jobs/{created[1]['id']}").json()["models"] == ["knn"]
    assert client.get("/api/v1/models/training-jobs/999").status_code == 404


# --- job runner ----------------------------------------------------------------

def queue_job(session_factory, models: list[str], tune: bool = False) -> int:
    with session_factory() as db:
        job = ml_services.create_training_job(db, models=models, tune=tune)
        db.commit()
        return job.id


def test_successful_job_stores_metrics(session_factory, monkeypatch):
    calls = []

    def fake_train_model(models, tune):
        calls.append((models, tune))
        return fake_results(models)

    monkeypatch.setattr(training, "train_model", fake_train_model)
    job_id = queue_job(session_factory, ["xgboost"], tune=True)

    ml_services.run_training_job(job_id, session_factory=session_factory)

    with session_factory() as db:
        job = db.get(TrainingJob, job_id)
        assert job.status == "succeeded"
        assert job.started_at is not None and job.finished_at is not None
        metrics = json.loads(job.metrics)
    assert calls == [(["xgboost"], True)]
    assert metrics["xgboost"]["f1_score"] == 0.61
    assert "classification_report" not in metrics["xgboost"]


def test_failed_job_records_the_error(session_factory, monkeypatch):
    def broken_train_model(models, tune):
        raise ValueError("No usable training features")

    monkeypatch.setattr(training, "train_model", broken_train_model)
    job_id = queue_job(session_factory, ["svm"])

    ml_services.run_training_job(job_id, session_factory=session_factory)

    with session_factory() as db:
        job = db.get(TrainingJob, job_id)
        assert job.status == "failed"
        assert job.error == "ValueError: No usable training features"
        assert job.metrics is None


def test_finished_job_is_served_with_parsed_metrics(client, session_factory, monkeypatch):
    monkeypatch.setattr(training, "train_model", lambda models, tune: fake_results(models))
    job_id = queue_job(session_factory, ["random_forest"])
    ml_services.run_training_job(job_id, session_factory=session_factory)

    job = client.get(f"/api/v1/models/training-jobs/{job_id}").json()

    assert job["status"] == "succeeded"
    assert job["metrics"]["random_forest"]["accuracy"] == 0.61


def test_fail_stale_training_jobs(db):
    db.add_all([TrainingJob(models="[]", status="running"), TrainingJob(models="[]", status="succeeded")])
    db.flush()

    assert ml_services.fail_stale_training_jobs(db) == 1
    assert ml_services.get_active_training_job(db) is None
