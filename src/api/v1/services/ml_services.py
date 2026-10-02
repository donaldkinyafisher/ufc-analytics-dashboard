"""Trained models and training jobs.

Training is CPU-bound and can take minutes (much longer with tuning), so it
runs as a background job: create_training_job queues it and
run_training_job runs it to completion, recording the outcome on the job.
"""

import json
import logging
from pathlib import Path

from sqlalchemy import select, update
from sqlalchemy.orm import Session, sessionmaker

from src.api.v1.database import SessionLocal
from src.api.v1.models import TrainingJob, utc_now
from src.ml import utils as ml_utils

logger = logging.getLogger(__name__)

ACTIVE_STATUSES = ("queued", "running")
# Metrics kept on a job; the full classification report stays in the metrics file.
JOB_METRIC_KEYS = (
    "accuracy", "precision", "recall", "f1_score",
    "train_size", "test_size", "test_from", "trained_at",
)


def _load_metrics(path: Path) -> dict:
    try:
        return ml_utils.load_model_metrics(path)
    except FileNotFoundError:
        return {}


def list_models(models_dir: Path | None = None, metrics_path: Path | None = None) -> list[dict]:
    """Every supported model, whether it is trained, its test metrics and the best one.

    Paths default to the artifact locations in src.ml.utils.
    """
    models_dir = models_dir or ml_utils.MODELS_DIR
    metrics = _load_metrics(metrics_path or ml_utils.MODEL_METRICS_PATH)
    summaries = []
    for name in ml_utils.MODEL_NAMES:
        trained = (models_dir / f"{name}{ml_utils.MODEL_FILE_SUFFIX}").is_file()
        summaries.append({
            "name": name,
            "trained": trained,
            "metrics": metrics.get(name) if trained else None,
            "is_best": False,
        })
    scored = [s for s in summaries if s["metrics"] and s["metrics"].get("f1_score") is not None]
    if scored:
        max(scored, key=lambda s: s["metrics"]["f1_score"])["is_best"] = True
    return summaries


def best_model_name(models_dir: Path | None = None, metrics_path: Path | None = None) -> str | None:
    """The trained model with the highest test F1, or None if none is trained."""
    return next((s["name"] for s in list_models(models_dir, metrics_path) if s["is_best"]), None)


def create_training_job(db: Session, models: list[str], tune: bool = False) -> TrainingJob:
    job = TrainingJob(models=json.dumps(models), tune=tune, status="queued")
    db.add(job)
    db.flush()
    return job


def get_active_training_job(db: Session) -> TrainingJob | None:
    return db.scalar(
        select(TrainingJob).where(TrainingJob.status.in_(ACTIVE_STATUSES)).order_by(TrainingJob.id)
    )


def fail_stale_training_jobs(db: Session) -> int:
    """Mark jobs left queued/running by a previous process as failed."""
    return db.execute(
        update(TrainingJob)
        .where(TrainingJob.status.in_(ACTIVE_STATUSES))
        .values(status="failed", finished_at=utc_now(), error="Interrupted: API process restarted")
    ).rowcount or 0


def run_training_job(job_id: int, session_factory: sessionmaker[Session] = SessionLocal) -> None:
    """Run a queued TrainingJob. Never raises; the outcome is on the job.

    A plain function, so FastAPI runs it in a worker thread and training
    doesn't block the event loop.
    """
    # Imported here so the API only loads PyTorch and friends when it trains.
    from src.ml import training

    with session_factory() as db:
        job = db.get(TrainingJob, job_id)
        if job is None:
            raise ValueError(f"TrainingJob {job_id} not found")
        job.status = "running"
        job.started_at = utc_now()
        db.commit()

        try:
            results = training.train_model(models=json.loads(job.models), tune=job.tune)
            job.metrics = json.dumps({
                name: {key: result[key] for key in JOB_METRIC_KEYS if key in result}
                for name, result in results.items()
            })
            job.status = "succeeded"
        except Exception as exc:
            logger.exception("Training job %s failed", job_id)
            job.error = f"{type(exc).__name__}: {exc}"
            job.status = "failed"
        finally:
            job.finished_at = utc_now()
            db.commit()
