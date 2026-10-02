"""What is stored and what is running, for the Streamlit setup gate.

needs_setup stays true until an ingestion job has succeeded: each event is
committed as soon as it is scraped, so counting ingested events alone would
open the app partway through the first backfill, or after one that failed.
"""

from pathlib import Path

from sqlalchemy import exists, func, select
from sqlalchemy.orm import Session

from src.api.v1.models import Event, Fight, Fighter, ScrapeJob
from src.api.v1.services.ingestion_services import get_active_job, ingested_filter
from src.api.v1.services.ml_services import get_active_training_job, list_models


def get_status(db: Session, models_dir: Path | None = None, metrics_path: Path | None = None) -> dict:
    events_ingested, oldest, newest = db.execute(
        select(func.count(Event.id), func.min(Event.event_date), func.max(Event.event_date))
        .where(ingested_filter())
    ).one()
    fights = db.scalar(select(func.count(Fight.id)).join(Event).where(ingested_filter()))
    fighters = db.scalar(select(func.count(Fighter.id)))

    active_ingestion = get_active_job(db)
    last_ingestion = db.scalar(select(ScrapeJob).order_by(ScrapeJob.id.desc()).limit(1))
    any_succeeded = db.scalar(select(exists().where(ScrapeJob.status == "succeeded")))

    return {
        "events_ingested": events_ingested,
        "fights": fights,
        "fighters": fighters,
        "oldest_event_date": oldest,
        "newest_event_date": newest,
        "active_ingestion_job": active_ingestion,
        "last_ingestion_job": last_ingestion,
        "trained_models": sum(m["trained"] for m in list_models(models_dir, metrics_path)),
        "active_training_job": get_active_training_job(db),
        "needs_setup": events_ingested == 0 or not any_succeeded,
    }
