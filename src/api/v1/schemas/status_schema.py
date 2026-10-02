from datetime import date

from pydantic import BaseModel

from src.api.v1.schemas.ingestion_schema import IngestionJobResponse
from src.api.v1.schemas.ml_schema import TrainingJobResponse


class StatusResponse(BaseModel):
    """Stored data, running jobs and trained models.

    Counts cover fully ingested (completed and scraped) events only.
    needs_setup is true until an ingestion job has succeeded with events stored.
    """

    events_ingested: int
    fights: int
    fighters: int
    oldest_event_date: date | None
    newest_event_date: date | None
    active_ingestion_job: IngestionJobResponse | None
    last_ingestion_job: IngestionJobResponse | None
    trained_models: int
    active_training_job: TrainingJobResponse | None
    needs_setup: bool
