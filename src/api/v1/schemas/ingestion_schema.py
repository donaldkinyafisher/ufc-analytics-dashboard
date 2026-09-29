import json
from datetime import UTC, date, datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, field_validator, model_validator


class IngestionJobCreate(BaseModel):
    """Start a historical ingestion job.

    backfill: completed events with since <= event_date <= until
        (defaults: from the first event on ufcstats up to today).
    incremental: events not yet ingested, dated on/after the oldest
        ingested event. Takes no date window.
    """

    mode: Literal["backfill", "incremental"] = "incremental"
    since: date | None = None
    until: date | None = None
    refresh_existing: bool = False
    refresh_fighters: bool = True

    @model_validator(mode="after")
    def check_window(self) -> "IngestionJobCreate":
        if self.mode == "incremental" and (self.since or self.until or self.refresh_existing):
            raise ValueError("since, until and refresh_existing only apply to backfill jobs")
        today = datetime.now(UTC).date()
        if self.since and self.since > today:
            raise ValueError("since cannot be in the future")
        if self.since and self.until and self.since > self.until:
            raise ValueError("since must be on or before until")
        return self


class IngestionError(BaseModel):
    url: str | None
    error: str


class IngestionJobResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    mode: str
    since: date | None
    until: date | None
    refresh_existing: bool
    refresh_fighters: bool
    status: str
    events_total: int
    events_done: int
    fights_upserted: int
    fighters_upserted: int
    errors: int
    error_log: list[IngestionError]
    created_at: datetime
    started_at: datetime | None
    finished_at: datetime | None

    @field_validator("error_log", mode="before")
    @classmethod
    def parse_error_log(cls, value: object) -> object:
        if value is None:
            return []
        if isinstance(value, str):
            return json.loads(value)
        return value
