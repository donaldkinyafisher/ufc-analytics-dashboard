from datetime import date, datetime

from pydantic import BaseModel, ConfigDict

from src.api.v1.schemas.fights_schema import FightSummary


class EventResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    ufcstats_id: str | None
    name: str
    event_date: date | None
    location: str | None
    source_url: str | None
    status: str
    scraped_at: datetime | None


class EventDetailResponse(EventResponse):
    fights: list[FightSummary]


class EventSyncResponse(BaseModel):
    scraped: int
    inserted: int
    updated: int
    completed: int
    malformed: int
    synchronized_at: datetime
    events: list[EventResponse]
