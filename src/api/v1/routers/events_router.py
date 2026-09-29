"""
REST API for UFC upcoming events.

Wraps the Playwright scraper in a FastAPI service with a simple in-memory
TTL cache, since scraping takes a few seconds and the source page only
changes occasionally (not something you want to re-run on every request).

Run:
    pip install fastapi uvicorn playwright
    playwright install chromium
    uvicorn api:app --reload

Then:
GET http://localhost:8000/events/upcoming
POST http://localhost:8000/events/upcoming/sync
    GET http://localhost:8000/health
"""

import asyncio
import datetime  #import date, datetime, timezone
from typing import Annotated, Literal

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from src.api.v1.database import get_db
from src.api.v1.schemas.events_schema import (
    EventDetailResponse,
    EventResponse,
    EventSyncResponse,
)
from src.api.v1.services.events_services import (
    get_event_with_fights,
    list_events,
    list_upcoming_events,
    sync_upcoming_events,
)
from src.scrapers.upcoming_events_scraper import scrape_upcoming_events

router = APIRouter()

# Only one browser/database synchronization may run per API process.
_refresh_lock = asyncio.Lock()


@router.get("", response_model=list[EventResponse])
def get_events(
    db: Annotated[Session, Depends(get_db)],
    status: Annotated[Literal["upcoming", "completed"] | None, Query()] = None,
    since: datetime.date | None = None,
    until: datetime.date | None = None,
    limit: Annotated[int, Query(ge=1, le=500)] = 50,
    offset: Annotated[int, Query(ge=0)] = 0,
):
    """Stored events, newest first."""
    return list_events(db, status=status, since=since, until=until, limit=limit, offset=offset)


@router.get("/upcoming", response_model=list[EventResponse])
def get_upcoming_events(db: Annotated[Session, Depends(get_db)]):
    utc_now = datetime.datetime.now(datetime.UTC)
    return list_upcoming_events(db, utc_now.date())


@router.post("/upcoming/sync", response_model=EventSyncResponse)
async def sync_events(db: Annotated[Session, Depends(get_db)]):
    """
    Runs the scraper to get update upcoming events. 
    Inserts new events into the events table, and updates already stored events to past,
    if event_date < current_date.
    """
    async with _refresh_lock:
        try:
            raw_events = await scrape_upcoming_events(headless=True)
        except Exception as exc:
            raise HTTPException(status_code=502, detail=f"Could not scrape UFCStats: {exc}") from exc

        utc_now = datetime.datetime.now(datetime.UTC)
        today = utc_now.date()
        try:
            result = sync_upcoming_events(db, raw_events, today)
            db.commit()
        except Exception:
            db.rollback()
            raise

        events = list_upcoming_events(db, today)
        synchronized_at = datetime.datetime.now(datetime.UTC)
        return EventSyncResponse(
            **result.__dict__, synchronized_at=synchronized_at, events=events
        )


@router.get("/{event_id}", response_model=EventDetailResponse)
def get_event(event_id: int, db: Annotated[Session, Depends(get_db)]):
    """An event with its fights in card order (1 = main event)."""
    event = get_event_with_fights(db, event_id)
    if event is None:
        raise HTTPException(status_code=404, detail="Event not found")
    return event
