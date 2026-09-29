"""Event lifecycle and synchronization operations."""

import re
from dataclasses import dataclass
from datetime import date

from sqlalchemy import select, update
from sqlalchemy.orm import Session, selectinload

#from src.api.v1.schemas.events import EventResponse
from src.api.v1.models import Event, Fight
from src.scrapers.upcoming_events_scraper import parse_event_date


@dataclass
class EventSyncResult:
    scraped: int
    inserted: int
    updated: int
    completed: int
    malformed: int


def _event_key(name: str) -> str:
    return re.sub(r"\s+", " ", name).strip().casefold()


def _find_existing(
    db: Session,
    name: str,
    event_date: date,
    location: str | None,
    source_url: str | None,
) -> Event | None:
    if source_url:
        event = db.scalar(select(Event).where(Event.source_url == source_url))
        if event is not None:
            return event

    events = db.scalars(select(Event)).all()
    key = _event_key(name)
    candidates = [
        event
        for event in events
        if _event_key(event.name) == key
        and event.event_date == event_date
        and (not location or event.location == location)
    ]
    # A name is only a safe fallback when it identifies one row.
    return candidates[0] if len(candidates) == 1 else None


def _text(value: object) -> str:
    return str(value).strip() if value is not None else ""


def sync_upcoming_events(
    db: Session, scraped_events: list[dict], as_of: date
) -> EventSyncResult:
    """Expire past events and upsert valid scraped events atomically.

    Scraped events with an unparseable date are ignored. The caller should
    perform scraping before entering this operation so scraper failures leave
    the database untouched.
    """
    completed = db.execute(
        update(Event)
        .where(Event.status == "upcoming", Event.event_date <= as_of)
        .values(status="completed")
    ).rowcount or 0

    inserted = 0
    updated = 0
    malformed = 0

    for raw_event in scraped_events:
        name = _text(raw_event.get("event_name") or raw_event.get("name"))
        raw_date = _text(raw_event.get("date") or raw_event.get("event_date"))
        parsed_date = parse_event_date(raw_date)
        if not name or parsed_date is None:
            malformed += 1
            continue

        source_url = raw_event.get("source_url")
        values = {
            "name": name,
            "event_date": parsed_date.date(),
            "location": _text(raw_event.get("location")) or None,
            "status": "upcoming",
        }
        event = _find_existing(
            db, name, parsed_date.date(), values["location"], source_url
        )
        if event is None:
            db.add(Event(**values, source_url=source_url))
            inserted += 1
            continue

        # Preserve a known URL if a source temporarily omits it.
        if source_url:
            values["source_url"] = source_url
        changed = any(getattr(event, key) != value for key, value in values.items())
        if changed:
            for key, value in values.items():
                setattr(event, key, value)
            updated += 1

    db.flush()
    return EventSyncResult(
        scraped=len(scraped_events),
        inserted=inserted,
        updated=updated,
        completed=completed,
        malformed=malformed,
    )


def list_upcoming_events(db: Session, as_of: date) -> list:
    return list(
        db.scalars(
            select(Event)
            .where(Event.status == "upcoming", Event.event_date > as_of)
            .order_by(Event.event_date.asc().nulls_last(), Event.name.asc())
        ).all()
    )


def list_events(
    db: Session,
    status: str | None = None,
    since: date | None = None,
    until: date | None = None,
    limit: int = 50,
    offset: int = 0,
) -> list[Event]:
    """Stored events, newest first, optionally filtered by status and date window."""
    query = select(Event)
    if status:
        query = query.where(Event.status == status)
    if since:
        query = query.where(Event.event_date >= since)
    if until:
        query = query.where(Event.event_date <= until)
    query = query.order_by(Event.event_date.desc().nulls_last(), Event.name.asc())
    return list(db.scalars(query.limit(limit).offset(offset)).all())


def get_event_with_fights(db: Session, event_id: int) -> Event | None:
    """An event with its fights (in card order) and their fighters loaded."""
    event = db.scalar(
        select(Event)
        .where(Event.id == event_id)
        .options(
            selectinload(Event.fights).selectinload(Fight.red_fighter),
            selectinload(Event.fights).selectinload(Fight.blue_fighter),
        )
    )
    return event
