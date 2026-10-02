"""Event lifecycle and synchronization operations."""

import asyncio
import logging
import re
from dataclasses import dataclass, field
from datetime import date, datetime

from sqlalchemy import select, update
from sqlalchemy.orm import Session, selectinload

#from src.api.v1.schemas.events import EventResponse
from src.api.v1.models import Event, Fight, utc_now
from src.api.v1.services.ingestion_services import upsert_fighter_stub
from src.scrapers.historical_scraper import fetch_event, ufcstats_id
from src.scrapers.ufcstats_browser import UFCStatsBrowser
from src.scrapers.upcoming_events_scraper import parse_event_date

logger = logging.getLogger(__name__)


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
    db: Session, scraped_events: list[dict], as_of: date, synced_at: datetime | None = None
) -> EventSyncResult:
    """Expire past events and upsert valid scraped events atomically.

    Every upserted event gets scraped_at = synced_at, which records when the
    upcoming sync last saw it. Expired events have scraped_at cleared so
    historical ingestion picks them up for their results.

    Scraped events with an unparseable date are ignored. The caller should
    perform scraping before entering this operation so scraper failures leave
    the database untouched.
    """
    synced_at = synced_at or utc_now()
    completed = db.execute(
        update(Event)
        .where(Event.status == "upcoming", Event.event_date <= as_of)
        .values(status="completed", scraped_at=None)
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
            db.add(Event(**values, source_url=source_url, scraped_at=synced_at))
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
        event.scraped_at = synced_at

    db.flush()
    return EventSyncResult(
        scraped=len(scraped_events),
        inserted=inserted,
        updated=updated,
        completed=completed,
        malformed=malformed,
    )


@dataclass
class CardSyncResult:
    fights: int = 0
    failed_events: list[str] = field(default_factory=list)


def _is_complete_bout(bout: dict) -> bool:
    fighters = bout.get("fighters") or []
    return bool(bout.get("ufcstats_id")) and len(fighters) == 2 and all(
        fighter.get("ufcstats_id") and fighter.get("name") for fighter in fighters
    )


def store_upcoming_card(db: Session, event: Event, card: list[dict]) -> list[Fight]:
    """Store an upcoming event's card: fighters, weight class and bout type.

    Fighters are matched by ufcstats id; unknown fighters become stubs whose
    profiles are filled by the next ingestion run. No statistics are stored.
    Bouts that have left the card are deleted, unless the card came back
    empty, which is more likely a page problem than every bout being cancelled.
    """
    if event.status != "upcoming":
        raise ValueError(f"Event {event.id} is {event.status}, not upcoming")
    if event.ufcstats_id is None:
        event.ufcstats_id = ufcstats_id(event.source_url)

    bouts = [bout for bout in card if _is_complete_bout(bout)]
    if len(bouts) < len(card):
        logger.warning("Skipped %d malformed bouts on %s", len(card) - len(bouts), event.name)

    # Delete first, so a bout re-listed under a new id doesn't clash with
    # its old row on (event, red, blue).
    if bouts:
        on_card = {bout["ufcstats_id"] for bout in bouts}
        for fight in db.scalars(select(Fight).where(Fight.event_id == event.id)).all():
            if fight.ufcstats_id not in on_card:
                db.delete(fight)
        db.flush()

    fights = []
    for bout in bouts:
        red, blue = (upsert_fighter_stub(db, fighter) for fighter in bout["fighters"])
        values = {
            "event_id": event.id,
            "red_fighter_id": red.id,
            "blue_fighter_id": blue.id,
            "bout_order": bout.get("bout_order"),
            "weight_class": bout.get("weight_class"),
            "bout_type": bout.get("bout_type"),
            "is_title_bout": bool(bout.get("is_title_bout")),
            "source_url": bout.get("source_url"),
        }
        fight = db.scalar(select(Fight).where(Fight.ufcstats_id == bout["ufcstats_id"]))
        if fight is None:
            fight = Fight(ufcstats_id=bout["ufcstats_id"], **values)
            db.add(fight)
        else:
            for key, value in values.items():
                setattr(fight, key, value)
        fights.append(fight)
    db.flush()
    return fights


async def sync_upcoming_cards(db: Session, events: list[Event]) -> CardSyncResult:
    """Fetch each event's page and store its card.

    A page that fails to load leaves that event's stored card as it was.
    """
    result = CardSyncResult()
    targets = [event for event in events if event.source_url]
    async with UFCStatsBrowser() as browser:
        pages = await asyncio.gather(
            *(fetch_event(browser, event.source_url) for event in targets),
            return_exceptions=True,
        )
    for event, page in zip(targets, pages):
        if isinstance(page, Exception):
            logger.warning("Could not fetch card for %s: %s", event.name, page)
            result.failed_events.append(event.name)
            continue
        result.fights += len(store_upcoming_card(db, event, page["fights"]))
    return result


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
