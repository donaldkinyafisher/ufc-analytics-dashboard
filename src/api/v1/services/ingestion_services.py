"""Historical ingestion: completed events, fights and fighters from ufcstats.

The DB functions (upsert_event, upsert_fighter_stub, store_fight,
apply_fighter_profile, ingest_event) take parsed dicts from
src.scrapers.historical_scraper and a Session; they flush but never commit.

run_ingestion_job drives a whole ScrapeJob: it lists completed events,
ingests each event in its own transaction (so an interrupted run resumes
where it stopped), then refreshes the profiles of fighters it touched.
A page failure rolls back only that event and is recorded on the job; the
job is marked failed only when the crawl itself cannot run, e.g. the event
list fails to load or the browser dies (as on API shutdown).
"""

import asyncio
import json
import logging
from datetime import UTC, date, datetime

from sqlalchemy import func, select, update
from sqlalchemy.orm import Session, sessionmaker

from src.api.v1.config import get_settings
from src.api.v1.database import SessionLocal
from src.api.v1.models import (
    Event,
    Fight,
    Fighter,
    FighterCareerStats,
    FightStatistic,
    ScrapeJob,
)
from src.scrapers.historical_scraper import (
    fetch_completed_events,
    fetch_event_with_fights,
    fetch_fighter,
)
from src.scrapers.ufcstats_browser import BrowserClosedError, UFCStatsBrowser

logger = logging.getLogger(__name__)


def utc_now() -> datetime:
    return datetime.now(UTC)


def _apply(obj: object, values: dict, keep_existing: bool = False) -> bool:
    """Set changed attributes on obj; return whether anything changed.

    With keep_existing, None values never overwrite stored data.
    """
    changed = False
    for key, value in values.items():
        if keep_existing and value is None:
            continue
        if getattr(obj, key) != value:
            setattr(obj, key, value)
            changed = True
    return changed


# ---------------------------------------------------------------------------
# DB functions
# ---------------------------------------------------------------------------

def upsert_event(db: Session, data: dict) -> Event:
    """Insert or update a completed event.

    Matches on ufcstats_id, then source_url, so an event first stored by the
    upcoming-events sync is promoted to completed rather than duplicated.
    """
    event = None
    if data.get("ufcstats_id"):
        event = db.scalar(select(Event).where(Event.ufcstats_id == data["ufcstats_id"]))
    if event is None and data.get("source_url"):
        event = db.scalar(select(Event).where(Event.source_url == data["source_url"]))

    values = {
        "ufcstats_id": data.get("ufcstats_id"),
        "name": data.get("name"),
        "event_date": data.get("event_date"),
        "location": data.get("location"),
        "source_url": data.get("source_url"),
        "status": "completed",
    }
    if event is None:
        event = Event(**values)
        db.add(event)
    else:
        _apply(event, values, keep_existing=True)
    db.flush()
    return event


def _find_fighter(db: Session, ufcstats_id: str | None, profile_url: str | None) -> Fighter | None:
    if ufcstats_id:
        fighter = db.scalar(select(Fighter).where(Fighter.ufcstats_id == ufcstats_id))
        if fighter is not None:
            return fighter
    if profile_url:
        return db.scalar(select(Fighter).where(Fighter.profile_url == profile_url))
    return None


def upsert_fighter_stub(db: Session, data: dict) -> Fighter:
    """Get or create a fighter from a fight page (name, nickname, profile URL only)."""
    if not data.get("ufcstats_id") or not data.get("name"):
        raise ValueError(f"Fighter without ufcstats id or name: {data!r}")

    values = {
        "ufcstats_id": data["ufcstats_id"],
        "name": data["name"],
        "nickname": data.get("nickname"),
        "profile_url": data.get("profile_url"),
    }
    fighter = _find_fighter(db, data["ufcstats_id"], data.get("profile_url"))
    if fighter is None:
        fighter = Fighter(**values)
        db.add(fighter)
        # Flush so a fighter appearing twice on one card (early tournaments)
        # is found by the next lookup in this transaction.
        db.flush()
    else:
        _apply(fighter, values, keep_existing=True)
    return fighter


def store_fight(db: Session, event: Event, data: dict) -> Fight:
    """Upsert a fight and both fighters' statistics rows."""
    fighters = data.get("fighters") or []
    if len(fighters) != 2:
        raise ValueError(f"Expected 2 fighters, got {len(fighters)} for {data.get('source_url')}")

    red = upsert_fighter_stub(db, fighters[0])
    blue = upsert_fighter_stub(db, fighters[1])

    values = {
        "ufcstats_id": data.get("ufcstats_id"),
        "event_id": event.id,
        "red_fighter_id": red.id,
        "blue_fighter_id": blue.id,
        "bout_order": data.get("bout_order"),
        "weight_class": data.get("weight_class"),
        "bout_type": data.get("bout_type"),
        "is_title_bout": bool(data.get("is_title_bout")),
        "source_url": data.get("source_url"),
        "winner_side": data.get("winner_side"),
        "method": data.get("method"),
        "method_details": data.get("method_details"),
        "round": data.get("round"),
        "time": data.get("time"),
        "time_format": data.get("time_format"),
        "referee": data.get("referee"),
    }
    fight = None
    if values["ufcstats_id"]:
        fight = db.scalar(select(Fight).where(Fight.ufcstats_id == values["ufcstats_id"]))
    if fight is None:
        fight = Fight(**values)
        db.add(fight)
        db.flush()
    else:
        _apply(fight, values)

    stats = data.get("stats") or {}
    for corner, fighter, fighter_data in (("red", red, fighters[0]), ("blue", blue, fighters[1])):
        stat_values = {
            "fighter_id": fighter.id,
            "result": fighter_data.get("result"),
            **(stats.get(corner) or {}),
        }
        row = db.scalar(
            select(FightStatistic).where(
                FightStatistic.fight_id == fight.id, FightStatistic.corner == corner
            )
        )
        if row is None:
            db.add(FightStatistic(fight_id=fight.id, corner=corner, **stat_values))
        else:
            _apply(row, stat_values)
    db.flush()
    return fight


def ingest_event(db: Session, data: dict) -> tuple[Event, list[Fight]]:
    """Store an event with all its fights and mark it as scraped."""
    event = upsert_event(db, data)
    fights = [store_fight(db, event, fight) for fight in data.get("fights", [])]
    event.scraped_at = utc_now()
    db.flush()
    return event, fights


def apply_fighter_profile(db: Session, data: dict) -> Fighter:
    """Update a fighter and their career stats from their profile page.

    The profile is authoritative: its record includes non-UFC fights and
    missing attributes (e.g. reach '--') are stored as None.
    """
    fighter = _find_fighter(db, data.get("ufcstats_id"), data.get("profile_url"))
    if fighter is None:
        if not data.get("name"):
            raise ValueError(f"Profile without a name: {data.get('profile_url')}")
        fighter = Fighter(name=data["name"])
        db.add(fighter)

    _apply(fighter, {"name": data.get("name"), "ufcstats_id": data.get("ufcstats_id"),
                     "profile_url": data.get("profile_url")}, keep_existing=True)
    _apply(
        fighter,
        {
            "nickname": data.get("nickname"),
            "height_in": data.get("height_in"),
            "weight_lbs": data.get("weight_lbs"),
            "reach_in": data.get("reach_in"),
            "stance": data.get("stance"),
            "dob": data.get("dob"),
            "wins": data.get("wins"),
            "losses": data.get("losses"),
            "draws": data.get("draws"),
            "no_contests": data.get("no_contests"),
        },
    )
    fighter.last_synced_at = utc_now()

    career = data.get("career_stats") or {}
    if fighter.career_stats is None:
        fighter.career_stats = FighterCareerStats(**career)
    else:
        _apply(fighter.career_stats, career)
    db.flush()
    return fighter


# ---------------------------------------------------------------------------
# Jobs
# ---------------------------------------------------------------------------

def create_job(
    db: Session,
    mode: str,
    since: date | None = None,
    until: date | None = None,
    refresh_existing: bool = False,
    refresh_fighters: bool = True,
) -> ScrapeJob:
    job = ScrapeJob(
        mode=mode,
        since=since,
        until=until,
        refresh_existing=refresh_existing,
        refresh_fighters=refresh_fighters,
        status="queued",
    )
    db.add(job)
    db.flush()
    return job


def get_active_job(db: Session) -> ScrapeJob | None:
    return db.scalar(
        select(ScrapeJob).where(ScrapeJob.status.in_(("queued", "running"))).order_by(ScrapeJob.id)
    )


def fail_stale_jobs(db: Session) -> int:
    """Mark jobs left queued/running by a previous process as failed."""
    return db.execute(
        update(ScrapeJob)
        .where(ScrapeJob.status.in_(("queued", "running")))
        .values(status="failed", finished_at=utc_now(), error_log=json.dumps(
            [{"url": None, "error": "Interrupted: API process restarted"}]
        ))
    ).rowcount or 0


def ingested_filter():
    """Events whose results are stored.

    The upcoming sync also sets scraped_at, so status must be checked too.
    """
    return (Event.status == "completed") & Event.scraped_at.is_not(None)


def incremental_since(db: Session) -> date:
    """Lower date bound for an incremental job: the oldest ingested event.

    Incremental jobs pick up new events and fill gaps inside the range
    already stored, but never extend it backwards; that takes a backfill.
    """
    oldest = db.scalar(select(func.min(Event.event_date)).where(ingested_filter()))
    if oldest is None:
        raise ValueError("No events ingested yet; run a backfill first")
    return oldest


def select_events_to_ingest(db: Session, listed: list[dict], job: ScrapeJob) -> list[dict]:
    """Events from the completed list this job should ingest, oldest first.

    listed is already limited to the job's date window (see
    incremental_since for incremental jobs). Completed events with scraped_at
    set are skipped unless a backfill asks to refresh them.
    """
    scraped = set(db.scalars(select(Event.ufcstats_id).where(ingested_filter())))
    refresh = job.mode == "backfill" and job.refresh_existing
    pending = [event for event in listed if refresh or event["ufcstats_id"] not in scraped]
    return sorted(pending, key=lambda event: event["event_date"])


def _record_error(job: ScrapeJob, errors: list[dict], url: str | None, exc: Exception) -> None:
    logger.warning("Ingestion error for %s: %s", url, exc)
    errors.append({"url": url, "error": f"{type(exc).__name__}: {exc}"})
    job.errors = len(errors)
    job.error_log = json.dumps(errors)


async def _refresh_fighters(
    db: Session, browser: UFCStatsBrowser, job: ScrapeJob, touched: set[str], errors: list[dict]
) -> None:
    """Fetch profiles for touched fighters and any never synced; commit per fighter."""
    fighters = db.scalars(
        select(Fighter).where(
            Fighter.profile_url.is_not(None),
            (Fighter.ufcstats_id.in_(touched)) | (Fighter.last_synced_at.is_(None)),
        )
    ).all()
    targets = [(fighter.ufcstats_id, fighter.profile_url) for fighter in fighters]
    logger.info("Refreshing %d fighter profiles", len(targets))

    batch_size = 20
    for start in range(0, len(targets), batch_size):
        batch = targets[start:start + batch_size]
        results = await asyncio.gather(
            *(fetch_fighter(browser, url) for _, url in batch), return_exceptions=True
        )
        for (_, url), result in zip(batch, results):
            if isinstance(result, BrowserClosedError):
                raise result  # every remaining profile would fail too
            try:
                if isinstance(result, Exception):
                    raise result
                apply_fighter_profile(db, result)
                job.fighters_upserted += 1
                db.commit()
            except Exception as exc:
                db.rollback()
                _record_error(job, errors, url, exc)
                db.commit()


async def run_ingestion_job(
    job_id: int, session_factory: sessionmaker[Session] = SessionLocal
) -> None:
    """Run a queued ScrapeJob to completion. Never raises; the outcome is on the job."""
    base_url = get_settings().ufc_stats_base_url
    with session_factory() as db:
        job = db.get(ScrapeJob, job_id)
        if job is None:
            raise ValueError(f"ScrapeJob {job_id} not found")

        job.status = "running"
        job.started_at = utc_now()
        db.commit()
        errors: list[dict] = []

        try:
            today = utc_now().date()
            if job.mode == "backfill":
                since, until = job.since, job.until or today
            else:
                since, until = incremental_since(db), today

            async with UFCStatsBrowser() as browser:
                listed = await fetch_completed_events(browser, base_url, since, until)

                pending = select_events_to_ingest(db, listed, job)
                job.events_total = len(pending)
                db.commit()
                logger.info("Job %s: %d events to ingest", job.id, len(pending))

                touched: set[str] = set()
                for listed_event in pending:
                    url = listed_event["source_url"]
                    try:
                        page = await fetch_event_with_fights(browser, url)
                        data = {**listed_event, **{k: v for k, v in page.items() if v is not None}}
                        _, fights = ingest_event(db, data)
                        job.fights_upserted += len(fights)
                        touched.update(
                            fighter["ufcstats_id"]
                            for fight in data["fights"]
                            for fighter in fight["fighters"]
                        )
                    except BrowserClosedError:
                        raise  # every remaining event would fail too
                    except Exception as exc:
                        db.rollback()
                        _record_error(job, errors, url, exc)
                    job.events_done += 1
                    db.commit()
                    logger.info("Job %s: [%d/%d] %s (%s)", job.id, job.events_done,
                                job.events_total, listed_event["name"], listed_event["event_date"])

                if job.refresh_fighters:
                    await _refresh_fighters(db, browser, job, touched, errors)

            job.status = "succeeded"
        except Exception as exc:
            logger.exception("Job %s failed", job_id)
            db.rollback()
            _record_error(job, errors, None, exc)
            job.status = "failed"
        finally:
            job.finished_at = utc_now()
            db.commit()
