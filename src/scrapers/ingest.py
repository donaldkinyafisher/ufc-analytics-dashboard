"""
Run a historical ingestion job from the command line (no API server needed).

Run:
    uv run python -m src.scrapers.ingest --mode backfill --since 2025-09-28
    uv run python -m src.scrapers.ingest --mode incremental
    uv run python -m src.scrapers.ingest --mode backfill              # everything
"""

import argparse
import asyncio
import json
import logging
from datetime import date

from src.api.v1.database import Base, SessionLocal, engine
from src.api.v1.models import ScrapeJob
from src.api.v1.services.ingestion_services import (
    create_job,
    get_active_job,
    run_ingestion_job,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mode", choices=("backfill", "incremental"), default="incremental")
    parser.add_argument("--since", type=date.fromisoformat, help="backfill: oldest event date (YYYY-MM-DD)")
    parser.add_argument("--until", type=date.fromisoformat, help="backfill: newest event date (YYYY-MM-DD)")
    parser.add_argument("--refresh-existing", action="store_true", help="backfill: re-scrape stored events")
    parser.add_argument("--no-fighters", action="store_true", help="skip fighter profile refresh")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    Base.metadata.create_all(bind=engine)

    with SessionLocal() as db:
        active = get_active_job(db)
        if active is not None:
            raise SystemExit(f"Job {active.id} is already {active.status}; wait for it to finish.")
        job = create_job(
            db,
            mode=args.mode,
            since=args.since,
            until=args.until,
            refresh_existing=args.refresh_existing,
            refresh_fighters=not args.no_fighters,
        )
        db.commit()
        job_id = job.id

    asyncio.run(run_ingestion_job(job_id))

    with SessionLocal() as db:
        job = db.get(ScrapeJob, job_id)
        print(
            f"\nJob {job.id} {job.status}: {job.events_done}/{job.events_total} events, "
            f"{job.fights_upserted} fights, {job.fighters_upserted} fighter profiles, "
            f"{job.errors} errors"
        )
        if job.error_log:
            for error in json.loads(job.error_log):
                print(f"  - {error['url']}: {error['error']}")


if __name__ == "__main__":
    main()
