"""
REST API for historical ingestion jobs.

A job crawls completed events on ufcstats.com and stores events, fights,
fight statistics and fighter profiles. Jobs run in the background; poll the
job resource for progress.

POST /api/v1/ingestion/jobs            start a job -> 202 + Location header
GET  /api/v1/ingestion/jobs            recent jobs, newest first
GET  /api/v1/ingestion/jobs/{job_id}   status, progress counters and errors

Examples:
    {"mode": "backfill", "since": "2025-09-28"}   last year only
    {"mode": "backfill"}                          everything on ufcstats
    {"mode": "incremental"}                       new events since the last run
"""

import asyncio
from typing import Annotated

from fastapi import (
    APIRouter,
    BackgroundTasks,
    Depends,
    HTTPException,
    Query,
    Request,
    Response,
    status,
)
from sqlalchemy import select
from sqlalchemy.orm import Session

from src.api.v1.database import get_db
from src.api.v1.models import ScrapeJob
from src.api.v1.schemas.ingestionSchema import IngestionJobCreate, IngestionJobResponse
from src.api.v1.services.ingestionServices import (
    create_job,
    get_active_job,
    run_ingestion_job,
)

router = APIRouter()

# Serialises the "is a job active? -> create job" check within this process.
_job_lock = asyncio.Lock()


@router.post("/jobs", response_model=IngestionJobResponse, status_code=status.HTTP_202_ACCEPTED)
async def create_ingestion_job(
    payload: IngestionJobCreate,
    background_tasks: BackgroundTasks,
    request: Request,
    response: Response,
    db: Annotated[Session, Depends(get_db)],
):
    """Queue a job and start it in the background. Only one job runs at a time."""
    async with _job_lock:
        active = get_active_job(db)
        if active is not None:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=f"Ingestion job {active.id} is already {active.status}",
            )
        job = create_job(db, **payload.model_dump())
        db.commit()
        db.refresh(job)

    background_tasks.add_task(run_ingestion_job, job.id)
    response.headers["Location"] = str(request.url_for("get_ingestion_job", job_id=job.id))
    return job


@router.get("/jobs", response_model=list[IngestionJobResponse])
def list_ingestion_jobs(
    db: Annotated[Session, Depends(get_db)],
    limit: Annotated[int, Query(ge=1, le=100)] = 20,
):
    return db.scalars(select(ScrapeJob).order_by(ScrapeJob.id.desc()).limit(limit)).all()


@router.get("/jobs/{job_id}", response_model=IngestionJobResponse, name="get_ingestion_job")
def get_ingestion_job(job_id: int, db: Annotated[Session, Depends(get_db)]):
    job = db.get(ScrapeJob, job_id)
    if job is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Ingestion job not found")
    return job
