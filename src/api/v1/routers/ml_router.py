"""
REST API for trained models and training jobs.

GET  /api/v1/models                        supported models, test metrics, best model
POST /api/v1/models/training-jobs          start training -> 202 + Location header
GET  /api/v1/models/training-jobs          recent jobs, newest first
GET  /api/v1/models/training-jobs/{job_id} status and, once finished, metrics or error

Examples:
    {}                                      train every model
    {"models": ["xgboost"], "tune": true}   tune and train one model
"""

import asyncio
from typing import Annotated

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Query, Request, Response, status
from sqlalchemy import select
from sqlalchemy.orm import Session

from src.api.v1.database import get_db
from src.api.v1.models import TrainingJob
from src.api.v1.schemas.ml_schema import ModelSummary, TrainingJobCreate, TrainingJobResponse
from src.api.v1.services.ml_services import (
    create_training_job,
    get_active_training_job,
    list_models,
    run_training_job,
)

router = APIRouter()

# Serialises the "is a job active? -> create job" check within this process.
_job_lock = asyncio.Lock()


@router.get("", response_model=list[ModelSummary])
def get_models():
    return list_models()


@router.post("/training-jobs", response_model=TrainingJobResponse, status_code=status.HTTP_202_ACCEPTED)
async def create_job(
    payload: TrainingJobCreate,
    background_tasks: BackgroundTasks,
    request: Request,
    response: Response,
    db: Annotated[Session, Depends(get_db)],
):
    """Queue a training job and start it in the background.

    Only one job runs at a time, since every job writes the same model files.
    """
    async with _job_lock:
        active = get_active_training_job(db)
        if active is not None:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=f"Training job {active.id} is already {active.status}",
            )
        job = create_training_job(db, models=payload.models, tune=payload.tune)
        db.commit()
        db.refresh(job)

    background_tasks.add_task(run_training_job, job.id)
    response.headers["Location"] = str(request.url_for("get_training_job", job_id=job.id))
    return job


@router.get("/training-jobs", response_model=list[TrainingJobResponse])
def list_jobs(
    db: Annotated[Session, Depends(get_db)],
    limit: Annotated[int, Query(ge=1, le=100)] = 20,
):
    return db.scalars(select(TrainingJob).order_by(TrainingJob.id.desc()).limit(limit)).all()


@router.get("/training-jobs/{job_id}", response_model=TrainingJobResponse, name="get_training_job")
def get_training_job(job_id: int, db: Annotated[Session, Depends(get_db)]):
    job = db.get(TrainingJob, job_id)
    if job is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Training job not found")
    return job
