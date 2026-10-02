"""
REST API for the app's overall state.

GET /api/v1/status   stored data, running jobs, trained models and needs_setup

Streamlit calls this on every page load to decide whether to show only the
Setup page (empty database) or the full app.
"""

from typing import Annotated

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from src.api.v1.database import get_db
from src.api.v1.schemas.status_schema import StatusResponse
from src.api.v1.services.status_services import get_status

router = APIRouter()


@router.get("", response_model=StatusResponse)
def read_status(db: Annotated[Session, Depends(get_db)]):
    return get_status(db)
