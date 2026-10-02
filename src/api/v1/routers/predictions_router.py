"""
REST API for fight predictions.

GET /api/v1/predictions/events/{event_id}?model_name=xgboost
    every fight on the event's card with both fighters' win probabilities;
    model_name defaults to the best trained model (see GET /api/v1/models)
"""

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from src.api.v1.database import get_db
from src.api.v1.models import Event
from src.api.v1.schemas.ml_schema import ModelName
from src.api.v1.schemas.predictions_schema import EventPredictionsResponse
from src.api.v1.services.predictions_services import NoTrainedModelError, predict_event

router = APIRouter()


@router.get("/events/{event_id}", response_model=EventPredictionsResponse)
def get_event_predictions(
    event_id: int,
    db: Annotated[Session, Depends(get_db)],
    model_name: ModelName | None = None,
):
    event = db.get(Event, event_id)
    if event is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Event not found")
    try:
        return predict_event(db, event, model_name)
    except (FileNotFoundError, NoTrainedModelError) as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
