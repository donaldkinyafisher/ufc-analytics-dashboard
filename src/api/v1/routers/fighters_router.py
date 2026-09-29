"""
REST API for stored fighters.

GET /api/v1/fighters?name=          search fighters by name
GET /api/v1/fighters/{fighter_id}   profile and career stats
GET /api/v1/fighters/{fighter_id}/fights   fight history, newest first
"""

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy import or_, select
from sqlalchemy.orm import Session, selectinload

from src.api.v1 import models
from src.api.v1.database import get_db
from src.api.v1.schemas.fighters_schema import FighterResponse, FighterSummary
from src.api.v1.schemas.fights_schema import FightSummary

router = APIRouter()


def _get_fighter_or_404(db: Session, fighter_id: int) -> models.Fighter:
    fighter = db.scalar(
        select(models.Fighter)
        .where(models.Fighter.id == fighter_id)
        .options(selectinload(models.Fighter.career_stats))
    )
    if fighter is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Fighter not found")
    return fighter


@router.get("", response_model=list[FighterSummary])
def get_fighters(
    db: Annotated[Session, Depends(get_db)],
    name: Annotated[str | None, Query(min_length=2)] = None,
    limit: Annotated[int, Query(ge=1, le=500)] = 50,
    offset: Annotated[int, Query(ge=0)] = 0,
):
    """Fighters ordered by name; `name` matches any part of the name, case-insensitively."""
    query = select(models.Fighter)
    if name:
        query = query.where(models.Fighter.name.ilike(f"%{name}%"))
    query = query.order_by(models.Fighter.name).limit(limit).offset(offset)
    return db.scalars(query).all()


@router.get("/{fighter_id}", response_model=FighterResponse)
def get_fighter(fighter_id: int, db: Annotated[Session, Depends(get_db)]):
    return _get_fighter_or_404(db, fighter_id)


@router.get("/{fighter_id}/fights", response_model=list[FightSummary])
def get_fighter_fights(fighter_id: int, db: Annotated[Session, Depends(get_db)]):
    _get_fighter_or_404(db, fighter_id)
    return db.scalars(
        select(models.Fight)
        .join(models.Fight.event)
        .where(
            or_(
                models.Fight.red_fighter_id == fighter_id,
                models.Fight.blue_fighter_id == fighter_id,
            )
        )
        .options(
            selectinload(models.Fight.event),
            selectinload(models.Fight.red_fighter),
            selectinload(models.Fight.blue_fighter),
        )
        .order_by(models.Event.event_date.desc(), models.Fight.bout_order)
    ).all()
