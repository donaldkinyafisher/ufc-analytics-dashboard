"""
REST API for stored fights.

GET /api/v1/fights/{fight_id}   fight with both fighters' statistics
"""

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.orm import Session, selectinload

from src.api.v1.database import get_db
from src.api.v1.models import Fight
from src.api.v1.schemas.fights_schema import FightDetailResponse

router = APIRouter()


@router.get("/{fight_id}", response_model=FightDetailResponse)
def get_fight(fight_id: int, db: Annotated[Session, Depends(get_db)]):
    fight = db.scalar(
        select(Fight)
        .where(Fight.id == fight_id)
        .options(
            selectinload(Fight.event),
            selectinload(Fight.red_fighter),
            selectinload(Fight.blue_fighter),
            selectinload(Fight.statistics),
        )
    )
    if fight is None:
        raise HTTPException(status_code=404, detail="Fight not found")
    return fight
