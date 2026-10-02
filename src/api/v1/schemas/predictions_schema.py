from pydantic import BaseModel

from src.api.v1.schemas.fights_schema import EventRef, FighterRef


class FightPrediction(BaseModel):
    fight_id: int
    bout_order: int | None
    weight_class: str | None
    bout_type: str | None
    is_title_bout: bool
    red_fighter: FighterRef
    blue_fighter: FighterRef
    # Earlier UFC fights stored for each fighter; 0 means a UFC debut.
    red_ufc_fights: int
    blue_ufc_fights: int
    # Null when either fighter is debuting: with no UFC record in the
    # database there are no career stats to base a prediction on.
    predicted_winner: FighterRef | None
    red_win_probability: float | None
    blue_win_probability: float | None


class EventPredictionsResponse(BaseModel):
    event: EventRef
    model_name: str
    fights: list[FightPrediction]
