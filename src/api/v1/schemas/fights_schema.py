from datetime import date

from pydantic import BaseModel, ConfigDict


class FighterRef(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    name: str
    ufcstats_id: str | None


class EventRef(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    name: str
    event_date: date | None


class FightSummary(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    ufcstats_id: str | None
    event: EventRef
    bout_order: int | None
    red_fighter: FighterRef
    blue_fighter: FighterRef
    weight_class: str | None
    bout_type: str | None
    is_title_bout: bool
    winner_side: str | None
    method: str | None
    method_details: str | None
    round: int | None
    time: str | None
    time_format: str | None
    referee: str | None
    source_url: str | None


class FightStatisticResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    fighter_id: int
    corner: str
    result: str | None
    knockdowns: int | None
    sig_str_landed: int | None
    sig_str_attempted: int | None
    total_str_landed: int | None
    total_str_attempted: int | None
    td_landed: int | None
    td_attempted: int | None
    submission_attempts: int | None
    reversals: int | None
    control_time_seconds: int | None
    head_landed: int | None
    head_attempted: int | None
    body_landed: int | None
    body_attempted: int | None
    leg_landed: int | None
    leg_attempted: int | None
    distance_landed: int | None
    distance_attempted: int | None
    clinch_landed: int | None
    clinch_attempted: int | None
    ground_landed: int | None
    ground_attempted: int | None


class FightDetailResponse(FightSummary):
    statistics: list[FightStatisticResponse]
