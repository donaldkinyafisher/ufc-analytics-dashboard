from datetime import date, datetime

from pydantic import BaseModel, ConfigDict


class FighterCareerStatsResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    strikes_landed_per_minute: float | None
    striking_accuracy: float | None
    strikes_absorbed_per_minute: float | None
    striking_defense: float | None
    takedown_average: float | None
    takedown_accuracy: float | None
    takedown_defense: float | None
    submission_average: float | None


class FighterSummary(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    ufcstats_id: str | None
    name: str
    nickname: str | None
    wins: int | None
    losses: int | None
    draws: int | None
    no_contests: int | None


class FighterResponse(FighterSummary):
    profile_url: str | None
    height_in: float | None
    weight_lbs: float | None
    reach_in: float | None
    stance: str | None
    dob: date | None
    last_synced_at: datetime | None
    career_stats: FighterCareerStatsResponse | None
