from datetime import UTC, date, datetime

from sqlalchemy import (
    Boolean,
    CheckConstraint,
    Date,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from src.api.v1.database import Base


def utc_now() -> datetime:
    return datetime.now(UTC)


class Fighter(Base):
    __tablename__ = "fighters"
    __table_args__ = (
        UniqueConstraint("profile_url", name="uq_fighters_profile_url"),
        UniqueConstraint("ufcstats_id", name="uq_fighters_ufcstats_id"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    ufcstats_id: Mapped[str | None] = mapped_column(String(32), nullable=True, index=True)
    name: Mapped[str] = mapped_column(String(160), index=True)
    nickname: Mapped[str | None] = mapped_column(String(160), nullable=True)
    profile_url: Mapped[str | None] = mapped_column(String(500), nullable=True)
    height_in: Mapped[float | None] = mapped_column(Float, nullable=True)
    weight_lbs: Mapped[float | None] = mapped_column(Float, nullable=True)
    reach_in: Mapped[float | None] = mapped_column(Float, nullable=True)
    stance: Mapped[str | None] = mapped_column(String(80), nullable=True)
    dob: Mapped[date | None] = mapped_column(Date, nullable=True)
    wins: Mapped[int | None] = mapped_column(Integer, nullable=True)
    losses: Mapped[int | None] = mapped_column(Integer, nullable=True)
    draws: Mapped[int | None] = mapped_column(Integer, nullable=True)
    no_contests: Mapped[int | None] = mapped_column(Integer, nullable=True)
    # Null until the fighter's profile page has been scraped.
    last_synced_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)

    red_fights: Mapped[list["Fight"]] = relationship(
        foreign_keys="Fight.red_fighter_id", back_populates="red_fighter"
    )
    blue_fights: Mapped[list["Fight"]] = relationship(
        foreign_keys="Fight.blue_fighter_id", back_populates="blue_fighter"
    )
    fight_statistics: Mapped[list["FightStatistic"]] = relationship(
        back_populates="fighter"
    )
    career_stats: Mapped["FighterCareerStats | None"] = relationship(
        back_populates="fighter", cascade="all, delete-orphan", uselist=False
    )


class Event(Base):
    __tablename__ = "events"
    __table_args__ = (
        UniqueConstraint("source_url", name="uq_events_source_url"),
        UniqueConstraint("ufcstats_id", name="uq_events_ufcstats_id"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    ufcstats_id: Mapped[str | None] = mapped_column(String(32), nullable=True, index=True)
    name: Mapped[str] = mapped_column(String(220), index=True)
    event_date: Mapped[date | None] = mapped_column(Date, nullable=True, index=True)
    location: Mapped[str | None] = mapped_column(String(220), nullable=True)
    source_url: Mapped[str | None] = mapped_column(String(500), nullable=True)
    status: Mapped[str] = mapped_column(String(40), default="upcoming")
    # Set once every fight on the card has been stored; null means the
    # event still needs (re-)ingesting.
    scraped_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utc_now)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=utc_now, onupdate=utc_now)

    fights: Mapped[list["Fight"]] = relationship(
        back_populates="event", cascade="all, delete-orphan", order_by="Fight.bout_order"
    )


class Fight(Base):
    __tablename__ = "fights"
    __table_args__ = (
        UniqueConstraint("event_id", "red_fighter_id", "blue_fighter_id", name="uq_event_fighters"),
        UniqueConstraint("ufcstats_id", name="uq_fights_ufcstats_id"),
        CheckConstraint(
            "winner_side IS NULL OR winner_side IN ('red', 'blue', 'draw', 'nc')",
            name="ck_fights_winner_side",
        ),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    ufcstats_id: Mapped[str | None] = mapped_column(String(32), nullable=True, index=True)
    event_id: Mapped[int] = mapped_column(ForeignKey("events.id"), index=True)
    red_fighter_id: Mapped[int] = mapped_column(ForeignKey("fighters.id"), index=True)
    blue_fighter_id: Mapped[int] = mapped_column(ForeignKey("fighters.id"), index=True)
    # Position on the card: 1 = main event.
    bout_order: Mapped[int | None] = mapped_column(Integer, nullable=True)
    weight_class: Mapped[str | None] = mapped_column(String(120), nullable=True)
    bout_type: Mapped[str | None] = mapped_column(String(120), nullable=True)
    is_title_bout: Mapped[bool] = mapped_column(Boolean, default=False)
    source_url: Mapped[str | None] = mapped_column(String(500), nullable=True)
    # 'red' | 'blue' | 'draw' | 'nc'; null while the fight is unresolved.
    winner_side: Mapped[str | None] = mapped_column(String(20), nullable=True)
    method: Mapped[str | None] = mapped_column(String(120), nullable=True)
    # Finish detail (e.g. 'Punches to Head From Mount') or judges' scorecards.
    method_details: Mapped[str | None] = mapped_column(Text, nullable=True)
    round: Mapped[int | None] = mapped_column(Integer, nullable=True)
    time: Mapped[str | None] = mapped_column(String(20), nullable=True)
    time_format: Mapped[str | None] = mapped_column(String(80), nullable=True)
    referee: Mapped[str | None] = mapped_column(String(160), nullable=True)

    event: Mapped[Event] = relationship(back_populates="fights")
    red_fighter: Mapped[Fighter] = relationship(
        foreign_keys=[red_fighter_id], back_populates="red_fights"
    )
    blue_fighter: Mapped[Fighter] = relationship(
        foreign_keys=[blue_fighter_id], back_populates="blue_fights"
    )

    statistics: Mapped[list["FightStatistic"]] = relationship(
        back_populates="fight", cascade="all, delete-orphan"
    )
    prediction: Mapped["Prediction | None"] = relationship(
        back_populates="fight",
        cascade="all, delete-orphan",
        single_parent=True,
        uselist=False,
    )


class Prediction(Base):
    __tablename__ = "predictions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    fight_id: Mapped[int] = mapped_column(ForeignKey("fights.id"), unique=True, index=True)
    predicted_winner_id: Mapped[int | None] = mapped_column(ForeignKey("fighters.id"), nullable=True)
    red_win_probability: Mapped[float] = mapped_column(Float)
    blue_win_probability: Mapped[float] = mapped_column(Float)
    model_name: Mapped[str] = mapped_column(String(120), default="baseline_record_heuristic")
    feature_snapshot: Mapped[str] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utc_now)

    fight: Mapped[Fight] = relationship(back_populates="prediction")
    predicted_winner: Mapped[Fighter | None] = relationship(foreign_keys=[predicted_winner_id])

class FightStatistic(Base):
    """One fighter's performance statistics from one fight."""

    __tablename__ = "fight_statistics"
    __table_args__ = (
        UniqueConstraint("fight_id", "fighter_id", name="uq_fight_stats_fighter"),
        UniqueConstraint("fight_id", "corner", name="uq_fight_stats_corner"),
        CheckConstraint("corner IN ('red', 'blue')", name="ck_fight_stats_corner"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    fight_id: Mapped[int] = mapped_column(
        ForeignKey("fights.id", ondelete="CASCADE"), index=True
    )
    fighter_id: Mapped[int] = mapped_column(
        ForeignKey("fighters.id", ondelete="CASCADE"), index=True
    )
    corner: Mapped[str] = mapped_column(String(4))
    # 'W' | 'L' | 'D' | 'NC'
    result: Mapped[str | None] = mapped_column(String(4), nullable=True)

    # Fight totals. Raw landed/attempted counts are stored; percentages are
    # derived from them when building features.
    knockdowns: Mapped[int | None] = mapped_column(Integer, nullable=True)
    sig_str_landed: Mapped[int | None] = mapped_column(Integer, nullable=True)
    sig_str_attempted: Mapped[int | None] = mapped_column(Integer, nullable=True)
    total_str_landed: Mapped[int | None] = mapped_column(Integer, nullable=True)
    total_str_attempted: Mapped[int | None] = mapped_column(Integer, nullable=True)
    td_landed: Mapped[int | None] = mapped_column(Integer, nullable=True)
    td_attempted: Mapped[int | None] = mapped_column(Integer, nullable=True)
    submission_attempts: Mapped[int | None] = mapped_column(Integer, nullable=True)
    reversals: Mapped[int | None] = mapped_column(Integer, nullable=True)
    control_time_seconds: Mapped[int | None] = mapped_column(Integer, nullable=True)

    # Significant strikes by target and position.
    head_landed: Mapped[int | None] = mapped_column(Integer, nullable=True)
    head_attempted: Mapped[int | None] = mapped_column(Integer, nullable=True)
    body_landed: Mapped[int | None] = mapped_column(Integer, nullable=True)
    body_attempted: Mapped[int | None] = mapped_column(Integer, nullable=True)
    leg_landed: Mapped[int | None] = mapped_column(Integer, nullable=True)
    leg_attempted: Mapped[int | None] = mapped_column(Integer, nullable=True)
    distance_landed: Mapped[int | None] = mapped_column(Integer, nullable=True)
    distance_attempted: Mapped[int | None] = mapped_column(Integer, nullable=True)
    clinch_landed: Mapped[int | None] = mapped_column(Integer, nullable=True)
    clinch_attempted: Mapped[int | None] = mapped_column(Integer, nullable=True)
    ground_landed: Mapped[int | None] = mapped_column(Integer, nullable=True)
    ground_attempted: Mapped[int | None] = mapped_column(Integer, nullable=True)

    fight: Mapped[Fight] = relationship(back_populates="statistics")
    fighter: Mapped[Fighter] = relationship(back_populates="fight_statistics")


class FighterCareerStats(Base):
    """Latest aggregate statistics for a fighter.

    This is intentionally the current state only; historical snapshots are
    not stored in this version.
    """

    __tablename__ = "fighter_career_stats"

    fighter_id: Mapped[int] = mapped_column(
        ForeignKey("fighters.id", ondelete="CASCADE"), primary_key=True
    )
    strikes_landed_per_minute: Mapped[float | None] = mapped_column(Float, nullable=True)
    striking_accuracy: Mapped[float | None] = mapped_column(Float, nullable=True)
    strikes_absorbed_per_minute: Mapped[float | None] = mapped_column(Float, nullable=True)
    striking_defense: Mapped[float | None] = mapped_column(Float, nullable=True)
    takedown_average: Mapped[float | None] = mapped_column(Float, nullable=True)
    takedown_accuracy: Mapped[float | None] = mapped_column(Float, nullable=True)
    takedown_defense: Mapped[float | None] = mapped_column(Float, nullable=True)
    submission_average: Mapped[float | None] = mapped_column(Float, nullable=True)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=utc_now, onupdate=utc_now)

    fighter: Mapped[Fighter] = relationship(back_populates="career_stats")


class ScrapeJob(Base):
    """One historical ingestion run (backfill or incremental update)."""

    __tablename__ = "scrape_jobs"
    __table_args__ = (
        CheckConstraint("mode IN ('backfill', 'incremental')", name="ck_scrape_jobs_mode"),
        CheckConstraint(
            "status IN ('queued', 'running', 'succeeded', 'failed')",
            name="ck_scrape_jobs_status",
        ),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    mode: Mapped[str] = mapped_column(String(20))
    since: Mapped[date | None] = mapped_column(Date, nullable=True)
    until: Mapped[date | None] = mapped_column(Date, nullable=True)
    refresh_existing: Mapped[bool] = mapped_column(Boolean, default=False)
    refresh_fighters: Mapped[bool] = mapped_column(Boolean, default=True)
    status: Mapped[str] = mapped_column(String(20), default="queued", index=True)

    events_total: Mapped[int] = mapped_column(Integer, default=0)
    events_done: Mapped[int] = mapped_column(Integer, default=0)
    fights_upserted: Mapped[int] = mapped_column(Integer, default=0)
    fighters_upserted: Mapped[int] = mapped_column(Integer, default=0)
    errors: Mapped[int] = mapped_column(Integer, default=0)
    # JSON list of {"url": ..., "error": ...} for pages that failed.
    error_log: Mapped[str | None] = mapped_column(Text, nullable=True)

    created_at: Mapped[datetime] = mapped_column(DateTime, default=utc_now)
    started_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)


class TrainingJob(Base):
    """One run of model training, started through the API."""

    __tablename__ = "training_jobs"
    __table_args__ = (
        CheckConstraint(
            "status IN ('queued', 'running', 'succeeded', 'failed')",
            name="ck_training_jobs_status",
        ),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    # JSON list of model names, e.g. ["xgboost", "svm"].
    models: Mapped[str] = mapped_column(Text)
    tune: Mapped[bool] = mapped_column(Boolean, default=False)
    status: Mapped[str] = mapped_column(String(20), default="queued", index=True)
    # JSON object of per-model test metrics, set when the job succeeds.
    metrics: Mapped[str | None] = mapped_column(Text, nullable=True)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)

    created_at: Mapped[datetime] = mapped_column(DateTime, default=utc_now)
    started_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
