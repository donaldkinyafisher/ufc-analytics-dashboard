import json
from datetime import date, datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from src.ml.utils import MODEL_NAMES

ModelName = Literal[tuple(MODEL_NAMES)]


class ModelMetrics(BaseModel):
    """Scores on the held-out test set (the newest fights)."""

    accuracy: float
    precision: float
    recall: float
    f1_score: float
    train_size: int | None = None
    test_size: int | None = None
    test_from: date | None = None
    trained_at: datetime | None = None


class ModelSummary(BaseModel):
    name: ModelName
    trained: bool
    # Highest F1 among trained models; the default for predictions.
    is_best: bool
    metrics: ModelMetrics | None


class TrainingJobCreate(BaseModel):
    """Train some or all models. Tuning runs an Optuna search first (not for pytorch_mlp)."""

    models: list[ModelName] = Field(default_factory=lambda: list(MODEL_NAMES), min_length=1)
    tune: bool = False

    @field_validator("models")
    @classmethod
    def unique(cls, models: list[str]) -> list[str]:
        return list(dict.fromkeys(models))


class TrainingJobResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    models: list[ModelName]
    tune: bool
    status: str
    metrics: dict[str, ModelMetrics] | None
    error: str | None
    created_at: datetime
    started_at: datetime | None
    finished_at: datetime | None

    @field_validator("models", "metrics", mode="before")
    @classmethod
    def parse_json(cls, value: object) -> object:
        return json.loads(value) if isinstance(value, str) else value
