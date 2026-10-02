"""Prediction helpers for upcoming UFC fights.

Each saved model artifact holds the estimator together with the preprocessor
it was trained with, so prediction applies exactly the training transform.
Input rows come from src.ml.dataset.fight_features.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from scipy.special import expit

from src.ml.dataset import ID_COLUMNS
from src.ml.features import select_features, swap_corners
from src.ml.utils import MODEL_FILE_SUFFIX, MODELS_DIR

# Logistic regression has the highest recorded validation accuracy in the
# bundled metrics and does not require optional XGBoost runtime support.
DEFAULT_MODEL_NAME = "logistic_regression"


def predict_fights(fights: pd.DataFrame, model_name: str = DEFAULT_MODEL_NAME) -> pd.DataFrame:
    """Predict the winner of each fight in a fight_features dataframe.

    Each fight is scored as listed and with corners swapped, and the two are
    averaged, so the result doesn't depend on which fighter is listed first.
    The returned dataframe preserves the input index, carries the input's
    identifier columns, and adds red/blue win probabilities and
    ``predicted_winner``.
    """
    if not isinstance(fights, pd.DataFrame):
        raise TypeError("fights must be a pandas DataFrame.")
    if fights.empty:
        return _empty_prediction_frame(fights.index)

    artifact = load_model_artifact(model_name)
    as_listed = _red_probabilities(artifact, fights)
    swapped = _red_probabilities(artifact, swap_corners(fights))
    red_probabilities = (as_listed + (1.0 - swapped)) / 2
    blue_probabilities = 1.0 - red_probabilities

    predictions = fights[[column for column in ID_COLUMNS if column in fights]].copy()
    predictions["red_win_probability"] = red_probabilities.round(4)
    predictions["blue_win_probability"] = blue_probabilities.round(4)
    predictions["predicted_winner"] = np.where(red_probabilities >= 0.5, "red", "blue")
    predictions["model_name"] = model_name
    return predictions


def load_model_artifact(model_name: str = DEFAULT_MODEL_NAME) -> dict[str, Any]:
    """Load a named artifact from ``src/ml/artifacts/models``.

    Returns a dict with ``model``, ``preprocessor``, ``feature_columns`` and
    ``trained_at``.
    """
    if not model_name or Path(model_name).name != model_name or model_name.endswith(MODEL_FILE_SUFFIX):
        raise ValueError("model_name must be an artifact name without a file extension.")

    model_path = MODELS_DIR / f"{model_name}{MODEL_FILE_SUFFIX}"
    if not model_path.is_file():
        available_models = sorted(path.stem for path in MODELS_DIR.glob(f"*{MODEL_FILE_SUFFIX}"))
        raise FileNotFoundError(
            f"Trained model '{model_name}' was not found in {MODELS_DIR}. "
            f"Available models: {', '.join(available_models) or 'none'}"
        )

    artifact = joblib.load(model_path)
    if not isinstance(artifact, dict) or "model" not in artifact:
        raise ValueError(f"Model '{model_name}' was saved in an old format; retrain it.")
    return artifact


def load_trained_model(model_name: str = DEFAULT_MODEL_NAME) -> Any:
    """The estimator alone, without its preprocessor."""
    return load_model_artifact(model_name)["model"]


def red_win_probabilities(model: Any, features: np.ndarray) -> np.ndarray:
    """Return probabilities for label 1 (the red fighter) for any saved model."""
    if hasattr(model, "predict_proba"):
        probabilities = np.asarray(model.predict_proba(features), dtype=float)
        classes = np.asarray(model.classes_)
        red_index = _red_class_index(classes)
        return probabilities[:, red_index]

    if hasattr(model, "decision_function"):
        scores = np.asarray(model.decision_function(features), dtype=float)
        probabilities = expit(scores)
        classes = np.asarray(model.classes_)
        return probabilities if classes[-1] == 1 else 1.0 - probabilities

    # FightWinnerNet is the only supplied estimator without the sklearn API.
    try:
        import torch

        model.eval()
        with torch.no_grad():
            logits = model(torch.tensor(np.asarray(features), dtype=torch.float32))
        return torch.sigmoid(logits).cpu().numpy()
    except Exception as exc:
        raise TypeError("The selected model does not expose a supported prediction interface.") from exc


def _red_probabilities(artifact: dict[str, Any], fights: pd.DataFrame) -> np.ndarray:
    transformed = artifact["preprocessor"].transform(select_features(fights, artifact["feature_columns"]))
    dense = transformed.toarray() if hasattr(transformed, "toarray") else np.asarray(transformed)
    return red_win_probabilities(artifact["model"], dense.astype(np.float32))


def _red_class_index(classes: np.ndarray) -> int:
    matches = np.flatnonzero(classes == 1)
    if len(matches) != 1:
        raise ValueError(f"Model classes must include label 1 for the red fighter; got {classes.tolist()}.")
    return int(matches[0])


def _empty_prediction_frame(index: pd.Index) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "red_win_probability": pd.Series(index=index, dtype=float),
            "blue_win_probability": pd.Series(index=index, dtype=float),
            "predicted_winner": pd.Series(index=index, dtype=str),
            "model_name": pd.Series(index=index, dtype=str),
        },
        index=index,
    )
