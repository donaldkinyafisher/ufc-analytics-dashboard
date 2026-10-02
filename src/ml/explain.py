"""SHAP explanations for any saved model.

Every model is explained through red_win_probabilities, so all of them
(including the PyTorch network, which SHAP can't call directly) are
explained on the same scale: the probability that red wins.
"""

from typing import Any

import numpy as np
import shap

from src.ml.predictor import red_win_probabilities


def shap_values(
    model: Any,
    features: np.ndarray,
    max_rows: int = 100,
    random_state: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """(SHAP values, explained rows) for a random sample of preprocessed rows.

    Features are "removed" by setting them to a single reference row, the
    median of `features`, rather than averaging over a background sample.
    Each SHAP value is then the change in P(red wins) from moving that
    feature away from the typical matchup. One reference row keeps slow
    models (kernel SVM, KNN) fast enough for an interactive page.
    """
    rng = np.random.default_rng(random_state)
    rows = features[rng.choice(len(features), min(max_rows, len(features)), replace=False)]
    reference = np.median(features, axis=0, keepdims=True)

    explainer = shap.Explainer(
        lambda x: red_win_probabilities(model, x),
        shap.maskers.Independent(reference),
        seed=random_state,
    )
    return explainer(rows).values, rows
