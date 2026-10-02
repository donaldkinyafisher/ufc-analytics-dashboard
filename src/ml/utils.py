"""Artifact paths and loaders for the ML package.

No Streamlit here: the API imports this package too. The Streamlit app wraps
these loaders with st.cache_* in src/utils.py.
"""

import json
from pathlib import Path

import numpy as np

ARTIFACT_DIR = Path(__file__).resolve().parent / "artifacts"
MODELS_DIR = ARTIFACT_DIR / "models"
DATA_DIR = ARTIFACT_DIR / "data"
METRICS_DIR = ARTIFACT_DIR / "metrics"
DATASET_PATH = DATA_DIR / "ufc_split_data.npz"
MODEL_METRICS_PATH = METRICS_DIR / "model_metrics.json"
MODEL_FILE_SUFFIX = ".joblib"

# Every model type training supports, in the order the app lists them.
MODEL_NAMES = [
    "pytorch_mlp",
    "logistic_regression",
    "svm",
    "knn",
    "random_forest",
    "xgboost",
]


def load_preprocessed_data(path: Path = DATASET_PATH) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """The saved train/test split: (X_train, y_train, X_test, y_test, feature_names)."""
    with np.load(path) as archive:
        return (
            archive["X_train"],
            archive["y_train"],
            archive["X_test"],
            archive["y_test"],
            archive["feature_names"],
        )


def load_model_metrics(path: Path = MODEL_METRICS_PATH) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Model metrics file not found at {path}.")
    with path.open("r", encoding="utf-8") as metrics_file:
        return json.load(metrics_file)
