from __future__ import annotations

import json
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import optuna
import pandas as pd
import torch
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    precision_recall_fscore_support,
)
from sklearn.model_selection import KFold, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from xgboost import XGBClassifier

from src.analytics.extract import load_raw_tables

from .config import SEARCH_SPACES
from .dataset import fight_features, training_set
from .features import FEATURE_COLUMNS, LABEL_MAPPING, build_preprocessor, select_features
from .nets import FightWinnerNet
from .utils import (
    DATA_DIR,
    DATASET_PATH,
    METRICS_DIR,
    MODEL_FILE_SUFFIX,
    MODEL_METRICS_PATH,
    MODEL_NAMES,
    MODELS_DIR,
)

MODELS_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR.mkdir(parents=True, exist_ok=True)
METRICS_DIR.mkdir(parents=True, exist_ok=True)

DEFAULT_COMPARISON_MODELS = MODEL_NAMES


def train_model(
    models: list = ["pytorch_mlp"],
    model_configs: dict[str, dict[str, Any]] | None = None,
    metrics_path: str | Path = MODEL_METRICS_PATH,
    test_size: float = 0.2,
    random_state: int = 42,
    tune:bool = False,
    raw: dict[str, pd.DataFrame] | None = None,
):
    """
    Train one or more fight-winner classifiers on the database.

    The training set is rebuilt on every run (the database grows), and the
    split is saved to artifacts/data for the Train page. Each model is saved
    to artifacts/models together with the preprocessor it was trained with,
    and its evaluation metrics are merged into the metrics JSON file.
    `raw` overrides the tables read from the database (used by tests).
    """

    model_configs = model_configs or {}
    candidate_model_types = models
    unknown_models = set(candidate_model_types) - set(_MODEL_TRAINERS)
    if unknown_models:
        raise ValueError(f"Unknown model type(s): {', '.join(sorted(unknown_models))}")

    features = fight_features(raw if raw is not None else load_raw_tables())
    X_train, X_test, y_train, y_test = training_set(
        features, test_size=test_size, random_state=random_state
    )
    y_train, y_test = y_train.to_numpy(), y_test.to_numpy()

    preprocessor = build_preprocessor(FEATURE_COLUMNS)
    X_train_processed = _as_dense_float32(preprocessor.fit_transform(select_features(X_train)))
    X_test_processed = _as_dense_float32(preprocessor.transform(select_features(X_test)))
    feature_names = preprocessor.get_feature_names_out().tolist()

    np.savez_compressed(
        DATASET_PATH,
        X_train=X_train_processed,
        X_test=X_test_processed,
        y_train=y_train,
        y_test=y_test,
        feature_names=feature_names,
    )
    split_info = {
        "train_size": len(X_train),
        "test_size": len(X_test),
        "test_from": X_test["event_date"].min().date().isoformat(),
    }

    results = {}
    for candidate in candidate_model_types:
        if tune and candidate not in ['pytorch_mlp']:
            best_params, best_score = _tune_model(
                candidate,
                X_train_processed,
                y_train,
                X_test_processed,
                y_test,
                random_state=random_state,
            )
            model_configs[candidate] = best_params
            #print(f"Best hyperparameters for {candidate}: {best_params}")
            print(f"Best cross-validation score for {candidate}: {best_score:.4f}")

        trainer = _MODEL_TRAINERS[candidate]
        model = trainer(
            X_train_processed,
            y_train,
            X_test_processed,
            y_test,
            random_state=random_state,
            config=model_configs.get(candidate, {}),
        )
        trained_at = datetime.now(UTC).isoformat(timespec="seconds")
        joblib.dump(
            {
                "model": model,
                "preprocessor": preprocessor,
                "feature_columns": FEATURE_COLUMNS,
                "trained_at": trained_at,
            },
            MODELS_DIR / f"{candidate}{MODEL_FILE_SUFFIX}",
        )

        metrics = _evaluate_model(candidate, model, X_test_processed, y_test)
        results[candidate] = {**metrics, **split_info, "trained_at": trained_at}

    #Write metrics to file
    _write_metrics(results, metrics_path)

    return results

def _tune_model(
    model_name: str,
    x_train, y_train, x_val, y_val,
    random_state: int,
    n_trials: int = 5,
) -> tuple[dict[str, Any], float]:
    trainer = _MODEL_TRAINERS[model_name]
    space_fn = SEARCH_SPACES[model_name]

    def objective(trial: optuna.Trial) -> float:
        config = space_fn(trial)
        model = trainer(x_train, y_train, x_val, y_val, random_state=random_state, config=config, tune=True)

        #Configure k-Fold splitter
        skfold = KFold(n_splits=5, shuffle=True, random_state=0)
        scores = cross_val_score(model, x_train, y_train, cv=skfold, scoring='f1_weighted', n_jobs=1)

        return float(np.mean(scores))  # or accuracy — pick your target metric

    sampler = optuna.samplers.TPESampler(seed=random_state)
    pruner = optuna.pruners.MedianPruner(n_warmup_steps=5)
    study = optuna.create_study( sampler=sampler, pruner=pruner, study_name = model_name, direction="maximize",)
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    print(f"\n---{model_name} Optimization Complete ---")

    return study.best_trial.params, study.best_value

def _train_pytorch_mlp(
    x_train: Any,
    y_train: pd.Series,
    x_val: Any,
    y_val: pd.Series,
    random_state: int,
    config: dict[str, Any],
    tune: bool = False
) -> FightWinnerNet:
    torch.manual_seed(random_state)
    np.random.seed(random_state)

    x_train_array = _as_dense_float32(x_train)
    y_train_array = _as_dense_float32(y_train)
    x_val_array = _as_dense_float32(x_val)
    y_val_array = _as_dense_float32(y_val)

    hidden_dims = tuple(config.get("hidden_dims", (64, 32)))
    dropout = float(config.get("dropout", 0.2))
    learning_rate = float(config.get("learning_rate", 0.001))
    batch_size = int(config.get("batch_size", 64))
    epochs = int(config.get("epochs", 12))
    patience = int(config.get("patience", 5))

    model = FightWinnerNet(input_dim=x_train_array.shape[1], hidden_dims=hidden_dims, dropout=dropout)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = nn.BCEWithLogitsLoss()
    train_loader = DataLoader(
        TensorDataset(torch.tensor(x_train_array), torch.tensor(y_train_array)),
        batch_size=batch_size,
        shuffle=True,
    )

    best_loss = float("inf")
    best_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
    epochs_without_improvement = 0

    val_features = torch.tensor(x_val_array)
    val_targets = torch.tensor(y_val_array)

    for _epoch in range(epochs):
        model.train()
        for batch_features, batch_targets in train_loader:
            optimizer.zero_grad()
            loss = loss_fn(model(batch_features), batch_targets)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_loss = loss_fn(model(val_features), val_targets).item()

        if val_loss < best_loss:
            best_loss = val_loss
            best_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= patience:
            break

    model.load_state_dict(best_state)
    return model


def _train_logistic_regression(
    x_train: Any,
    y_train: pd.Series,
    _x_val: Any,
    _y_val: pd.Series,
    random_state: int,
    config: dict[str, Any],
    tune: bool = False
) -> LogisticRegression:
    model = LogisticRegression(
        max_iter=int(config.get("max_iter", 1000)),
        class_weight=config.get("class_weight"),
        random_state=random_state,
    )
    if not tune:
        model.fit(x_train, y_train)
    return model


def _train_svm(
    x_train: Any,
    y_train: pd.Series,
    _x_val: Any,
    _y_val: pd.Series,
    random_state: int,
    config: dict[str, Any],
    tune: bool = False
) -> SVC:
    model = SVC(
        C=float(config.get("C", 1.0)),
        kernel=config.get("kernel", "rbf"),
        gamma=config.get("gamma", "scale"),
        class_weight=config.get("class_weight"),
        probability=False,
        random_state=random_state,
    )
    if not tune:
        model.fit(x_train, y_train)
    return model


def _train_knn(
    x_train: Any,
    y_train: pd.Series,
    _x_val: Any,
    _y_val: pd.Series,
    random_state: int,
    config: dict[str, Any],
    tune: bool = False,
) -> KNeighborsClassifier:
    # Retained for a consistent trainer interface; KNN itself is deterministic.
    _ = random_state
    model = KNeighborsClassifier(
        n_neighbors=int(config.get("n_neighbors", 15)),
        weights=config.get("weights", "distance"),
        p=int(config.get("p", 2)),
        n_jobs=int(config.get("n_jobs", -1)),
    )
    if not tune:
        model.fit(x_train, y_train)
    return model


def _train_random_forest(
    x_train: Any,
    y_train: pd.Series,
    _x_val: Any,
    _y_val: pd.Series,
    random_state: int,
    config: dict[str, Any],
    tune: bool = False,
) -> RandomForestClassifier:
    model = RandomForestClassifier(
        n_estimators=int(config.get("n_estimators", 300)),
        max_depth=config.get("max_depth"),
        min_samples_leaf=int(config.get("min_samples_leaf", 1)),
        class_weight=config.get("class_weight"),
        n_jobs=int(config.get("n_jobs", -1)),
        random_state=random_state,
    )
    if not tune:
        model.fit(x_train, y_train)
    return model


def _train_xgboost(
    x_train: Any,
    y_train: pd.Series,
    _x_val: Any,
    _y_val: pd.Series,
    random_state: int,
    config: dict[str, Any],
    tune: bool = False,
) -> Any:
    

    model = XGBClassifier(
        n_estimators=int(config.get("n_estimators", 300)),
        max_depth=int(config.get("max_depth", 4)),
        learning_rate=float(config.get("learning_rate", 0.05)),
        subsample=float(config.get("subsample", 0.8)),
        colsample_bytree=float(config.get("colsample_bytree", 0.8)),
        objective="binary:logistic",
        eval_metric="logloss",
        n_jobs=1,
        random_state=random_state,
    )
    if not tune:
        model.fit(x_train, y_train)

    return model

def _train_transformer_model(
    x_train: Any,
    y_train: pd.Series,
    x_val: Any,
    y_val: pd.Series,
    random_state: int,
    config: dict[str, Any],
    tune: bool = False
) -> Any:
    """
    Placeholder for training a transformer-based model.
    This function should be implemented with the specific transformer architecture and training logic.
    """
    raise NotImplementedError("Transformer model training is not yet implemented.")

def _evaluate_model(model_name: str, model: Any, x_val: Any, y_val: pd.Series) -> dict[str, Any]:
    if model_name == "pytorch_mlp":
        model.eval()
        with torch.no_grad():
            logits = model(torch.tensor(_as_dense_float32(x_val), dtype=torch.float32))
            probabilities = torch.sigmoid(logits).cpu().numpy()
        predictions = (probabilities >= 0.5).astype(int)
    else:
        predictions = model.predict(x_val)

    precision, recall, f1_score, _support = precision_recall_fscore_support(
        y_val,
        predictions,
        average="binary",
        pos_label=LABEL_MAPPING["red"],
        zero_division=0,
    )
    report = classification_report(
        y_val,
        predictions,
        target_names=["blue", "red"],
        zero_division=0,
        output_dict=True,
    )

    return {
        "accuracy": float(accuracy_score(y_val, predictions)),
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1_score),
        "classification_report": report,
    }


def _as_dense_float32(features: Any) -> np.ndarray:
    if hasattr(features, "toarray"):
        features = features.toarray()
    return np.asarray(features, dtype=np.float32)


def _write_metrics(metrics: dict[str, Any], metrics_path: str | Path) -> None:
    path = Path(metrics_path)

    # Load existing metrics file if present and valid
    existing: dict[str, Any] = {}
    if path.exists():
        try:
            with path.open("r", encoding="utf-8") as f:
                existing = json.load(f) or {}
        except (json.JSONDecodeError, OSError):
            existing = {}

    # Get all results keys
    models = list(metrics.keys())

    #Store metrics under top-level keys by model_key
    for model_key in models:
        existing[model_key] = metrics[model_key]

    with path.open("w", encoding="utf-8") as metrics_file:
        json.dump(existing, metrics_file, indent=2)



_MODEL_TRAINERS: dict[str, Callable[..., Any]] = {
    "pytorch_mlp": _train_pytorch_mlp,
    "logistic_regression": _train_logistic_regression,
    "svm": _train_svm,
    "knn": _train_knn,
    "random_forest": _train_random_forest,
    "xgboost": _train_xgboost,
    "transformer": _train_transformer_model,
}

if __name__ == "__main__":

    results = train_model(
        models=["svm", "knn", "xgboost"],
        tune=True
    )
    #print("Training completed. Results:\n", results)
