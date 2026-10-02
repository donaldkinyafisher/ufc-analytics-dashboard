"""Model features: the columns used, the label, and the preprocessing.

Shared by training and prediction so the two always transform fights the
same way. The feature rows themselves are built from the database in
src/ml/dataset.py. Lengths are inches and rates are fractions (0-1), as
stored in the database.
"""

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

TARGET_COLUMN = "winner_side"
LABEL_MAPPING = {"blue": 0, "red": 1}
INVERSE_LABEL_MAPPING = {value: key for key, value in LABEL_MAPPING.items()}
CORNERS = ("red", "blue")

# Each per-corner feature has a red_ and a blue_ column. The career stats are
# computed from fights before the one being predicted (see dataset.py).
CORNER_NUMERIC = [
    "height_in",
    "reach_in",
    "age",
    "ufc_fights",
    "win_rate",
    "slpm",
    "str_acc",
    "sapm",
    "str_def",
    "td_avg",
    "td_acc",
    "td_def",
    "sub_avg",
]
CORNER_CATEGORICAL = ["stance"]
FIGHT_NUMERIC = ["weight_class_lbs", "is_title_bout"]
FIGHT_CATEGORICAL = ["sex"]

# Red minus blue, for each per-corner numeric feature.
DIFF_FEATURES = [f"{feature}_diff" for feature in CORNER_NUMERIC]

NUMERIC_FEATURES = (
    [f"{corner}_{feature}" for corner in CORNERS for feature in CORNER_NUMERIC]
    + DIFF_FEATURES
    + FIGHT_NUMERIC
)
CATEGORICAL_FEATURES = [
    f"{corner}_{feature}" for corner in CORNERS for feature in CORNER_CATEGORICAL
] + FIGHT_CATEGORICAL

FEATURE_COLUMNS = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def swap_corners(fights: pd.DataFrame) -> pd.DataFrame:
    """The same fights with red and blue exchanged.

    Every red_*/blue_* pair is swapped, differences change sign and the
    winner flips. Used to balance corners in training (ufcstats lists the
    winner first) and to make predictions independent of corner order.
    """
    swapped = fights.copy()
    for column in fights.columns:
        if column.startswith("red_") and f"blue_{column[4:]}" in fights.columns:
            blue = f"blue_{column[4:]}"
            swapped[column], swapped[blue] = fights[blue], fights[column]
    for column in DIFF_FEATURES:
        if column in fights.columns:
            swapped[column] = -fights[column]
    if TARGET_COLUMN in fights.columns:
        swapped[TARGET_COLUMN] = fights[TARGET_COLUMN].replace({"red": "blue", "blue": "red"})
    return swapped


def select_features(df: pd.DataFrame, feature_columns: list[str] = FEATURE_COLUMNS) -> pd.DataFrame:
    """The feature columns in training order; missing columns are all-NaN."""
    selected = pd.DataFrame(index=df.index)
    for column in feature_columns:
        selected[column] = df[column] if column in df.columns else np.nan
        if column in NUMERIC_FEATURES:
            selected[column] = pd.to_numeric(selected[column], errors="coerce").astype(float)
        else:
            selected[column] = selected[column].where(selected[column].notna(), np.nan)
    return selected


def build_preprocessor(feature_columns: list[str] = FEATURE_COLUMNS) -> ColumnTransformer:
    numeric_features = [column for column in NUMERIC_FEATURES if column in feature_columns]
    categorical_features = [column for column in CATEGORICAL_FEATURES if column in feature_columns]

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, numeric_features),
            ("categorical", categorical_pipeline, categorical_features),
        ],
        remainder="drop",
    )
