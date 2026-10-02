"""Training set from the database: pre-fight stats, corner swap, split, prediction."""

from datetime import date

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from src.analytics.extract import load_raw_tables
from src.api.v1.models import Event, Fight, Fighter
from src.ml import predictor
from src.ml.dataset import fight_features, training_set
from src.ml.features import swap_corners
from tests.helpers import synthetic_features


def by_names(features: pd.DataFrame, red: str, blue: str, event_date: str) -> pd.Series:
    rows = features[
        (features["red_name"] == red) & (features["blue_name"] == blue)
        & (features["event_date"] == pd.Timestamp(event_date))
    ]
    assert len(rows) == 1
    return rows.iloc[0]


@pytest.fixture
def rematch_raw(db, raw):
    """The conftest data plus an Alpha vs Bravo rematch after UFC 300."""
    alpha = db.query(Fighter).filter_by(name="Alpha").one()
    bravo = db.query(Fighter).filter_by(name="Bravo").one()
    later = Event(name="UFC 301", event_date=date(2024, 5, 4), status="completed")
    db.add_all([later, Fight(event=later, red_fighter=alpha, blue_fighter=bravo, bout_order=1,
                             winner_side="red", weight_class="Lightweight")])
    db.commit()
    return load_raw_tables(db.get_bind())


# --- pre-fight stats -----------------------------------------------------------

def test_first_ufc_fight_has_no_history(raw):
    features = fight_features(raw)
    ufc1 = by_names(features, "Alpha", "Charlie", "1994-03-11")

    assert ufc1["red_ufc_fights"] == 0
    assert np.isnan(ufc1["red_win_rate"])
    assert np.isnan(ufc1["red_slpm"])


def test_fights_on_the_same_card_do_not_count_towards_each_other(raw):
    features = fight_features(raw)
    title = by_names(features, "Alpha", "Bravo", "2024-04-13")

    # Only UFC 1 came before; it has a result but no statistics.
    assert title["red_ufc_fights"] == 1
    assert title["red_win_rate"] == 1.0
    assert np.isnan(title["red_slpm"])
    assert title["blue_ufc_fights"] == 0


def test_career_rates_use_only_earlier_fights(rematch_raw):
    features = fight_features(rematch_raw)
    rematch = by_names(features, "Alpha", "Bravo", "2024-05-04")

    # Alpha before the rematch: UFC 1 (W), title (L, 6.5 min with stats), draw.
    assert rematch["red_ufc_fights"] == 3
    assert rematch["red_win_rate"] == pytest.approx(1 / 3)
    assert rematch["red_slpm"] == pytest.approx(13 / 6.5)
    assert rematch["red_str_acc"] == pytest.approx(0.5)
    assert rematch["red_sapm"] == pytest.approx(26 / 6.5)
    assert rematch["red_str_def"] == pytest.approx(1 - 26 / 40)
    assert rematch["red_td_avg"] == pytest.approx(1 * 15 / 6.5)
    assert rematch["red_td_acc"] == pytest.approx(0.25)
    assert np.isnan(rematch["red_td_def"])  # Bravo attempted no takedowns
    # Bravo: title (W) and split decision (W, no stats).
    assert rematch["blue_ufc_fights"] == 2
    assert rematch["blue_win_rate"] == 1.0
    assert rematch["blue_td_def"] == pytest.approx(0.75)
    assert rematch["slpm_diff"] == pytest.approx(13 / 6.5 - 26 / 6.5)


def test_attributes_stay_in_inches_and_fight_fields_are_derived(raw):
    title = by_names(fight_features(raw), "Alpha", "Bravo", "2024-04-13")

    assert (title["red_height_in"], title["red_reach_in"]) == (70, 72)
    assert title["height_in_diff"] == 2
    assert title["red_age"] == pytest.approx(34.28, abs=0.01)
    assert title["sex"] == "Women"
    assert title["weight_class_lbs"] == 125
    assert title["is_title_bout"] == 1


def test_upcoming_fight_gets_full_history(db, raw):
    features = fight_features(load_raw_tables(db.get_bind(), include_upcoming=True))
    upcoming = by_names(features, "Alpha", "Bravo", "2030-01-01")

    assert pd.isna(upcoming["winner_side"])
    assert upcoming["red_ufc_fights"] == 3
    assert upcoming["red_slpm"] == pytest.approx(2.0)
    assert len(fight_features(raw)) == 4  # excluded by default


# --- corner swap and split -----------------------------------------------------

def test_swap_corners_exchanges_sides_and_flips_winner():
    frame = synthetic_features(3)
    swapped = swap_corners(frame)

    pd.testing.assert_series_equal(swapped["red_height_in"], frame["blue_height_in"], check_names=False)
    pd.testing.assert_series_equal(swapped["blue_name"], frame["red_name"], check_names=False)
    pd.testing.assert_series_equal(swapped["height_in_diff"], -frame["height_in_diff"])
    assert (swapped["winner_side"] == "blue").all()
    assert (swapped["sex"] == "Men").all()


def test_training_set_balances_corners_and_splits_chronologically():
    X_train, X_test, y_train, y_test = training_set(synthetic_features(), test_size=0.2)

    assert len(X_test) == 20
    assert X_train["event_date"].max() < X_test["event_date"].min()
    assert 0.3 < pd.concat([y_train, y_test]).mean() < 0.7
    # Swapped rows carry the original winner in the blue corner.
    blue_won = X_train[y_train == 0]
    assert blue_won["blue_name"].str.startswith("R").all()


def test_draws_and_upcoming_fights_are_not_trained_on():
    frame = synthetic_features(10)
    frame.loc[0, "winner_side"] = "draw"
    frame.loc[1, "winner_side"] = None

    X_train, X_test, _, _ = training_set(frame)

    assert len(X_train) + len(X_test) == 8


# --- prediction ---------------------------------------------------------------

def test_predictions_are_symmetric_in_corner_order(tiny_model):
    fights = synthetic_features(5)

    as_listed = predictor.predict_fights(fights, tiny_model)
    swapped = predictor.predict_fights(swap_corners(fights), tiny_model)

    assert np.allclose(as_listed["red_win_probability"] + as_listed["blue_win_probability"], 1)
    assert np.allclose(as_listed["red_win_probability"], swapped["blue_win_probability"])
    assert list(as_listed["red_name"]) == list(fights["red_name"])
    taller_red = fights["height_in_diff"] > 0
    assert (as_listed.loc[taller_red, "predicted_winner"] == "red").all()


def test_unknown_model_lists_available(tiny_model):
    with pytest.raises(FileNotFoundError, match="Available models: tiny"):
        predictor.predict_fights(synthetic_features(1), "missing")


def test_old_format_artifact_asks_for_retraining(tmp_path, monkeypatch):
    joblib.dump(LogisticRegression(), tmp_path / "old.joblib")
    monkeypatch.setattr(predictor, "MODELS_DIR", tmp_path)

    with pytest.raises(ValueError, match="retrain"):
        predictor.load_model_artifact("old")
