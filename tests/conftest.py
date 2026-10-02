import warnings
from datetime import date

import joblib
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from src.analytics.extract import load_raw_tables
from src.api.v1.database import Base
from src.api.v1.models import Event, Fight, Fighter, FightStatistic
from src.ml import predictor
from src.ml import utils as ml_utils
from src.ml.features import FEATURE_COLUMNS, build_preprocessor, select_features, swap_corners
from tests.helpers import synthetic_features


@pytest.fixture
def session_factory():
    """Fresh in-memory SQLite database per test."""
    engine = create_engine(
        "sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool
    )
    Base.metadata.create_all(bind=engine)
    yield sessionmaker(bind=engine, autocommit=False, autoflush=False)
    engine.dispose()


@pytest.fixture
def db(session_factory):
    with session_factory() as session:
        yield session


def _stats(fight, fighter, corner, **counts):
    return FightStatistic(fight=fight, fighter=fighter, corner=corner, **counts)


@pytest.fixture
def raw(db):
    """Two completed events (four fights) plus one upcoming event, extracted via SQL."""
    alpha = Fighter(name="Alpha", height_in=70, reach_in=72, stance="Orthodox", dob=date(1990, 1, 1))
    bravo = Fighter(name="Bravo", height_in=68, reach_in=70, stance="Southpaw", dob=date(1995, 1, 1))
    charlie = Fighter(name="Charlie", height_in=None, reach_in=None, stance="", dob=None)
    old = Event(name="UFC 1", event_date=date(1994, 3, 11), status="completed")
    new = Event(name="UFC 300", event_date=date(2024, 4, 13), status="completed")
    upcoming = Event(name="UFC 999", event_date=date(2030, 1, 1), status="upcoming")
    db.add_all([alpha, bravo, charlie, old, new, upcoming])
    common = {"time_format": "3 Rnd (5-5-5)"}
    # Blue corner wins the title fight, which lasts 390 s (6.5 min).
    title = Fight(event=new, red_fighter=alpha, blue_fighter=bravo, bout_order=1, winner_side="blue",
                  weight_class="Women's Flyweight", is_title_bout=True, method="KO/TKO",
                  round=2, time="1:30", **common)
    split = Fight(event=new, red_fighter=bravo, blue_fighter=charlie, bout_order=2, winner_side="red",
                  weight_class="Lightweight", method="Decision - Split", round=3, time="5:00", **common)
    draw = Fight(event=new, red_fighter=alpha, blue_fighter=charlie, bout_order=3, winner_side="draw",
                 weight_class=None, method="Decision - Majority", round=3, time="5:00", **common)
    ufc1 = Fight(event=old, red_fighter=alpha, blue_fighter=charlie, bout_order=1, winner_side="red",
                 weight_class="Open Weight", method="Submission", round=1, time="0:26",
                 time_format="No Time Limit")
    db.add_all([title, split, draw, ufc1, Fight(event=upcoming, red_fighter=alpha, blue_fighter=bravo)])
    db.add_all([
        _stats(title, alpha, "red", knockdowns=0, sig_str_landed=13, sig_str_attempted=26,
               td_landed=1, td_attempted=4, control_time_seconds=39,
               head_landed=10, body_landed=2, leg_landed=1,
               distance_landed=8, clinch_landed=3, ground_landed=2),
        _stats(title, bravo, "blue", knockdowns=1, sig_str_landed=26, sig_str_attempted=40,
               td_landed=0, td_attempted=0, control_time_seconds=0,
               head_landed=20, body_landed=4, leg_landed=2,
               distance_landed=20, clinch_landed=6, ground_landed=0),
        # UFC 1 era: no stats recorded for either corner beyond the rows themselves.
        _stats(ufc1, alpha, "red"),
        _stats(ufc1, charlie, "blue"),
    ])
    db.commit()
    return load_raw_tables(db.get_bind())


@pytest.fixture
def tiny_model(tmp_path, monkeypatch):
    """A logistic regression trained on synthetic rows where taller wins."""
    frame = synthetic_features(200)
    frame = pd.concat([frame.iloc[::2], swap_corners(frame.iloc[1::2])])
    labels = (frame["height_in_diff"] > 0).astype(int)
    preprocessor = build_preprocessor(FEATURE_COLUMNS)
    with warnings.catch_warnings():
        # The synthetic rows only fill a few feature columns.
        warnings.filterwarnings("ignore", message="Skipping features without any observed values")
        X = preprocessor.fit_transform(select_features(frame))
    model = LogisticRegression().fit(X, labels)
    joblib.dump({"model": model, "preprocessor": preprocessor, "feature_columns": FEATURE_COLUMNS,
                 "trained_at": "2026-10-02T00:00:00+00:00"}, tmp_path / "tiny.joblib")
    monkeypatch.setattr(predictor, "MODELS_DIR", tmp_path)
    monkeypatch.setattr(ml_utils, "MODELS_DIR", tmp_path)
    return "tiny"
