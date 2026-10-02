"""Shared test data: parsed fixture pages and URLs."""

from pathlib import Path

import numpy as np
import pandas as pd
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from src.scrapers.historical_scraper import parse_event_details, parse_fight_details

FIXTURES = Path(__file__).parent / "fixtures"
BASE = "http://ufcstats.com"

EVENT_URL = f"{BASE}/event-details/8a0a35e7c74bebcc"
FIGHT_URLS = {
    "fight_title_decision.html": f"{BASE}/fight-details/568ec6af4008355a",
    "fight_no_contest.html": f"{BASE}/fight-details/5ab286735ccee803",
    "fight_draw.html": f"{BASE}/fight-details/c258996570faeec9",
}
VAN_URL = f"{BASE}/fighter-details/17e97649403ba428"
SMITH_URL = f"{BASE}/fighter-details/46c8ec317aff28ac"


def fixture(name: str) -> str:
    return (FIXTURES / name).read_text(encoding="utf-8")


def event_data() -> dict:
    """UFC 331 metadata with three parsed fight fixtures as its card."""
    event = parse_event_details(fixture("event_ufc331.html"), EVENT_URL)
    event["fights"] = [
        {**parse_fight_details(fixture(name), url), "bout_order": order}
        for order, (name, url) in enumerate(FIGHT_URLS.items(), start=1)
    ]
    return event


def count(db: Session, model) -> int:
    return db.scalar(select(func.count()).select_from(model))


class FakeBrowser:
    """Stands in for UFCStatsBrowser when fetch functions are patched."""

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc_info):
        return None


def synthetic_features(n: int = 100) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    frame = pd.DataFrame({
        "fight_id": range(n),
        "event_date": pd.date_range("2020-01-01", periods=n, freq="7D"),
        "winner_side": "red",  # ufcstats lists the winner first
        "red_name": [f"R{i}" for i in range(n)],
        "blue_name": [f"B{i}" for i in range(n)],
        "red_height_in": rng.normal(72, 3, n),
        "blue_height_in": rng.normal(70, 3, n),
        "red_stance": "Orthodox",
        "blue_stance": "Southpaw",
        "sex": "Men",
    })
    frame["height_in_diff"] = frame["red_height_in"] - frame["blue_height_in"]
    return frame
