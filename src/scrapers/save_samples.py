"""
Render one real ufcstats page of each type, save it as a test fixture and
print what the parser extracts from it.

Run:
    uv run python -m src.scrapers.save_samples            # save + print
    uv run python -m src.scrapers.save_samples --no-save  # print only
    uv run python -m src.scrapers.save_samples --only event_upcoming.html
"""

import argparse
import asyncio
from pathlib import Path
from pprint import pprint

from src.scrapers.historical_scraper import (
    COMPLETED_EVENTS_SELECTOR,
    EVENT_DETAILS_SELECTOR,
    FIGHT_DETAILS_SELECTOR,
    FIGHTER_DETAILS_SELECTOR,
    parse_completed_events,
    parse_event_details,
    parse_fight_details,
    parse_fighter_details,
)
from src.scrapers.ufcstats_browser import UFCStatsBrowser

FIXTURES_DIR = Path(__file__).resolve().parents[2] / "tests" / "fixtures"

BASE = "http://ufcstats.com"

# fixture name -> (url, selector to wait for, parser)
SAMPLES = {
    # page=1 has the same markup as page=all, at a fraction of the size.
    "completed_events.html": (
        f"{BASE}/statistics/events/completed?page=1",
        COMPLETED_EVENTS_SELECTOR,
        parse_completed_events,
    ),
    # UFC 331: Van vs. Pantoja 2
    "event_ufc331.html": (
        f"{BASE}/event-details/8a0a35e7c74bebcc",
        EVENT_DETAILS_SELECTOR,
        parse_event_details,
    ),
    # UFC 332: Silva vs. Wang, an upcoming card (fighters and weight classes, no results).
    "event_upcoming.html": (
        f"{BASE}/event-details/ad3fdba28a7540cf",
        EVENT_DETAILS_SELECTOR,
        parse_event_details,
    ),
    # Van vs. Pantoja 2: title bout, unanimous decision, full stats.
    "fight_title_decision.html": (
        f"{BASE}/fight-details/568ec6af4008355a",
        FIGHT_DETAILS_SELECTOR,
        parse_fight_details,
    ),
    # UFC 2 Gracie vs. Smith: no weight class, no control time.
    "fight_early_era.html": (
        f"{BASE}/fight-details/00835554f95fa911",
        FIGHT_DETAILS_SELECTOR,
        parse_fight_details,
    ),
    # UFC 214 Cormier vs. Jones 2: overturned to a no contest.
    "fight_no_contest.html": (
        f"{BASE}/fight-details/5ab286735ccee803",
        FIGHT_DETAILS_SELECTOR,
        parse_fight_details,
    ),
    # UFC 256 Figueiredo vs. Moreno 1: majority draw.
    "fight_draw.html": (
        f"{BASE}/fight-details/c258996570faeec9",
        FIGHT_DETAILS_SELECTOR,
        parse_fight_details,
    ),
    # Joshua Van: complete profile.
    "fighter_complete.html": (
        f"{BASE}/fighter-details/17e97649403ba428",
        FIGHTER_DETAILS_SELECTOR,
        parse_fighter_details,
    ),
    # Patrick Smith: early-era profile with missing attributes.
    "fighter_sparse.html": (
        f"{BASE}/fighter-details/46c8ec317aff28ac",
        FIGHTER_DETAILS_SELECTOR,
        parse_fighter_details,
    ),
}


async def main(save: bool, only: list[str] | None = None) -> None:
    if save:
        FIXTURES_DIR.mkdir(parents=True, exist_ok=True)

    samples = {name: sample for name, sample in SAMPLES.items() if not only or name in only}
    async with UFCStatsBrowser() as browser:
        pages = await asyncio.gather(
            *(browser.fetch_rendered(url, selector) for url, selector, _ in samples.values())
        )

    for (name, (url, _, parser)), html in zip(samples.items(), pages):
        if save:
            (FIXTURES_DIR / name).write_text(html, encoding="utf-8")
        print(f"\n===== {name}  ({url})")
        parsed = parser(html, url) if parser is not parse_completed_events else parser(html)[:3]
        pprint(parsed, sort_dicts=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-save", action="store_true", help="print parsed output only")
    parser.add_argument("--only", nargs="+", choices=SAMPLES, help="fixture names to fetch")
    args = parser.parse_args()
    asyncio.run(main(save=not args.no_save, only=args.only))
