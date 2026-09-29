"""
Scraping logic for completed UFC events, fights and fighters on ufcstats.com.

Crawl path:
    /statistics/events/completed?page=all -> /event-details/<id>
        -> /fight-details/<id> -> /fighter-details/<id>

The parse_* functions are pure (rendered HTML in, dict out) so they can be
tested offline against saved pages. The fetch_* functions render a page with
a shared UFCStatsBrowser and hand the HTML to the matching parser.

Fighter order on a fight page is stored as-is: the first fighter is "red",
the second "blue". ufcstats usually lists the winner first, so corner order
must be balanced before training.
"""

import asyncio
import re
from datetime import date

from bs4 import BeautifulSoup, Tag

from src.scrapers.ufcstats_browser import UFCStatsBrowser
from src.scrapers.upcoming_events_scraper import parse_event_date

COMPLETED_EVENTS_PATH = "/statistics/events/completed?page=all"

COMPLETED_EVENTS_SELECTOR = "table.b-statistics__table-events"
EVENT_DETAILS_SELECTOR = "table.b-fight-details__table"
FIGHT_DETAILS_SELECTOR = ".b-fight-details__person"
FIGHTER_DETAILS_SELECTOR = ".b-list__info-box"

MISSING_VALUES = {"", "-", "--", "---"}

RESULT_CODES = {"W": "W", "L": "L", "D": "D", "NC": "NC"}

WEIGHT_CLASS_PATTERN = re.compile(
    r"(Women's )?(Strawweight|Flyweight|Bantamweight|Featherweight|Lightweight|"
    r"Welterweight|Middleweight|Light Heavyweight|Super Heavyweight|Heavyweight|"
    r"Catch ?Weight|Open ?Weight)",
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# Value helpers
# ---------------------------------------------------------------------------

def ufcstats_id(url: str | None) -> str | None:
    """Return the hex code at the end of a ufcstats URL."""
    if not url:
        return None
    return url.rstrip("/").rsplit("/", 1)[-1] or None


def _clean(text: str | None) -> str:
    return " ".join(text.split()) if text else ""


def _or_none(text: str | None) -> str | None:
    text = _clean(text)
    return None if text in MISSING_VALUES else text


def _to_int(text: str | None) -> int | None:
    text = _or_none(text)
    try:
        return int(text) if text is not None else None
    except ValueError:
        return None


def _to_float(text: str | None) -> float | None:
    text = _or_none(text)
    try:
        return float(text) if text is not None else None
    except ValueError:
        return None


def _pct(text: str | None) -> float | None:
    """'57%' -> 0.57"""
    text = _or_none(text)
    if text is None:
        return None
    value = _to_float(text.rstrip("%"))
    return round(value / 100, 4) if value is not None else None


def _landed_attempted(text: str | None) -> tuple[int | None, int | None]:
    """'181 of 305' -> (181, 305)"""
    match = re.fullmatch(r"(\d+)\s+of\s+(\d+)", _clean(text))
    if not match:
        return None, None
    return int(match.group(1)), int(match.group(2))


def _seconds(text: str | None) -> int | None:
    """'2:13' -> 133"""
    match = re.fullmatch(r"(\d+):(\d{2})", _clean(text))
    if not match:
        return None
    return int(match.group(1)) * 60 + int(match.group(2))


def _inches(text: str | None) -> float | None:
    """Height '5\\' 5"' -> 65.0, reach '65"' -> 65.0"""
    text = _or_none(text)
    if text is None:
        return None
    match = re.fullmatch(r"(?:(\d+)'\s*)?(\d+(?:\.\d+)?)\"?", text)
    if not match:
        return None
    feet = int(match.group(1)) if match.group(1) else 0
    return feet * 12 + float(match.group(2))


def _pounds(text: str | None) -> float | None:
    """'125 lbs.' -> 125.0"""
    match = re.match(r"(\d+(?:\.\d+)?)", _clean(text))
    return float(match.group(1)) if match else None


def _parse_date(text: str | None) -> date | None:
    parsed = parse_event_date(_clean(text))
    return parsed.date() if parsed else None


def _labelled_value(item: Tag) -> str:
    """Text of a 'Label: value' element with the label removed."""
    label = item.select_one(".b-list__box-item-title, .b-fight-details__label")
    text = _clean(item.get_text(" "))
    if label is not None:
        text = _clean(text.removeprefix(_clean(label.get_text(" "))))
    return text


def _column_values(cell: Tag) -> list[str]:
    """Each <p> in a fight table cell holds one fighter's value."""
    return [_clean(p.get_text(" ")) for p in cell.select("p.b-fight-details__table-text")]


# ---------------------------------------------------------------------------
# Parsers
# ---------------------------------------------------------------------------

def parse_completed_events(html: str) -> list[dict]:
    """Parse /statistics/events/completed into events, newest first.

    `is_next` marks the row ufcstats flags as the next/most recent card;
    its results may not be published yet.
    """
    soup = BeautifulSoup(html, "html.parser")
    events = []
    for row in soup.select(f"{COMPLETED_EVENTS_SELECTOR} tbody tr.b-statistics__table-row"):
        link = row.select_one("a.b-link")
        cells = row.select("td")
        if link is None or len(cells) < 2:
            continue
        source_url = link.get("href")
        date_el = row.select_one("span.b-statistics__date")
        events.append(
            {
                "ufcstats_id": ufcstats_id(source_url),
                "name": _clean(link.get_text()),
                "event_date": _parse_date(date_el.get_text() if date_el else None),
                "location": _or_none(cells[1].get_text()),
                "source_url": source_url,
                "is_next": row.select_one("img[src*='next']") is not None,
            }
        )
    return events


def parse_event_details(html: str, source_url: str | None = None) -> dict:
    """Parse /event-details/<id> into event metadata and its fight links.

    `bout_order` is 1 for the main event and increases down the card.
    """
    soup = BeautifulSoup(html, "html.parser")
    info = {}
    for item in soup.select("ul.b-list__box-list li.b-list__box-list-item"):
        label = item.select_one(".b-list__box-item-title")
        if label is not None:
            info[_clean(label.get_text()).rstrip(":").lower()] = _labelled_value(item)

    fights = []
    for order, row in enumerate(soup.select(f"{EVENT_DETAILS_SELECTOR} tbody tr"), start=1):
        fight_url = row.get("data-link")
        if not fight_url:
            continue
        fighters = [_clean(a.get_text()) for a in row.select("a[href*='fighter-details']")]
        fights.append(
            {
                "ufcstats_id": ufcstats_id(fight_url),
                "source_url": fight_url,
                "bout_order": order,
                "fighter_names": fighters,
            }
        )

    title = soup.select_one(".b-content__title")
    return {
        "ufcstats_id": ufcstats_id(source_url),
        "name": _clean(title.get_text()) if title else None,
        "event_date": _parse_date(info.get("date")),
        "location": _or_none(info.get("location")),
        "source_url": source_url,
        "fights": fights,
    }


def _parse_fight_fighters(soup: BeautifulSoup) -> list[dict]:
    fighters = []
    for corner, person in zip(("red", "blue"), soup.select(FIGHT_DETAILS_SELECTOR)):
        link = person.select_one("a.b-fight-details__person-link") or person.select_one(
            ".b-fight-details__person-name"
        )
        status = person.select_one(".b-fight-details__person-status")
        nickname = person.select_one(".b-fight-details__person-title")
        profile_url = link.get("href") if link is not None and link.name == "a" else None
        fighters.append(
            {
                "corner": corner,
                "name": _clean(link.get_text()) if link is not None else None,
                "ufcstats_id": ufcstats_id(profile_url),
                "profile_url": profile_url,
                "nickname": _or_none(nickname.get_text().strip().strip('"')) if nickname else None,
                "result": RESULT_CODES.get(_clean(status.get_text()).upper()) if status else None,
            }
        )
    return fighters


def _parse_fight_meta(soup: BeautifulSoup) -> dict:
    meta: dict = {}
    content = soup.select_one(".b-fight-details__content")
    if content is None:
        return meta

    paragraphs = content.select("p.b-fight-details__text")
    if paragraphs:
        for item in paragraphs[0].select(
            "i.b-fight-details__text-item_first, i.b-fight-details__text-item"
        ):
            label = item.select_one(".b-fight-details__label")
            if label is not None:
                meta[_clean(label.get_text()).rstrip(":").lower()] = _labelled_value(item)
    if len(paragraphs) > 1:
        meta["details"] = _labelled_value(paragraphs[1])
    return meta


def _stats_tables(soup: BeautifulSoup) -> tuple[Tag | None, Tag | None]:
    """Return the fight-total tables (Totals, Significant Strikes).

    Per-round tables carry the `js-fight-table` class and are skipped.
    """
    totals = sig_strikes = None
    for table in soup.select("table"):
        if "js-fight-table" in (table.get("class") or []):
            continue
        header = _clean(table.select_one("thead").get_text(" ")) if table.select_one("thead") else ""
        if "KD" in header and totals is None:
            totals = table
        elif "Head" in header and sig_strikes is None:
            sig_strikes = table
    return totals, sig_strikes


def _table_columns(table: Tag) -> dict[str, list[str]]:
    """Map each header to its per-fighter values: {'KD': ['1', '0'], ...}."""
    headers = [_clean(th.get_text()) for th in table.select("thead th")]
    row = table.select_one("tbody tr")
    if row is None:
        return {}
    cells = row.select("td")
    columns = {}
    for header, cell in zip(headers, cells):
        # Keep the first column for a repeated header.
        columns.setdefault(header, _column_values(cell))
    return columns


def _value(columns: dict[str, list[str]], header: str, index: int) -> str | None:
    values = columns.get(header) or []
    return values[index] if index < len(values) else None


def _parse_fight_stats(soup: BeautifulSoup) -> dict[str, dict | None]:
    totals, sig_strikes = _stats_tables(soup)
    totals_cols = _table_columns(totals) if totals is not None else {}
    sig_cols = _table_columns(sig_strikes) if sig_strikes is not None else {}
    if not totals_cols and not sig_cols:
        return {"red": None, "blue": None}

    stats: dict[str, dict | None] = {}
    for index, corner in enumerate(("red", "blue")):
        sig_landed, sig_attempted = _landed_attempted(_value(totals_cols, "Sig. str.", index))
        total_landed, total_attempted = _landed_attempted(_value(totals_cols, "Total str.", index))
        td_landed, td_attempted = _landed_attempted(_value(totals_cols, "Td", index))
        corner_stats = {
            "knockdowns": _to_int(_value(totals_cols, "KD", index)),
            "sig_str_landed": sig_landed,
            "sig_str_attempted": sig_attempted,
            "total_str_landed": total_landed,
            "total_str_attempted": total_attempted,
            "td_landed": td_landed,
            "td_attempted": td_attempted,
            "submission_attempts": _to_int(_value(totals_cols, "Sub. att", index)),
            "reversals": _to_int(_value(totals_cols, "Rev.", index)),
            "control_time_seconds": _seconds(_value(totals_cols, "Ctrl", index)),
        }
        for target in ("Head", "Body", "Leg", "Distance", "Clinch", "Ground"):
            landed, attempted = _landed_attempted(_value(sig_cols, target, index))
            corner_stats[f"{target.lower()}_landed"] = landed
            corner_stats[f"{target.lower()}_attempted"] = attempted
        stats[corner] = corner_stats
    return stats


def _winner_side(fighters: list[dict]) -> str | None:
    results = [fighter["result"] for fighter in fighters]
    if len(results) != 2:
        return None
    if results[0] == "W":
        return "red"
    if results[1] == "W":
        return "blue"
    if "D" in results:
        return "draw"
    if "NC" in results:
        return "nc"
    return None


def parse_fight_details(html: str, source_url: str | None = None) -> dict:
    """Parse /fight-details/<id> into bout info, both fighters and fight totals."""
    soup = BeautifulSoup(html, "html.parser")
    fighters = _parse_fight_fighters(soup)
    meta = _parse_fight_meta(soup)

    title = soup.select_one(".b-fight-details__fight-title")
    bout_type = _clean(title.get_text()) if title else None
    weight_match = WEIGHT_CLASS_PATTERN.search(bout_type or "")

    event_link = soup.select_one("h2.b-content__title a")
    event_url = event_link.get("href") if event_link else None

    return {
        "ufcstats_id": ufcstats_id(source_url),
        "source_url": source_url,
        "event_ufcstats_id": ufcstats_id(event_url),
        "event_name": _clean(event_link.get_text()) if event_link else None,
        "bout_type": bout_type,
        "weight_class": _clean(weight_match.group(0)) if weight_match else None,
        "is_title_bout": bool(bout_type and "title" in bout_type.lower()),
        "method": _or_none(meta.get("method")),
        "method_details": _or_none(meta.get("details")),
        "round": _to_int(meta.get("round")),
        "time": _or_none(meta.get("time")),
        "time_format": _or_none(meta.get("time format")),
        "referee": _or_none(meta.get("referee")),
        "winner_side": _winner_side(fighters),
        "fighters": fighters,
        "stats": _parse_fight_stats(soup),
    }


def _parse_record(text: str | None) -> dict:
    """'Record: 18-2-0 (1 NC)' -> wins/losses/draws/no_contests"""
    match = re.search(r"(\d+)-(\d+)-(\d+)(?:\s*\((\d+)\s*NC\))?", text or "")
    if not match:
        return {"wins": None, "losses": None, "draws": None, "no_contests": None}
    return {
        "wins": int(match.group(1)),
        "losses": int(match.group(2)),
        "draws": int(match.group(3)),
        "no_contests": int(match.group(4)) if match.group(4) else 0,
    }


def parse_fighter_details(html: str, source_url: str | None = None) -> dict:
    """Parse /fighter-details/<id> into physical attributes, record and career stats."""
    soup = BeautifulSoup(html, "html.parser")

    info = {}
    for item in soup.select(f"{FIGHTER_DETAILS_SELECTOR} li.b-list__box-list-item"):
        label = item.select_one(".b-list__box-item-title")
        if label is not None:
            info[_clean(label.get_text()).rstrip(":").lower()] = _labelled_value(item)

    name = soup.select_one(".b-content__title-highlight")
    record = soup.select_one(".b-content__title-record")
    nickname = soup.select_one(".b-content__Nickname")

    return {
        "ufcstats_id": ufcstats_id(source_url),
        "profile_url": source_url,
        "name": _clean(name.get_text()) if name else None,
        "nickname": _or_none(nickname.get_text()) if nickname else None,
        **_parse_record(record.get_text() if record else None),
        "height_in": _inches(info.get("height")),
        "weight_lbs": _pounds(info.get("weight")),
        "reach_in": _inches(info.get("reach")),
        "stance": _or_none(info.get("stance")),
        "dob": _parse_date(info.get("dob")),
        "career_stats": {
            "strikes_landed_per_minute": _to_float(info.get("slpm")),
            "striking_accuracy": _pct(info.get("str. acc.")),
            "strikes_absorbed_per_minute": _to_float(info.get("sapm")),
            "striking_defense": _pct(info.get("str. def")),
            "takedown_average": _to_float(info.get("td avg.")),
            "takedown_accuracy": _pct(info.get("td acc.")),
            "takedown_defense": _pct(info.get("td def.")),
            "submission_average": _to_float(info.get("sub. avg.")),
        },
    }


# ---------------------------------------------------------------------------
# Fetchers
# ---------------------------------------------------------------------------

async def fetch_completed_events(
    browser: UFCStatsBrowser,
    base_url: str,
    since: date | None = None,
    until: date | None = None,
) -> list[dict]:
    """Completed events with since <= event_date <= until, newest first.

    Rows flagged as the next card and rows without a parseable date are skipped.
    """
    html = await browser.fetch_rendered(
        base_url.rstrip("/") + COMPLETED_EVENTS_PATH, COMPLETED_EVENTS_SELECTOR
    )
    events = []
    for event in parse_completed_events(html):
        event_date = event["event_date"]
        if event["is_next"] or event_date is None:
            continue
        if until is not None and event_date > until:
            continue
        if since is not None and event_date < since:
            break  # list is newest first
        events.append(event)
    return events


async def fetch_event(browser: UFCStatsBrowser, url: str) -> dict:
    html = await browser.fetch_rendered(url, EVENT_DETAILS_SELECTOR)
    return parse_event_details(html, url)


async def fetch_fight(browser: UFCStatsBrowser, url: str) -> dict:
    html = await browser.fetch_rendered(url, FIGHT_DETAILS_SELECTOR)
    return parse_fight_details(html, url)


async def fetch_fighter(browser: UFCStatsBrowser, url: str) -> dict:
    html = await browser.fetch_rendered(url, FIGHTER_DETAILS_SELECTOR)
    return parse_fighter_details(html, url)


async def fetch_event_with_fights(browser: UFCStatsBrowser, url: str) -> dict:
    """Fetch an event and all its fights; fights are fetched concurrently."""
    event = await fetch_event(browser, url)
    event["fights"] = await asyncio.gather(
        *(fetch_fight(browser, fight["source_url"]) for fight in event["fights"])
    )
    for order, fight in enumerate(event["fights"], start=1):
        fight["bout_order"] = order
    return event
