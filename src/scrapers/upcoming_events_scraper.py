"""
Shared scraping logic for UFC upcoming events.

Used by both the CLI script (scrape_ufc_events.py) and the REST API (api.py).
Uses Playwright's ASYNC API so it plays nicely inside an async web framework
like FastAPI without blocking the event loop.
"""

from datetime import datetime

from playwright.async_api import async_playwright

UPCOMING_URL = "http://ufcstats.com/statistics/events/upcoming"


def parse_event_date(raw_date: str) -> datetime | None:
    raw_date = " ".join(raw_date.split())
    for fmt in ("%B %d, %Y", "%b %d, %Y"):
        try:
            return datetime.strptime(raw_date, fmt)
        except ValueError:
            continue
    return None


async def scrape_upcoming_events(headless: bool = True) -> list[dict]:
    """Async version: launches Chromium, scrapes the table, returns raw events."""
    events = []

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=headless)
        page = await browser.new_page(
            user_agent=(
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/124.0.0.0 Safari/537.36"
            )
        )

        await page.goto(UPCOMING_URL, wait_until="domcontentloaded", timeout=30000)
        await page.wait_for_selector("table.b-statistics__table-events", timeout=15000)

        rows = await page.query_selector_all(
            "table.b-statistics__table-events tbody tr.b-statistics__table-row"
        )

        for row in rows:
            cells = await row.query_selector_all("td")
            if len(cells) < 2:
                continue

            name_el = await cells[0].query_selector("a.b-link")
            if name_el is None:
                continue
            event_name = (await name_el.inner_text()).strip()
            source_url = await name_el.get_attribute("href")

            date_el = await cells[0].query_selector("span.b-statistics__date")
            date_text = (await date_el.inner_text()).strip() if date_el else ""

            location = (await cells[1].inner_text()).strip()

            events.append(
                {
                    "event_name": event_name,
                    "date": date_text,
                    "location": location,
                    "source_url": source_url,
                }
            )

        await browser.close()

    return events


def filter_events_after(events: list[dict], reference_date: datetime) -> list[dict]:
    filtered = []
    for event in events:
        parsed = parse_event_date(event["date"])
        if parsed is None:
            continue
        if parsed.date() > reference_date.date():
            filtered.append(event)
    return filtered
