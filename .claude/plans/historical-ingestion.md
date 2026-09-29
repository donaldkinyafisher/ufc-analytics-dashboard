# Historical fight ingestion + fighter database (ufcstats.com)

## Context
Right now the only scraping is for *upcoming* events (`POST /api/v1/events/upcoming/sync`). `src/data/historical_fights.csv` is only a sample and is not used. We are building the training database from scratch with the scraper: crawl every completed UFC event (newest → UFC 1), store every fight + per-fighter fight stats, and keep a fighter table current. It must support a `since` date to limit a backfill (e.g. last year only, for testing) and an incremental "update" mode that only adds what's new.

Crawl path: `/statistics/events/completed?page=all` → `/event-details/<code>` → `/fight-details/<code>` → `/fighter-details/<code>`.

## Working order (user-paced)
First, copy this plan to `.claude/plans/historical-ingestion.md` in the project. After that we do one step at a time, and you pick the next one.

**Step 1 (this session): Playwright fetch + parsers only. No DB or API changes.**
- `src/scrapers/ufcstats_browser.py`: a shared-browser `fetch_rendered(url, wait_selector)`.
- `src/scrapers/historical_scraper.py`: `parse_completed_events`, `parse_event_details`, `parse_fight_details`, `parse_fighter_details`.
- `src/scrapers/save_samples.py`: renders one real page of each type (plus a draw/NC fight) into `tests/fixtures/`, then prints the parsed dicts so you can check them by eye.
- Tests: `tests/test_historical_scraper.py`, which runs the parsers against the saved fixtures (`uv run pytest`).

Later steps are sections 4 → 3 → 5 below. Each one waits until you say go.

## Recommended approach

### 1. REST shape: scraping is a *job resource*
A full backfill is ~750 events + ~8.5k fight pages + ~2.5k fighter pages, and in a headless browser that takes on the order of an hour or more, so it can't be a blocking request like the upcoming-events sync. Model it as a job:

| Method | Path | Purpose |
|---|---|---|
| `POST` | `/api/v1/ingestion/jobs` | Start a job → `202 Accepted`, `Location: /api/v1/ingestion/jobs/{id}`. `409` if one is already running. |
| `GET` | `/api/v1/ingestion/jobs` | List jobs (newest first). |
| `GET` | `/api/v1/ingestion/jobs/{id}` | Status + progress counters + error. |
| `GET` | `/api/v1/events?status=completed&since=&until=` | Stored events (generalises existing `/upcoming`). |
| `GET` | `/api/v1/events/{id}` | Event with its fights. |
| `GET` | `/api/v1/fights/{id}` | Fight with both corners' statistics. |
| `GET` | `/api/v1/fighters?name=` · `/fighters/{id}` · `/fighters/{id}/fights` | Fighter profile, career stats, fight history. |

Job request body (`IngestionJobCreate`):
```json
{ "mode": "backfill" | "incremental",
  "since": "2025-09-28",          // optional lower bound on event_date (backfill); default = UFC 1
  "until": null,                   // optional upper bound; default = today
  "refresh_existing": false,       // re-scrape events already fully ingested
  "refresh_fighters": true }       // fetch profile pages for fighters touched by this job
```
- **backfill**: all completed events with `since <= event_date <= until`.
- **incremental**: events not yet ingested, dated on/after the **oldest** ingested event (fills gaps and retries failures inside the stored range, never extends it backwards). Fails with "run a backfill first" on an empty DB. This is the "update the training db" path, and what a cron/Streamlit button would call.

Jobs run in-process via FastAPI `BackgroundTasks`, guarded by an `asyncio.Lock` like `_refresh_lock` in [eventsRouter.py](src/api/v1/routers/eventsRouter.py). Progress is persisted to a `scrape_jobs` table so it survives a restart and is visible via `GET`.

### 2. Scraper layer (`src/scrapers/`)
ufcstats pages are rendered with JavaScript, so every fetch goes through **Playwright** (async API, matching [upcoming_events_scraper.py](src/scrapers/upcoming_events_scraper.py)).

- `ufcstats_browser.py`: an async context manager that launches **one** Chromium browser per job with the same user agent. `fetch_rendered(url, wait_selector)` opens a page from a small pool (`asyncio.Semaphore(4)`), runs `goto(..., wait_until="domcontentloaded")` then `wait_for_selector`, returns `page.content()` and closes the page. It retries with backoff on timeouts. The base URL comes from `Settings.ufc_stats_base_url` ([config.py](src/api/v1/config.py)). Launching the browser once per job, instead of once per page as the upcoming scraper does, keeps a full backfill of about 12k pages practical.
- `historical_scraper.py`: **pure parse functions** (`rendered html -> dict`) that parse the DOM *after* Playwright has rendered it, plus thin async fetchers. Plain HTTP + BeautifulSoup returns nothing on these pages. Running BS4/selectors on Playwright's rendered HTML is fine, and it means the parsers can be unit-tested offline against saved rendered pages. The wait selectors per page type are `table.b-statistics__table-events`, `.b-fight-details__table`, `.b-fight-details__person` and `.b-list__info-box`.
  - `parse_completed_events(html)` → `[{name, date, location, source_url, ufcstats_id}]` (list is newest-first, so the crawl stops as soon as `date < since`). Reuse `parse_event_date` from [upcoming_events_scraper.py](src/scrapers/upcoming_events_scraper.py).
  - `parse_event_details(html)` → event meta + fight-detail URLs.
  - `parse_fight_details(html)` → both fighters (name, fighter URL, W/L/D/NC), bout type/weight class, method, round, time, format, referee, and for each fighter the *Totals* and *Significant Strikes* tables as landed/attempted counts.
  - `parse_fighter_details(html)` → height, weight, reach, stance, DOB, record, SLpM/Str.Acc/SApM/Str.Def/TD Avg/TD Acc/TD Def/Sub Avg.
  - `ufcstats_id` = the hex code at the end of each URL; it's the stable natural key everywhere.

### 3. Service layer (`src/api/v1/services/ingestionServices.py`)
Follows the style of [eventsServices.py](src/api/v1/services/eventsServices.py) (upsert helpers, dataclass result). Flow per job:
1. Resolve date window (mode + `since`/`until`), fetch the completed-events list.
2. For each event in the window (skip if `scraped_at` set and not `refresh_existing`):
   - fetch event page, then its fight pages concurrently;
   - **one transaction per event**: upsert `Event` (status `completed`), upsert both `Fighter` stubs by `ufcstats_id` (name + profile_url), upsert `Fight`, upsert two `FightStatistic` rows; set `event.scraped_at`; commit; update job counters.
   - Collect touched fighter ids in a set.
3. If `refresh_fighters`: fetch each touched fighter's profile once (deduped), update `Fighter` + upsert `FighterCareerStats`, set `last_synced_at`.
4. Mark job `succeeded`/`failed`; per-page failures are logged into the job's `errors` count/list rather than aborting the whole run.

Per-event commits make backfills **resumable**: re-running skips completed events. The existing upcoming sync should be taught to match on `ufcstats_id`/`source_url` so an event that was "upcoming" gets promoted rather than duplicated (it already matches on `source_url`, which is the same URL).

A CLI wrapper `python -m src.scrapers.ingest --since 2025-09-28` calls the same service for testing without the server.

### 4. Model changes ([models.py](src/api/v1/models.py))
- `Event`: add `ufcstats_id` (unique), `scraped_at` (nullable; null = fights not yet ingested).
- `Fighter`: add `ufcstats_id` (unique), `nickname`, `no_contests`; parse `dob` to `Date` and height/reach to inches (floats) so they're usable as features.
- `Fight`: add `ufcstats_id` (unique), `referee`, `method_details`; make `winner_side` nullable-safe for draws/NC (`'red'|'blue'|'draw'|'nc'`).
- `FightStatistic`: replace pct-only columns with raw counts — `sig_str_landed/attempted`, `total_str_landed/attempted`, `td_landed/attempted`, head/body/leg/distance/clinch/ground `_landed/_attempted`, `control_time_seconds` (int), plus existing `knockdowns`, `submission_attempts`, `reversals`. Percentages are derived, and counts are what career / pre-fight rolling aggregates need.
- New `ScrapeJob`: `id, mode, since, until, refresh_existing, status, events_total, events_done, fights_upserted, fighters_upserted, errors (int), error_log (Text/JSON), created_at, started_at, finished_at`.

Schema migration: `Base.metadata.create_all` won't alter existing tables. The local SQLite DB only holds upcoming events (re-syncable), so drop `ufc_analytics.db` and recreate. Adding Alembic is worth doing next, but it's out of scope here.

### 5. Schemas / routers
- `src/api/v1/schemas/ingestionSchema.py`: `IngestionJobCreate` (validate `since <= until`, `since` not in future), `IngestionJobResponse`.
- `src/api/v1/routers/ingestionRouter.py` mounted at `/api/v1/ingestion` in [main.py](src/api/v1/main.py).
- Extend `eventsRouter` with `GET /` and `GET /{id}`. Add a small `fightsRouter`.
- Fix the existing untracked [fighters.py](src/api/v1/routers/fighters.py): it `await`s an `AsyncSession` while `get_db` yields a sync `Session`. Make it sync, and align [schemas/fighter.py](src/api/v1/schemas/fighter.py) with the model (it has `firstName/lastName`, but the model has `name`).

### Known data caveat (flag, not solved here)
- ufcstats tends to list the **winner first** on fight pages. If "first listed = red" is stored naively, red wins ~100% and the model leaks the label. Store the corner order exactly as the page shows it, and handle red/blue balancing (random swap) in the training-dataset export. Verify on a few fights with a known blue-corner winner.
- `FighterCareerStats` from profile pages are *current* aggregates, so using them as features for old fights leaks the future. Storing per-fight counts lets you compute pre-fight rolling features later.

## Files
New: `src/scrapers/ufcstats_browser.py`, `src/scrapers/historical_scraper.py`, `src/scrapers/ingest.py` (CLI), `src/api/v1/services/ingestionServices.py`, `src/api/v1/schemas/ingestionSchema.py`, `src/api/v1/routers/ingestionRouter.py`, `src/api/v1/routers/fightsRouter.py`, `tests/fixtures/*.html` + `tests/test_historical_scraper.py`, `tests/test_ingestion_service.py`.
Modified: `src/api/v1/models.py`, `src/api/v1/main.py`, `src/api/v1/routers/eventsRouter.py`, `src/api/v1/services/eventsServices.py`, `src/api/v1/routers/fighters.py`, `src/api/v1/schemas/fighter.py`, `pyproject.toml` (add `pytest` dev dep), `.env.example`.

## Verification
1. **Parser unit tests**: use Playwright to save one *rendered* page (`page.content()`) of each type into `tests/fixtures/` (including a draw/NC fight and a fighter with missing reach/DOB). Assert the parsed dicts. Run `uv run pytest`.
2. **Service test**: in-memory SQLite, patch the fetchers to return fixtures. Run ingestion twice and assert there are no duplicates (idempotent) and that `scraped_at` is set.
3. **Live small backfill**: `uv run uvicorn src.api.v1.main:app --reload`, then `POST /api/v1/ingestion/jobs {"mode":"backfill","since":"2025-09-28"}`. Poll `GET /jobs/{id}` until `succeeded`. Check `/api/v1/events?status=completed` (~40 events), a fight's stats, and a fighter's profile + `/fights`.
4. **Incremental**: run `{"mode":"incremental"}` right after. Expect 0 new events. Delete the newest event's `scraped_at` and re-run: exactly that event is re-ingested.
5. Spot-check 3–5 stored fights against the live ufcstats fight pages in a browser (method, round, sig-strike counts, winner).

## Status
- [x] Step 1: Playwright fetch + parsers, fixtures, tests.
- [x] Step 2: Model changes (section 4) applied as proposed; old `ufc_analytics.db` deleted and recreated.
- [x] Step 3: Ingestion service (section 3) + CLI `src/scrapers/ingest.py` + tests. Decisions: page failures → job `succeeded` with `errors`; fighter W/L from profile page (incl. non-UFC); incremental = "only new events" as defined above.
- [x] Step 4: Schemas / routers (section 5): ingestion jobs, events list/detail, fights detail, fighters search/profile/fights; stale jobs failed at startup; `tests/test_api.py`.
- Deferred: per-round stats (`FightRoundStatistic`) — see `todo.md`.
