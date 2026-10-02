# UFC Data Analytics Dashboard

Streamlit project for collecting UFC fight data, displaying analytics, and storing feature-ready records for prediction models.

## Setup

Environment is handled through [uv package manager](https://docs.astral.sh/uv/). Package dependencies can be completed handled through `uv sync`. See web docs for more details

```bash
git clone <repo-name>
uv sync
uv run playwright install chromium   # browser used to scrape ufcstats.com
```

## Running locally

Run the API and the Streamlit app in separate terminal windows. Start the API first; the app reads everything through it.

```bash
uv run fastapi dev --entrypoint src.api.v1.main:app
```

```bash
uv run streamlit run src/streamlit/main.py
```

A deployed version does not exist yet.

## First run

The fight database and the trained models are built locally and are not in git (`ufc_analytics.db` and `src/ml/artifacts/` are ignored). On a fresh clone:

1. **Build the database.** While the database is empty, the app shows only a **Setup** page. Choose how far back to scrape (last 5 years, last 10 years or full history) and click **Build database**. A 5-year scrape takes about 15 minutes; full history takes about an hour. The scrape runs in the API, so you can close the browser tab, but keep the API running. If it fails or the API stops, click **Retry**: events already stored are kept, so it continues where it stopped.
2. **Train a model.** When the scrape finishes, the full app opens. On the **ML Model Training** page, pick one or more models and click **Train**. Training also runs in the API.
3. **Get predictions.** On the **Home** page, click **Sync upcoming events** to fetch the next cards. Pick an event to see its fight predictions; the trained model with the best test F1 score is selected by default.

To keep the data current later, click **Update fight history** on the Home page. It scrapes completed events newer than the stored history.

Settings are read from environment variables or a `.env` file:

| Variable | Default | Used by |
|---|---|---|
| `DATABASE_URL` | `sqlite:///./ufc_analytics.db` | API (and the Streamlit pages that read the database) |
| `API_BASE_URL` | `http://127.0.0.1:8000` | Streamlit, to reach the API |


## Project roadmap

> **Last updated:** 2026-10-02

- [ ] Deploy Beta on Streamlit Server ⚪
- [ ] Add orchestration for automated updates on training and predictions.
- [ ] Betting Performance Tracker - Identify betting arbitrage opportunities based on if in a particular fight, someone has been grossly over/under-estimated
    - [ ] Fetch historical odds from betting websites
    - Update model to compute betting odds.
    - [ ] Scrape betting odds for upcoming event.
    - [ ] Predictor analyzes odds and profitable opportunities based on betting odds.

### Phase 2
- [ ] Expansion for UFC BJJ

### Legend
-  🟡 In Progress  🔴 Blocked  ⚪ Not Started  ✅ Complete
