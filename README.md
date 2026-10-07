# UFC Data Analytics Dashboard

Streamlit project for collecting UFC fight data, displaying analytics, and storing feature-ready records for prediction models.

## First Deployment is now live

The service is now hosted on Render. Click on the link below to view a read-only set-up 

 

```bash
https://ufc-analytics-dashboard.onrender.com/
``` 

## Running locally

Running the model locally allows you to scrape fighter data, and train your ML models. Currently this is prohibited on the webpage due to computational constraints.

The environment is handled through the [uv package manager](https://docs.astral.sh/uv/). Package dependencies can be completely handled through `uv sync`. The script runs `uv sync`, installs Chromium for Playwright (the browser used to scrape ufcstats.com) and creates `.env` from `.env.example` if it doesn't exist.

Running `./scripts/setup.sh` takes care of the package and environment creation. 

```bash
git clone <repo-name>
cd <repo-name>
./scripts/setup.sh
```

Once Setup, run the API and the Streamlit app in separate terminal windows. Start the API first; the app reads everything through it.

```bash
uv run fastapi dev --entrypoint src.api.v1.main:app
```

```bash
uv run streamlit run src/streamlit/main.py
```

## First run

The repo includes a snapshot of the fight database (`ufc_analytics.db`) and the trained models (`src/ml/artifacts/`). The hosted read-only copy is built from the same snapshot. A fresh clone therefore skips the **Setup** page and opens the full app with fight history and predictions already available. To bring the snapshot up to date:

1. **Update the data.** On the **Home** page, click **Sync upcoming events** to fetch the next cards, then **Update fight history** to scrape completed events newer than the snapshot. Both run in the API, so keep it running.
2. **Retrain (optional).** On the **ML Model Training** page, pick one or more models and click **Train** to retrain them on the updated history. Training also runs in the API.
3. **Get predictions.** On the **Home** page, pick an event to see its fight predictions. The trained model with the best test F1 score is selected by default.

The snapshot files are tracked in git, so updating the data or retraining changes them in your working copy.

### Building the database from scratch

To scrape your own database instead of using the snapshot, set `DATABASE_URL` to a new file, for example `sqlite:///./ufc_local.db`. While the database is empty, the app shows only the **Setup** page. Choose how far back to scrape (last 5 years, last 10 years or full history) and click **Build database**. A 5-year scrape takes about 15 minutes; full history takes about an hour. The scrape runs in the API, so you can close the browser tab, but keep the API running. If it fails or the API stops, click **Retry**: events already stored are kept, so it continues where it stopped. The trained models in `src/ml/artifacts/` are still loaded, so retrain them on the new data.

Settings are read from environment variables or a `.env` file:

| Variable | Default | Used by |
|---|---|---|
| `DATABASE_URL` | `sqlite:///./ufc_analytics.db` | API (and the Streamlit pages that read the database) |
| `API_BASE_URL` | `http://127.0.0.1:8000` | Streamlit, to reach the API |
| `READ_ONLY` | unset | Streamlit. When set to `true`, the scrape, sync and training controls are hidden. The hosted copy sets this; leave it unset locally. |


## Project roadmap

> **Last updated:** 2026-10-07

- [ ] Expand ML model training to provide deeper insights into feature importance.
- [ ] Add orchestration for automated updates on training and predictions.
- [ ] Betting Performance Tracker - Identify betting arbitrage opportunities based on if in a particular fight, someone has been grossly over/under-estimated.

### Phase 2
- [ ] Expansion for UFC BJJ
