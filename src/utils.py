import os

import pandas as pd
import requests

import streamlit as st
from src.ml import utils as ml_utils
from src.ml.predictor import load_trained_model

API_BASE_URL = os.getenv("API_BASE_URL", "http://127.0.0.1:8000")


@st.cache_data(ttl=3600, show_spinner="Loading fight data...")
def load_fights_mart() -> pd.DataFrame:
    """One row per completed fight, built from the scraped database."""
    from src.analytics.extract import load_raw_tables
    from src.analytics.transform import build_fights_mart

    return build_fights_mart(load_raw_tables())

@st.cache_data(ttl=3600, show_spinner="Loading fighter careers...")
def load_fighter_careers() -> tuple[pd.DataFrame, pd.DataFrame]:
    """(one row per fighter per fight, career rates per fighter) for the fighter explorer."""
    from src.analytics.charts import career_stats
    from src.analytics.transform import build_fighter_fights

    fighter_fights = build_fighter_fights(load_fights_mart())
    return fighter_fights, career_stats(fighter_fights)

@st.cache_data
def load_preprocessed_data():
    """(X_train, y_train, X_test, y_test, feature_names) from the saved split."""
    return ml_utils.load_preprocessed_data()

@st.cache_resource
def load_model(model_name: str):
    return load_trained_model(model_name)

@st.cache_data(ttl=3600, show_spinner="Building training set...")
def load_training_fights() -> pd.DataFrame:
    """Decided fights with their model features, as built from the database."""
    from src.analytics.extract import load_raw_tables
    from src.ml.dataset import fight_features

    features = fight_features(load_raw_tables())
    return features[features["winner_side"].isin(["red", "blue"])]

@st.cache_data(persist="disk", show_spinner="Computing SHAP values...")
def load_shap_values(model_name: str, trained_at: str | None):
    """SHAP values on the saved test split; trained_at refreshes the cache after retraining."""
    from src.ml.explain import shap_values

    _, _, X_test, _, _ = ml_utils.load_preprocessed_data()
    return shap_values(load_trained_model(model_name), X_test)

def api_get(path: str, params: dict | None = None, timeout: float = 30):
    response = requests.get(f"{API_BASE_URL}{path}", params=params, timeout=timeout)
    response.raise_for_status()
    return response.json()

def api_post(path: str, params: dict | None = None, json: dict | None = None, timeout: float = 120):
    response = requests.post(f"{API_BASE_URL}{path}", params=params, json=json, timeout=timeout)
    response.raise_for_status()
    return response.json()

def load_status() -> dict:
    """Stored data, running jobs and needs_setup, from GET /api/v1/status.

    Not cached: the setup gate must see a finished ingestion job straight away.
    """
    return api_get("/api/v1/status", timeout=5)

@st.cache_data(ttl=60)
def load_models() -> list[dict]:
    """Supported models with their test metrics, from GET /api/v1/models."""
    return api_get("/api/v1/models")

@st.cache_data(ttl=600, show_spinner="Predicting fights...")
def load_event_predictions(event_id: int, model_name: str) -> dict:
    """Predictions for one event's card; cleared after a sync changes the cards."""
    return api_get(f"/api/v1/predictions/events/{event_id}", params={"model_name": model_name})

def load_model_metrics() -> dict:
    return ml_utils.load_model_metrics()

# --- API jobs ----------------------------------------------------------------

def error_detail(error: requests.exceptions.RequestException) -> str:
    """The API's error message when it sent one, else the request error."""
    try:
        return error.response.json()["detail"]
    except (AttributeError, ValueError, KeyError, TypeError):
        return str(error)

def start_job(path: str, payload: dict, action: str) -> None:
    """POST a job and rerun so the page shows it.

    A 409 means another job is already running; the rerun shows that one.
    """
    try:
        api_post(path, json=payload, timeout=10)
    except requests.exceptions.RequestException as error:
        if error.response is None or error.response.status_code != 409:
            st.error(f"Could not start {action}: {error_detail(error)}")
            st.stop()
    st.rerun()

@st.fragment(run_every=5)
def ingestion_progress(job_id: int) -> None:
    """Poll an ingestion job; when it ends, rerun the app with the result in session state."""
    try:
        job = api_get(f"/api/v1/ingestion/jobs/{job_id}", timeout=5)
    except requests.exceptions.RequestException as error:
        st.warning(f"Lost contact with the API, retrying: {error_detail(error)}")
        return

    if job["status"] not in ("queued", "running"):
        # The stored fights changed; the whole app reruns so the setup gate re-checks.
        load_fights_mart.clear()
        load_fighter_careers.clear()
        load_training_fights.clear()
        load_event_predictions.clear()
        st.session_state["finished_ingestion_job"] = job
        st.rerun(scope="app")

    done, total = job["events_done"], job["events_total"]
    if total == 0:
        st.progress(0.0, text="Listing completed events on ufcstats…")
    elif done < total:
        st.progress(done / total, text=f"Events {done} of {total} · {job['fights_upserted']:,} fights stored")
    else:
        st.progress(1.0, text=f"Fetching fighter profiles… {job['fighters_upserted']:,} updated")
    if job["errors"]:
        st.caption(f"{job['errors']} page(s) failed so far. They are retried on the next run.")

def show_ingestion_errors(job: dict) -> None:
    with st.expander(f"{len(job['error_log'])} error(s)"):
        st.dataframe(job["error_log"], hide_index=True, use_container_width=True)
    if any("playwright install" in entry["error"] for entry in job["error_log"]):
        st.markdown("The scraper's browser is missing. Install it, then retry:")
        st.code("uv run playwright install chromium", language="bash")

def show_ingestion_result(job: dict) -> None:
    """Outcome of an ingestion job that just finished."""
    if job["status"] == "failed":
        st.error("The scrape failed. Data stored before the failure is kept, so a retry continues from there.")
    elif job["events_total"] == 0:
        st.success("Fight history is already up to date.")
    else:
        # events_done counts attempts; an event whose page failed was rolled back.
        failed = sum("/event-details/" in (entry["url"] or "") for entry in job["error_log"])
        message = (f"Scrape finished: {job['events_done'] - failed:,} new event(s) and "
                   f"{job['fights_upserted']:,} fights stored.")
        if failed:
            message += f" {failed} event(s) could not be stored; see the errors below."
        (st.warning if job["errors"] else st.success)(message)
    if job["error_log"]:
        show_ingestion_errors(job)
