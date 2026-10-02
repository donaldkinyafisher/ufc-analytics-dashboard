"""First-run setup: build the database with a historical ingestion job.

main.py shows only this page while GET /api/v1/status reports needs_setup.
The job runs in the API, so closing the browser does not stop it; reopening
the page picks up the running job and keeps polling it. When the job ends
the whole app reruns, and the gate opens once a job has succeeded; the
homepage then shows the job's result.
"""

from datetime import date

import streamlit as st
from src.utils import ingestion_progress, load_status, show_ingestion_errors, start_job

# Rough durations, from a full backfill of ~470 events taking ~30 minutes.
SCOPES = {
    "Last 5 years": (5, "about 15 minutes"),
    "Last 10 years": (10, "about 30 minutes"),
    "Full history (since 1993)": (None, "about an hour"),
}


def years_ago(years: int) -> date:
    today = date.today()
    return date(today.year - years, today.month, min(today.day, 28))


st.title("Set up UFC Analytics")
st.markdown(
    """
    There's no fight data yet. This scrapes completed events, fights, fight statistics and
    fighter profiles from [ufcstats.com](http://ufcstats.com) into your local database.
    The scrape runs in the API, so you can close this tab and come back; keep the API running.

    You can scrape further back later. Once it finishes, train a model on the
    :red[**ML Model Training**] page to get fight predictions.
    """
)

status = load_status()
# The outcome of a job that just ended is read from status below instead.
st.session_state.pop("finished_ingestion_job", None)
active = status["active_ingestion_job"]
last = status["last_ingestion_job"]

if active is not None:
    st.subheader("Building the database")
    ingestion_progress(active["id"])
else:
    if last is not None and last["status"] == "failed":
        st.error("The last scrape failed. Data stored before the failure is kept, so a retry continues from there.")
        show_ingestion_errors(last)
    elif last is not None and last["status"] == "succeeded":
        st.warning("The last scrape finished but found no completed events in its date range. Try a wider range.")

    scope = st.radio(
        "How far back should the first scrape go?",
        options=list(SCOPES),
        format_func=lambda label: f"{label} ({SCOPES[label][1]})",
    )
    years, _ = SCOPES[scope]
    label = "Retry" if last is not None and last["status"] == "failed" else "Build database"
    if st.button(label, type="primary"):
        payload = {"mode": "backfill"}
        if years is not None:
            payload["since"] = years_ago(years).isoformat()
        start_job("/api/v1/ingestion/jobs", payload, "the scrape")
