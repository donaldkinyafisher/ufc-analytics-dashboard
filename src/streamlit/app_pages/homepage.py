from __future__ import annotations
from pathlib import Path

import pandas as pd
import requests

import streamlit as st
from src.utils import (
    api_get,
    api_post,
    ingestion_progress,
    load_event_predictions,
    load_models,
    load_status,
    show_ingestion_result,
    start_job,
)

st.set_page_config(page_title="UFC Analytics", layout="wide")

LOGO_PATH = Path(__file__).resolve().parents[2] / "assets" / "ufc_logo.png"
# The middle column keeps the logo centered; widen it to grow the logo.
_, logo_column, _ = st.columns([1, 1, 1])
with logo_column:
    st.image(str(LOGO_PATH), use_container_width=True, output_format="PNG")
st.title("UFC Data Analytics")

st.markdown(
    """
    Welcome to the UFC Data Analytics app!
    This app provides data analytics and machine learning predictions for UFC fights.
    Use the navigation menu on the left to explore different pages. If this is your first time opening the webpage, start by fetching the latest upcoming events using the :red[**Sync upcoming events**] button below. You can then view the upcoming events and their fight predictions using the trained machine learning models. If you haven't trained any models yet, head over to the :red[**ML Model Training**] page to train one. Enjoy exploring the data and predictions! :sunglasses:
    """
)
st.subheader("Upcoming Events")
# Filled once the events are fetched, so a sync below shows its own time.
last_synced_slot = st.empty()

status = load_status()
active_ingestion = status["active_ingestion_job"]
sync_col, history_col, _ = st.columns(3)
sync_clicked = sync_col.button("Sync upcoming events", use_container_width=True)
update_clicked = history_col.button(
    "Update fight history",
    use_container_width=True,
    disabled=active_ingestion is not None,
    help="Scrape completed events newer than the stored history, for analytics and training.",
)
if status["newest_event_date"]:
    st.caption(
        f"Fight history: {status['events_ingested']:,} events, "
        f"{pd.Timestamp(status['oldest_event_date']):%d %b %Y} to {pd.Timestamp(status['newest_event_date']):%d %b %Y}."
    )

if update_clicked:
    start_job("/api/v1/ingestion/jobs", {"mode": "incremental"}, "the update")
if active_ingestion is not None:
    ingestion_progress(active_ingestion["id"])
# Shown once, on the rerun after a job this session was polling ends.
finished_ingestion = st.session_state.pop("finished_ingestion_job", None)
if finished_ingestion is not None:
    show_ingestion_result(finished_ingestion)

if sync_clicked:
    try:
        sync_result = api_post("/api/v1/events/upcoming/sync", timeout=300)
    except requests.exceptions.RequestException as error:
        st.error(f"Sync failed: {error}")
    else:
        load_event_predictions.clear()  # cards may have changed
        st.success(f"Events synced: {sync_result['fights']} fights on upcoming cards.")
        if sync_result["card_failures"]:
            st.warning(f"Could not load cards for: {', '.join(sync_result['card_failures'])}")

try:
    events_df = pd.DataFrame(api_get("/api/v1/events/upcoming"))
    st.dataframe(events_df.drop(columns=["ufcstats_id"], errors="ignore"), use_container_width=True, hide_index=True)

except requests.exceptions.Timeout:
    events_df = pd.DataFrame()
    st.error("The API took too long to respond.")

except requests.exceptions.RequestException as error:
    events_df = pd.DataFrame()
    st.error(f"API request failed: {error}")

last_synced = (
    pd.to_datetime(events_df["scraped_at"]).max() if "scraped_at" in events_df else pd.NaT
)
last_synced_slot.caption(
    f"Last synced: {last_synced:%d %b %Y, %H:%M} UTC" if pd.notna(last_synced) else "Never synced"
)


def prediction_table(fights: list[dict]) -> pd.DataFrame:
    def fighter(fight: dict, corner: str) -> str:
        name = fight[f"{corner}_fighter"]["name"]
        return f"{name} (debut)" if fight[f"{corner}_ufc_fights"] == 0 else name

    def percent(probability: float | None) -> float | None:
        return None if probability is None else probability * 100

    # Fights with a debutant have no prediction, so those cells stay blank.
    return pd.DataFrame(
        {
            "Bout": fight["bout_order"],
            "Weight class": fight["weight_class"],
            "Bout type": fight["bout_type"],
            "Red fighter": fighter(fight, "red"),
            "Blue fighter": fighter(fight, "blue"),
            "Predicted winner": (fight["predicted_winner"] or {}).get("name"),
            "Red win %": percent(fight["red_win_probability"]),
            "Blue win %": percent(fight["blue_win_probability"]),
        }
        for fight in fights
    )


st.subheader("Predict fight winners")
try:
    trained_models = [model for model in load_models() if model["trained"]]
except requests.exceptions.RequestException as error:
    trained_models = []
    st.error(f"Could not load models: {error}")

if events_df.empty:
    st.info("No upcoming events stored. Run Sync upcoming events.")
elif not trained_models:
    st.info("No trained models yet. Train one to get fight predictions.")
    st.page_link("app_pages/train.py", label="Go to ML Model Training", icon="🤖")
else:
    event_names = {int(event_id): name for event_id, name in zip(events_df["id"], events_df["name"])}
    model_names = [model["name"] for model in trained_models]
    best_model = next((model["name"] for model in trained_models if model["is_best"]), model_names[0])

    select_cols = st.columns(2)
    # Events arrive soonest first, so the default is the next card.
    event_id = select_cols[0].selectbox("Event", options=list(event_names), format_func=event_names.get)
    model_name = select_cols[1].selectbox(
        "Model",
        options=model_names,
        index=model_names.index(best_model),
        format_func=lambda name: f"{name} (best)" if name == best_model else name,
    )

    try:
        fights = load_event_predictions(event_id, model_name)["fights"]
    except requests.exceptions.RequestException as error:
        fights = None
        st.error(f"Prediction failed: {error}")

    if fights == []:
        st.info("No fights stored for this card yet. Run Sync upcoming events.")
    elif fights:
        percent = st.column_config.ProgressColumn(format="%.0f%%", min_value=0, max_value=100)
        st.dataframe(
            prediction_table(fights),
            use_container_width=True,
            hide_index=True,
            column_config={"Red win %": percent, "Blue win %": percent},
        )
        if any(fight["red_ufc_fights"] == 0 or fight["blue_ufc_fights"] == 0 for fight in fights):
            st.caption(
                "(debut): the fighter has no UFC fights in the database, so no prediction is made for that fight."
            )
