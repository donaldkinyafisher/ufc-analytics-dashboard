import json
from datetime import UTC, datetime

import pandas as pd
import requests
import shap
from matplotlib import pyplot as plt

import streamlit as st
from src.ml.features import FEATURE_COLUMNS
from src.ml.utils import MODEL_NAMES
from src.utils import (
    READ_ONLY,
    api_get,
    error_detail,
    load_event_predictions,
    load_model,
    load_model_metrics,
    load_models,
    load_preprocessed_data,
    load_shap_values,
    load_status,
    load_training_fights,
    start_job,
)


@st.fragment(run_every=5)
def training_progress(job_id: int) -> None:
    """Poll a training job; when it ends, rerun the page to show its results."""
    try:
        job = api_get(f"/api/v1/models/training-jobs/{job_id}", timeout=5)
    except requests.exceptions.RequestException as error:
        st.warning(f"Lost contact with the API, retrying: {error_detail(error)}")
        return

    if job["status"] not in ("queued", "running"):
        # Training rewrote the model files, the saved split and the metrics.
        load_preprocessed_data.clear()
        load_models.clear()
        load_model.clear()
        load_event_predictions.clear()
        st.session_state["finished_training_job"] = job
        st.rerun()

    models = ", ".join(job["models"])
    tuning = " with hyper-parameter tuning" if job["tune"] else ""
    if job["started_at"] is None:
        st.info(f"Queued: {models}{tuning}.")
    else:
        started = datetime.fromisoformat(job["started_at"]).replace(tzinfo=UTC)
        minutes, seconds = divmod(int((datetime.now(UTC) - started).total_seconds()), 60)
        st.info(f"Training {models}{tuning}… {minutes}m {seconds:02d}s elapsed. "
                "This runs in the API, so you can leave this page.")

#st.title("Train and select model")

training_fights_df = load_training_fights()

#Preview data 
st.subheader("Preview Training Data")
st.caption(
    f"{len(training_fights_df):,} decided fights from the database. Career stats are from "
    "each fighter's earlier ufc fights only; heights and reaches are in inches."
)
st.dataframe(
    training_fights_df[["event_date", "red_name", "blue_name", "winner_side", *FEATURE_COLUMNS]].tail(),
    hide_index=True,
)

#Train model
st.subheader("Train Model")

# Training runs as an API job; the page polls it while it runs.
active_job = load_status()["active_training_job"]
if READ_ONLY:
    st.info("Training runs on the local copy of this app. The metrics and feature importance below "
            "are from the models in this snapshot.")
elif active_job is not None:
    training_progress(active_job["id"])
else:
    selected_model_to_train = st.multiselect("Select model to train", options=['All'] + MODEL_NAMES)
    if selected_model_to_train and 'All' in selected_model_to_train:
        selected_model_to_train = MODEL_NAMES

    #Tune Hyper-paramaters
    tune_hyperparameters = st.checkbox("Tune Hyper-parameters")
    st.warning(" Hyper-parameter tuning is currently not availbale for the Pytorch MLP model. Tuning may take a long time depending on the model and the number of trials.")
    tune = bool(tune_hyperparameters)

    if st.button("Train", type="primary") and selected_model_to_train:
        start_job("/api/v1/models/training-jobs",
                  {"models": selected_model_to_train, "tune": tune}, "training")

# Results of a job that finished while this page was polling it, shown once.
finished_job = st.session_state.pop("finished_training_job", None)

try:
    model_metrics = load_model_metrics()
except FileNotFoundError:
    model_metrics = {}
except (OSError, json.JSONDecodeError) as exc:
    st.warning(f"Could not load model metrics: {exc}")
    model_metrics = {}

if finished_job is not None and finished_job["status"] == "succeeded":
    st.success("Training finished; models and metrics saved.")
    for m in finished_job["models"]:
        st.write(f"Classification report for {m}")
        classifcation_report = pd.DataFrame.from_dict(model_metrics.get(m, {}).get("classification_report", {}))
        st.table(classifcation_report)
elif finished_job is not None:
    st.error(f"Training failed: {finished_job['error']}")

try:
    trained_models = [model["name"] for model in load_models() if model["trained"]]
except requests.exceptions.RequestException as error:
    st.error(f"Could not load models: {error_detail(error)}")
    st.stop()

if not trained_models:
    st.info("No trained models in this snapshot." if READ_ONLY
            else "No trained models yet. Train one above to see its metrics and feature importance.")
    st.stop()

st.subheader("Model Metrics")
#Show metrics in a table format
metrics_df = pd.DataFrame.from_dict(
    {name: metrics for name, metrics in model_metrics.items() if name in trained_models}, orient="index"
)
st.table(metrics_df.drop(columns=["classification_report"], errors="ignore"))

### -----------------

st.subheader("View Feature Importance")
st.caption("SHAP values for the probability that red wins, on a sample of the test set.")
selected_model_name = st.selectbox("Select Model", options=trained_models)
_, _, _, _, feature_names = load_preprocessed_data()
feature_names_simple = [name.split("__")[-1] for name in feature_names]
if selected_model_name:
    trained_at = model_metrics.get(selected_model_name, {}).get("trained_at")
    shap_values, X_shap = load_shap_values(selected_model_name, trained_at)

    shap.summary_plot(shap_values, X_shap, feature_names=feature_names_simple, show=False)
    fig = plt.gcf()
    ax = plt.gca()

    # Transparent background
    fig.patch.set_alpha(0.0)
    ax.patch.set_alpha(0.0)

    # White text for axis labels, title, ticks
    ax.xaxis.label.set_color('white')
    ax.yaxis.label.set_color('white')
    ax.title.set_color('white')
    ax.tick_params(colors='white', which='both')

    # White text for the feature name labels on the y-axis (these are separate text artists)
    for text in ax.get_yticklabels():
        text.set_color('white')
    for text in ax.get_xticklabels():
        text.set_color('white')

    # SHAP summary_plot also creates a colorbar — its ticks/labels need updating too
    # It's usually the last axes added to the figure
    for axis in fig.axes:
        if axis is not ax:  # this is the colorbar axis
            axis.patch.set_alpha(0.0)
            axis.tick_params(colors='white')
            for text in axis.get_yticklabels():
                text.set_color('white')
            if axis.yaxis.label:
                axis.yaxis.label.set_color('white')

    st.pyplot(fig, transparent=True)
    plt.close(fig)
