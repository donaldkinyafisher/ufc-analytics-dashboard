import json
import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import requests

import streamlit as st

API_BASE_URL = os.getenv("API_BASE_URL", "http://127.0.0.1:8000")
ARTIFACT_DIR = Path(__file__).resolve().parent /"ml"/ "artifacts"
DATA_FILES_DIR = Path(__file__).resolve().parent / "data"
MODEL_METRICS_PATH = ARTIFACT_DIR / "metrics" / "model_metrics.json"
MODELS_DIR = ARTIFACT_DIR / "models"
DATA_DIR = ARTIFACT_DIR / "data"
DATASET_PATH = DATA_DIR / "ufc_split_data.npz"


@st.cache_data(ttl=300)
def get_upcoming_events(refresh: bool = False) -> dict:
    response = requests.put(
        f"{API_BASE_URL}/api/v1/events/upcoming",
        params={"refresh": refresh},
        timeout=120,
    )
    response.raise_for_status()
    return response.json()


@st.cache_data
def load_data():
    try:
        df = pd.read_csv(DATA_FILES_DIR / "historical_fights.csv")
    except FileNotFoundError:
        st.error("Historical fights data not found. Run the import_ufcdata.py script to fetch the data.")
        st.stop()
    return df

@st.cache_data
def load_upcoming_events() -> pd.DataFrame:
    return pd.read_csv(DATA_FILES_DIR / "upcoming_events.csv", skipinitialspace=True)

@st.cache_data
def load_scheduled_fights() -> pd.DataFrame:
    try:
        return pd.read_csv(DATA_FILES_DIR / "scheduled_fights.csv", skipinitialspace=True)
    except FileNotFoundError:
        return pd.DataFrame()

@st.cache_data
def load_preprocessed_data() -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, list]:
    """

    Args:

    Returns:
        _type_: _description_
    """

    # 1. Open the archive wrapper
    data_archive = np.load(DATASET_PATH)

    # 2. Unpack them directly into variables
    X_train = data_archive['X_train']
    X_test = data_archive['X_test']
    y_train = data_archive['y_train']
    y_test = data_archive['y_test']
    feature_names = data_archive['feature_names']

    # 3. Always close the archive file when finished unpacking
    data_archive.close()

    return X_train, y_train, X_test, y_test, feature_names

@st.cache_resource
def load_model(model_name: str):
    return joblib.load(MODELS_DIR / f"{model_name}.joblib")

def api_get(path: str):
    response = requests.get(f"{API_BASE_URL}{path}", timeout=30)
    response.raise_for_status()
    return response.json()

def api_post(path: str, params: dict | None = None):
    response = requests.post(f"{API_BASE_URL}{path}", params=params, timeout=120)
    response.raise_for_status()
    return response.json()

def load_model_metrics() -> dict:
    if MODEL_METRICS_PATH.exists():
        with MODEL_METRICS_PATH.open("r", encoding="utf-8") as metrics_file:
            return json.load(metrics_file)
    else:
        raise FileNotFoundError(f"Model metrics file not found at {MODEL_METRICS_PATH}.")