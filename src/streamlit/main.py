import sys
from pathlib import Path

import requests

import streamlit as st

# Make the repository root importable when Streamlit executes page modules.
project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
	sys.path.insert(0, str(project_root))

from src.utils import API_BASE_URL, READ_ONLY, load_status  # noqa: E402  (needs the path above)

# 1. Ask the API what is stored; every page depends on it.
try:
	status = load_status()
except requests.exceptions.RequestException as error:
	st.error(f"Could not reach the API at {API_BASE_URL}.")
	st.caption(str(error))
	st.markdown("Start it in a separate terminal, then refresh this page:")
	st.code("uv run fastapi dev --entrypoint src.api.v1.main:app", language="bash")
	st.stop()

# 2. Define the pages pointing to your file paths
home_page = st.Page("app_pages/homepage.py", title="Home", icon="🏠", default=True)
data_page = st.Page("app_pages/analytics_dashboard.py", title="Analytics Dashboard", icon="📊")
ml_page = st.Page("app_pages/train.py", title="ML Model Training", icon="🤖")
setup_page = st.Page("app_pages/setup.py", title="Setup", icon="🛠️", default=True)

# 3. Until the first ingestion finishes the database is empty, so only Setup is shown.
#    The read-only copy can't scrape, so an empty snapshot is an error instead.
if status["needs_setup"] and READ_ONLY:
	st.error("This read-only copy has no fight data. Rebuild it from a populated database.")
	st.stop()
pg = st.navigation([setup_page] if status["needs_setup"] else [home_page, data_page, ml_page])

# 4. Run the selected page
pg.run()
