"""
Analytics ETL: read the scraped SQLite tables and reshape them into
analysis-ready DataFrames ("marts") for the Streamlit EDA dashboard.

    extract.load_raw_tables(engine) -> raw tables
    transform.build_fights_mart(raw)  -> one row per fight

Functions here are pure pandas (no Streamlit) so they can be unit tested
and reused from notebooks; caching lives in the Streamlit layer.
"""
