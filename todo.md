## Fighter data Architecture
- [x] Training.csv is built from historically logged UFC fights. Each row represents a fight.
- [x] Set-up fighter database with stats associated to each fighter. This table collects the data points from training.csv to build a fighter profile. Each row represents a fighter.
- [x] Upcoming fights are then fetched from UFC and matched with the fighter database to build a fight profile. Each row represents a fight.
- [x] The fight profile is then used to predict the outcome of the fight using a machine learning model.
- [x] Need to build a web scraper to fetch upcoming fights from UFC and match them with the fighter database. This will allow us to build a fight profile for each upcoming fight.

## Next: Front-end (Streamlit)
[] Update the Streamlit front-end to use the new API endpoints instead of the sample CSVs: `/api/v1/events`, `/api/v1/events/{id}`, `/api/v1/fights/{id}`, `/api/v1/fighters?name=`, `/api/v1/fighters/{id}`, `/api/v1/fighters/{id}/fights`.
[] Add an ingestion control (start backfill/incremental job via `POST /api/v1/ingestion/jobs`, poll `GET /api/v1/ingestion/jobs/{id}` for progress).
[] Build visualisations from the stored data (fighter profiles, fight stats, event cards).
[] Before relying on the data: fill the 2015–2026 gap in the local DB with an incremental job (~30–40 min + fighter profiles).

## Future

[] Per-round fight statistics: fight-details pages include round-by-round Totals and Significant Strikes tables (the `js-fight-table` tables, currently skipped by `historical_scraper._stats_tables`). Store them in a `FightRoundStatistic` table (fight_id, fighter_id, corner, round, same count columns as `FightStatistic`).
