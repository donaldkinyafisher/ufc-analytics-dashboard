"""Extract: read the normalised tables from the database into DataFrames."""

import pandas as pd
from sqlalchemy import Engine, bindparam, text

from src.api.v1.database import engine as default_engine

RAW_QUERIES = {
    "events": """
        SELECT id, name, event_date, location
        FROM events
        WHERE status IN :statuses
    """,
    "fights": """
        SELECT id, event_id, red_fighter_id, blue_fighter_id, bout_order,
               weight_class, is_title_bout, winner_side, method, method_details,
               round, time, time_format, referee
        FROM fights
    """,
    "fighters": """
        SELECT id, name, height_in, weight_lbs, reach_in, stance, dob
        FROM fighters
    """,
    "fight_statistics": "SELECT * FROM fight_statistics",
}

DATE_COLUMNS = {"events": ["event_date"], "fighters": ["dob"]}


def load_raw_tables(
    engine: Engine = default_engine, include_upcoming: bool = False
) -> dict[str, pd.DataFrame]:
    """One SELECT per table, with date columns parsed to datetime64.

    Only completed events are read unless include_upcoming is set; upcoming
    fights then come through with no winner and no statistics.
    """
    params = {"statuses": ["completed", "upcoming"] if include_upcoming else ["completed"]}
    statements = {name: text(query) for name, query in RAW_QUERIES.items()}
    statements["events"] = statements["events"].bindparams(bindparam("statuses", expanding=True))
    with engine.connect() as conn:
        return {
            name: pd.read_sql_query(
                statement,
                conn,
                params=params if name == "events" else None,
                parse_dates=DATE_COLUMNS.get(name),
            )
            for name, statement in statements.items()
        }
