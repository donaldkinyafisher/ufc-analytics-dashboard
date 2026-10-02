import pandas as pd

import streamlit as st
from src.analytics import charts
from src.utils import load_fighter_careers, load_fights_mart


def chart(fig, data: pd.DataFrame, key: str, view_table: bool = False) -> None:
    """Plot a figure with its aggregated table view available as an option."""
    st.plotly_chart(fig, use_container_width=True, key=key, config={"displayModeBar": False})
    if view_table:
        with st.expander("Table view"):
            st.dataframe(data, use_container_width=True, hide_index=True)


mart = load_fights_mart()

title_col, refresh_col = st.columns([5, 1], vertical_alignment="bottom")
title_col.title("Historical Fight Analysis")
if refresh_col.button("Refresh data", help="Reload from the database after a scrape"):
    load_fights_mart.clear()
    load_fighter_careers.clear()
    st.rerun()

#Textual description of the dashboard.

st.markdown(
    """ This dashboard provides an overview of historical fight data, allowing users to explore various metrics and trends in the sport. Use the filters above to narrow down the data by year, division, and weight class. Navigate through different sections to analyze fight outcomes, physical attributes, striking and grappling statistics, and individual fighter careers. """
)

# Filters: one row above all charts.
year_min, year_max = int(mart["year"].min()), int(mart["year"].max())
year_col, gender_col, weight_col = st.columns([2, 1, 3])
years = year_col.slider("Years", year_min, year_max, (year_min, year_max))
gender = gender_col.selectbox("Division", ["All", "Men", "Women"])
weight_options = [c for c in mart["weight_class"].cat.categories if c in set(mart["weight_class"].astype(str))]
weight_classes = weight_col.multiselect("Weight classes", weight_options, placeholder="All weight classes")

filtered = mart[mart["year"].between(*years)]
if gender != "All":
    filtered = filtered[filtered["gender"] == gender]
if weight_classes:
    filtered = filtered[filtered["weight_class"].isin(weight_classes)]

if filtered.empty:
    st.info("No fights match these filters.")
    st.stop()

fighter_count = pd.concat([filtered["red_fighter_id"], filtered["blue_fighter_id"]]).nunique()
metric_cols = st.columns(4)
metric_cols[0].metric("Fights", f"{len(filtered):,}")
metric_cols[1].metric("Events", f"{filtered['event_id'].nunique():,}")
metric_cols[2].metric("Fighters", f"{fighter_count:,}")
metric_cols[3].metric("Finish rate", f"{filtered['is_finish'].mean():.0%}")

# A server-side section switch rather than st.tabs: tabs reset to the first one
# whenever a widget inside them (e.g. the metric radios) triggers a rerun.
SECTIONS = ["Data Overview", "Fight Outcomes", "Physicals", "Striking & Grappling", "Fighter explorer"]
section = st.segmented_control("Section", SECTIONS, default=SECTIONS[0], key="section",
                               label_visibility="collapsed") or SECTIONS[0]

if section == "Data Overview":
    st.subheader("Fights per year")
    chart(charts.fights_per_year(filtered), charts.fights_per_year_data(filtered), "fights_per_year")

    st.subheader("Data completeness by decade")
    st.caption("Share of fights where the field is recorded (red corner shown for fighter fields).")
    completeness = charts.completeness_data(filtered)
    chart(charts.completeness_heatmap(filtered), completeness.reset_index(names="field"), "completeness")

    st.subheader("Corner bias")
    st.caption(
        "ufcstats lists the winner first, so the stored red corner is usually the winner. "
        "Red-minus-blue differences lean towards the winner; randomise sides before training."
    )
    chart(charts.red_win_share(filtered), charts.red_win_share_data(filtered), "red_win_share")

if section == "Fight Outcomes":
    st.subheader("How fights end, by year")
    st.caption("M-DEC, DQ, overturned and other rare results are grouped as Other.")
    chart(charts.method_share_by_year(filtered), charts.method_share_data(filtered, "year"), "method_by_year")

    st.subheader("How fights end, by weight class")
    chart(
        charts.method_share_by_weight_class(filtered),
        charts.method_share_data(filtered, "weight_class"),
        "method_by_weight_class",
    )

    round_col, duration_col = st.columns(2)
    with round_col:
        st.subheader("Finishes by round")
        chart(charts.finish_round(filtered), charts.finish_round_data(filtered), "finish_round")
    with duration_col:
        st.subheader("Fight length by weight class")
        chart(
            charts.fight_duration_by_weight_class(filtered),
            charts.fight_duration_data(filtered),
            "fight_duration",
        )

if section == "Physicals":
    metric = st.radio("Attribute", list(charts.PHYSICAL_METRICS), horizontal=True, key="physical_metric")

    st.subheader(f"{metric} by weight class")
    st.caption("One entry per fighter per fight, so active fighters count more than once.")
    chart(
        charts.physical_by_weight_class(filtered, metric),
        charts.physical_by_weight_class_data(filtered, metric),
        "physical_by_weight_class",
    )

    st.subheader("Does the physical advantage win?")
    st.caption(
        "Win rate of the taller, longer-reach or younger fighter, by the size of the gap. "
        "Compares the advantaged fighter with the winner, so the red/blue listing bias does not apply."
    )
    chart(
        charts.advantage_win_rate(filtered, metric),
        charts.advantage_win_rate_data(filtered, metric),
        "advantage_win_rate",
    )

if section == "Striking & Grappling":
    st.caption(f"Rates exclude fights shorter than {charts.MIN_RATE_SECONDS} seconds.")
    metric = st.radio("Metric", list(charts.STRIKING_METRICS), horizontal=True, key="striking_metric")
    st.subheader(f"{metric} by weight class")
    chart(
        charts.striking_by_weight_class(filtered, metric),
        charts.striking_by_weight_class_data(filtered, metric),
        "striking_by_weight_class",
    )

    kind = st.radio("Landed sig. strikes by", list(charts.STRIKE_MIX), horizontal=True, key="strike_mix_kind")
    st.subheader(f"Where strikes land: {kind.lower()} mix")
    chart(charts.strike_mix(filtered, kind), charts.strike_mix_data(filtered, kind), "strike_mix")

    leader_col, control_col = st.columns(2)
    with leader_col:
        st.subheader("Which stats go with winning?")
        st.caption("Win rate of the fighter who led each stat, ignoring ties.")
        chart(
            charts.stat_leader_win_rate(filtered),
            charts.stat_leader_win_rate_data(filtered),
            "stat_leader_win_rate",
        )
    with control_col:
        st.subheader("Takedowns and control time")
        st.caption(
            "Median control time in minutes compared against number of successful takedowns. Control time is the sum of time spent in top position and in the clinch."
        )
        chart(
            charts.control_by_takedowns(filtered),
            charts.control_by_takedowns_data(filtered),
            "control_by_takedowns",
        )

if section == "Fighter explorer":
    fighter_fights, careers = load_fighter_careers()
    st.caption("Shows the fighter's whole UFC career; the filters above do not apply here.")

    fight_counts = fighter_fights.groupby("fighter_id").agg(
        name=("fighter_name", "first"), fights=("fight_id", "size")
    )
    options = fight_counts.sort_values("name").index.tolist()
    fighter_id = st.selectbox(
        "Fighter", options, index=options.index(fight_counts["fights"].idxmax()),
        format_func=lambda fid: f"{fight_counts.at[fid, 'name']} ({fight_counts.at[fid, 'fights']} fights)",
        key="explorer_fighter",
    )

    record = charts.fighter_record(fighter_fights, fighter_id)
    history = charts.fighter_history(fighter_fights, fighter_id)
    metric_cols = st.columns(4)
    metric_cols[0].metric("Record (W-L-D)", record["record"])
    metric_cols[1].metric("Main weight class", record["main_weight_class"])
    metric_cols[2].metric(
        "Wins by finish", "-" if pd.isna(record["finish_rate"]) else f"{record['finish_rate']:.0%}"
    )
    metric_cols[3].metric(
        "UFC career", f"{record['first_fight']:%Y} - {record['last_fight']:%Y}"
    )

    st.subheader("Career timeline")
    st.caption("Running record: up one for each win, down one for each loss.")
    chart(charts.career_timeline(history), charts.fighter_history_table(history), "career_timeline")

    st.subheader("Compared with their weight class")
    if fighter_id in careers.index:
        radar = charts.radar_data(careers, fighter_id)
        st.caption(
            f"Percentile among {record['main_weight_class']} fighters with at least "
            f"{charts.MIN_PEER_FIGHTS} fights (50 = class median). Career totals, pooled."
        )
        chart(charts.career_radar(radar, record["name"], record["main_weight_class"]),
              radar.round(3), "career_radar")
    else:
        st.info("No fight statistics recorded for this fighter.")

    st.subheader("Significant strikes per fight")
    chart(charts.fight_strikes(history), charts.fighter_history_table(history), "fight_strikes")
