import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pytest

from src.analytics import charts
from src.analytics.transform import build_fighter_fights, build_fights_mart


@pytest.fixture
def mart(raw):
    return build_fights_mart(raw)


def test_fights_per_year_data(mart):
    data = charts.fights_per_year_data(mart)
    assert data.to_dict("records") == [
        {"year": 1994, "fights": 1, "events": 1},
        {"year": 2024, "fights": 3, "events": 1},
    ]


def test_red_win_share_ignores_draws(mart):
    data = charts.red_win_share_data(mart).set_index("year")
    assert data.loc[2024, "fights"] == 2  # title (blue win) + split (red win); draw excluded
    assert data.loc[2024, "red_win_share"] == 0.5


def test_method_share_sums_to_one_and_folds_rare_methods(mart):
    data = charts.method_share_data(mart, "year")
    assert data.groupby("year")["share"].sum().round(6).eq(1).all()
    assert set(data["method"]) == {"SUB", "KO/TKO", "S-DEC", "Other"}  # M-DEC folds into Other


def test_completeness_data(mart):
    data = charts.completeness_data(mart)
    assert list(data.columns) == ["1990s", "2020s"]
    assert data.loc["Height", "1990s"] == 1.0
    assert data.loc["Sig. strikes", "1990s"] == 0.0
    assert data.loc["Sig. strikes", "2020s"] == pytest.approx(1 / 3)


def test_finish_round_data(mart):
    data = charts.finish_round_data(mart)
    assert data.to_dict("records") == [
        {"round": 1, "method": "SUB", "fights": 1},
        {"round": 2, "method": "KO/TKO", "fights": 1},
    ]


@pytest.mark.parametrize(
    "figure",
    [
        charts.fights_per_year,
        charts.completeness_heatmap,
        charts.red_win_share,
        charts.method_share_by_year,
        charts.method_share_by_weight_class,
        charts.finish_round,
        charts.fight_duration_by_weight_class,
        charts.stat_leader_win_rate,
        charts.control_by_takedowns,
        *[lambda m, k=k: charts.physical_by_weight_class(m, k) for k in charts.PHYSICAL_METRICS],
        *[lambda m, k=k: charts.advantage_win_rate(m, k) for k in charts.PHYSICAL_METRICS],
        *[lambda m, k=k: charts.striking_by_weight_class(m, k) for k in charts.STRIKING_METRICS],
        *[lambda m, k=k: charts.strike_mix(m, k) for k in charts.STRIKE_MIX],
    ],
)
def test_figures_build(mart, figure):
    fig = figure(mart)
    assert isinstance(fig, go.Figure) and fig.data


def test_method_colors_follow_the_method(mart):
    """Filtering must not repaint the remaining series."""
    full = {t.name: t.marker.color for t in charts.method_share_by_year(mart).data}
    subset = {t.name: t.marker.color for t in charts.method_share_by_year(mart[mart["year"] == 2024]).data}
    assert all(full[name] == color for name, color in subset.items())
    assert full["KO/TKO"] == charts.SERIES[0]


def test_fighter_appearances_stacks_both_corners(mart):
    fighters = charts.fighter_appearances(mart, ["name", "height_in"])
    assert len(fighters) == 2 * len(mart)
    title = fighters[fighters["fight_id"] == mart.loc[mart["is_title_bout"], "fight_id"].item()]
    assert set(title["name"]) == {"Alpha", "Bravo"}


def test_advantage_win_rate_is_corner_neutral(mart):
    # Title: Alpha (red, 2 in taller, 5 yrs older) loses to Bravo (blue). Other fights lack a
    # height or age gap (Charlie has neither) or have no winner.
    height = charts.advantage_win_rate_data(mart, "Height (in)")
    assert height.to_dict("records") == [{"gap": "1-2 in", "win_rate": 0.0, "fights": 1}]
    age = charts.advantage_win_rate_data(mart, "Age at fight")
    assert age.to_dict("records") == [{"gap": "4-6 yrs", "win_rate": 1.0, "fights": 1}]


def test_rate_charts_skip_short_fights(mart):
    # UFC 1 lasted 26 s; the title fight (390 s) is the only one with stats left.
    data = charts.striking_by_weight_class_data(mart, "Sig. strikes landed per minute")
    assert data["count"].sum() == 2
    assert list(data["weight_class"].astype(str)) == ["Women's Flyweight"]


def test_strike_mix_shares(mart):
    data = charts.strike_mix_data(mart, "Target").set_index(["weight_class", "part"])
    # Title fight pooled: head 30, body 6, leg 3 of 39 landed.
    assert data.loc[("Women's Flyweight", "Head"), "share"] == pytest.approx(30 / 39)
    assert data.groupby(level="weight_class", observed=True)["share"].sum().dropna().round(6).eq(1).all()


def test_stat_leader_win_rate(mart):
    data = charts.stat_leader_win_rate_data(mart).set_index("stat")
    # Bravo (blue) led knockdowns and sig. strikes and won; Alpha led control and takedowns and lost.
    assert data.loc["Knockdowns", "win_rate"] == 1.0
    assert data.loc["Sig. strikes landed", "win_rate"] == 1.0
    assert data.loc["Control time", "win_rate"] == 0.0
    assert data.loc["Takedowns landed", "fights"] == 1
    assert data.loc["Submission attempts", "fights"] == 0


def test_control_by_takedowns(mart):
    data = charts.control_by_takedowns_data(mart)
    assert data.to_dict("records") == [
        {"takedowns": "0", "median_control_minutes": 0.0, "fighters": 1},
        {"takedowns": "1", "median_control_minutes": 0.65, "fighters": 1},
    ]


@pytest.fixture
def fighter_fights(mart):
    return build_fighter_fights(mart)


def _fighter_id(fighter_fights, name):
    return fighter_fights.loc[fighter_fights["fighter_name"] == name, "fighter_id"].iloc[0]


def test_fighter_record_and_history(fighter_fights):
    alpha = _fighter_id(fighter_fights, "Alpha")
    record = charts.fighter_record(fighter_fights, alpha)
    assert (record["record"], record["fights"], record["finish_rate"]) == ("1-1-1", 3, 1.0)
    history = charts.fighter_history(fighter_fights, alpha)
    assert list(history["result"]) == ["W", "L", "D"]
    assert list(history["net_record"]) == [1, 0, 0]
    table = charts.fighter_history_table(history)
    assert list(table["Result"]) == ["D", "L", "W"]  # most recent first


def test_career_stats_pool_fights_with_stats(fighter_fights):
    careers = charts.career_stats(fighter_fights)
    alpha = careers.loc[_fighter_id(fighter_fights, "Alpha")]
    bravo = careers.loc[_fighter_id(fighter_fights, "Bravo")]
    # Only the title fight (6.5 min) has stats; UFC 1 rows are empty.
    assert alpha["fights"] == 1
    assert alpha["Sig. strikes landed / min"] == pytest.approx(2.0)
    assert alpha["Sig. strike defence"] == pytest.approx(1 - 26 / 40)
    assert alpha["Control share"] == pytest.approx(0.1)
    assert np.isnan(alpha["Takedown defence"])  # opponent attempted no takedowns
    assert np.isnan(alpha["Finish rate (wins)"])  # no wins in fights with stats
    assert bravo["Finish rate (wins)"] == 1.0
    assert _fighter_id(fighter_fights, "Charlie") not in careers.index


def test_radar_percentiles_against_weight_class_peers():
    careers = pd.DataFrame(
        {metric: [1.0, 2.0, 3.0, 4.0, 9.0] for metric in charts.CAREER_METRICS}
        | {"fights": [5, 5, 5, 5, 1], "main_weight_class": ["Lightweight"] * 4 + ["Heavyweight"]},
        index=[10, 11, 12, 13, 14],
    )
    data = charts.radar_data(careers, 12).set_index("metric")
    assert data["percentile"].eq(50).all()  # beats 2 of 4 lightweight peers
    assert data["class_median"].eq(2.5).all() and data["peers"].eq(4).all()
    assert charts.radar_data(careers, 14)["percentile"].isna().all()  # no heavyweight peers


def test_explorer_figures_build(fighter_fights):
    alpha = _fighter_id(fighter_fights, "Alpha")
    history = charts.fighter_history(fighter_fights, alpha)
    careers = charts.career_stats(fighter_fights)
    for fig in (
        charts.career_timeline(history),
        charts.fight_strikes(history),
        charts.career_radar(charts.radar_data(careers, alpha), "Alpha", "Open Weight"),
    ):
        assert isinstance(fig, go.Figure) and fig.data
