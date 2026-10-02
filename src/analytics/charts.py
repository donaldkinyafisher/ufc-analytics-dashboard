"""
Present: Plotly figures built from the fights mart.

Each chart has a *_data function (the aggregated table, also shown as the
chart's table view) and a figure function. No Streamlit imports, so these can
be reused from notebooks and tested directly.
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go

# Reference categorical palette (light steps), assigned in fixed slot order.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
SINGLE = SERIES[0]
MUTED = "#898781"
GRID = "rgba(137, 135, 129, 0.25)"

# Rare methods fold into "Other" so the stack stays at five series.
METHOD_BUCKETS = {"KO/TKO": "KO/TKO", "SUB": "SUB", "U-DEC": "U-DEC", "S-DEC": "S-DEC"}
METHOD_ORDER = ["KO/TKO", "SUB", "U-DEC", "S-DEC", "Other"]
METHOD_COLORS = dict(zip(METHOD_ORDER, SERIES))

# Per-minute rates explode in very short fights, so rate charts skip them.
MIN_RATE_SECONDS = 60

# label -> (column suffix, axis tick format)
PHYSICAL_METRICS = {
    "Height (in)": ("height_in", ".0f"),
    "Reach (in)": ("reach_in", ".0f"),
    "Age at fight": ("age", ".0f"),
}
STRIKING_METRICS = {
    "Sig. strikes landed per minute": ("sig_str_per_min", ".1f"),
    "Sig. strike accuracy": ("sig_str_acc", ".0%"),
    "Takedowns per 15 min": ("td_per_15", ".1f"),
    "Takedown accuracy": ("td_acc", ".0%"),
    "Share of fight in control": ("control_share", ".0%"),
}

# Advantage bands for |red - blue|; (bin edges, labels). Age advantage = younger.
ADVANTAGE_BINS = {
    "height_in": ([0, 1, 2, 3, 4, 6, np.inf], ["0-1 in", "1-2 in", "2-3 in", "3-4 in", "4-6 in", "6+ in"]),
    "reach_in": ([0, 1, 2, 3, 4, 6, np.inf], ["0-1 in", "1-2 in", "2-3 in", "3-4 in", "4-6 in", "6+ in"]),
    "age": ([0, 2, 4, 6, 8, 10, np.inf], ["0-2 yrs", "2-4 yrs", "4-6 yrs", "6-8 yrs", "8-10 yrs", "10+ yrs"]),
}

STRIKE_MIX = {
    "Target": ["Head", "Body", "Leg"],
    "Position": ["Distance", "Clinch", "Ground"],
}
MIX_COLORS = {part: color for parts in STRIKE_MIX.values() for part, color in zip(parts, SERIES)}

# label -> fights_mart *_diff column prefix
LEADER_STATS = {
    "Knockdowns": "knockdowns",
    "Sig. strikes landed": "sig_str_landed",
    "Sig. strike accuracy": "sig_str_acc",
    "Control time": "control_time_seconds",
    "Takedowns landed": "td_landed",
    "Submission attempts": "submission_attempts",
}

# Share of fights missing each field, for the data-quality heatmap.
COMPLETENESS_FIELDS = {
    "Weight class": "weight_class_lbs",
    "Height": "red_height_in",
    "Reach": "red_reach_in",
    "Age": "red_age",
    "Stance": "red_stance",
    "Sig. strikes": "red_sig_str_landed",
    "Control time": "red_control_time_seconds",
}


def _layout(fig: go.Figure, *, x_title: str | None = None, y_title: str | None = None,
            percent_y: bool = False, legend: bool = False) -> go.Figure:
    """Shared chrome: recessive grid, legend in one row above the plot."""
    fig.update_layout(
        margin={"l": 8, "r": 8, "t": 40 if legend else 16, "b": 8},
        showlegend=legend,
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "left", "x": 0, "title": None},
        hoverlabel={"namelength": -1},
        bargap=0.2,
    )
    fig.update_xaxes(title=x_title, showgrid=False, linecolor=MUTED)
    fig.update_yaxes(title=y_title, gridcolor=GRID, zeroline=False,
                     tickformat=".0%" if percent_y else None)
    return fig


def _weight_classes_top_down(mart: pd.DataFrame) -> list[str]:
    """Weight classes present in the data, lightest at the top of a horizontal chart."""
    present = set(mart["weight_class"].astype(str))
    return [c for c in mart["weight_class"].cat.categories if c in present][::-1]


def _horizontal_by_weight_class(fig: go.Figure, mart: pd.DataFrame, x_title: str | None = None,
                                tickformat: str | None = None, legend: bool = False) -> go.Figure:
    """Weight classes down the y axis (lightest first), gridlines on x."""
    _layout(fig, x_title=x_title, legend=legend)
    fig.update_yaxes(categoryorder="array", categoryarray=_weight_classes_top_down(mart), showgrid=False)
    fig.update_xaxes(showgrid=True, gridcolor=GRID, tickformat=tickformat)
    return fig


def _box_by_weight_class(values: pd.Series, weight_class: pd.Series, mart: pd.DataFrame,
                         x_title: str, tickformat: str) -> go.Figure:
    keep = values.notna()
    fig = go.Figure(go.Box(
        x=values[keep], y=weight_class[keep].astype(str), orientation="h",
        marker={"color": SINGLE, "size": 3}, line={"width": 1.5}, boxpoints=False,
        hovertemplate=f"%{{y}}: %{{x:{tickformat}}}<extra></extra>",
    ))
    return _horizontal_by_weight_class(fig, mart, x_title, tickformat)


def _distribution_table(values: pd.Series, weight_class: pd.Series) -> pd.DataFrame:
    table = values.astype(float).groupby(weight_class, observed=True).describe()
    return table.loc[table["count"] > 0, ["count", "25%", "50%", "75%"]].round(3).reset_index()


def _share_stack(data: pd.DataFrame, by: str, series: str, order: list[str], colors: dict[str, str],
                 value: str, horizontal: bool) -> go.Figure:
    """100% stacked bars of data["share"], one trace per series value in fixed order."""
    fig = go.Figure()
    category_axis, share_axis = ("y", "x") if horizontal else ("x", "y")
    for name in order:
        subset = data[data[series] == name]
        if subset.empty:
            continue
        category, share = subset[by].astype(str), subset["share"]
        fig.add_trace(go.Bar(
            name=name,
            x=share if horizontal else category, y=category if horizontal else share,
            orientation="h" if horizontal else "v",
            marker={"color": colors[name], "line": {"width": 0}},
            customdata=subset[[value]],
            hovertemplate=f"{name}<br>%{{{category_axis}}}: %{{{share_axis}:.0%}} "
                          f"(%{{customdata[0]:,}} {value})<extra></extra>",
        ))
    fig.update_layout(barmode="stack", legend_traceorder="normal")
    return fig


def _fifty_percent_line(fig: go.Figure, horizontal: bool = False) -> None:
    line = {"color": MUTED, "dash": "dot", "width": 1}
    if horizontal:
        fig.add_vline(x=0.5, line=line)
    else:
        fig.add_hline(y=0.5, line=line)


def fighter_appearances(mart: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Stack red_* and blue_* into one row per fighter per fight."""
    base = mart[["fight_id", "year", "weight_class", "fight_seconds"]]
    return pd.concat(
        [base.assign(corner=corner, **{c: mart[f"{corner}_{c}"] for c in columns}) for corner in ("red", "blue")],
        ignore_index=True,
    )


def _method_bucket(mart: pd.DataFrame) -> pd.Series:
    bucket = mart["method_group"].astype(str).map(METHOD_BUCKETS).fillna("Other")
    return pd.Categorical(bucket, categories=METHOD_ORDER, ordered=True)


# --- Overview -------------------------------------------------------------

def fights_per_year_data(mart: pd.DataFrame) -> pd.DataFrame:
    return (
        mart.groupby("year")
        .agg(fights=("fight_id", "size"), events=("event_id", "nunique"))
        .reset_index()
    )


def fights_per_year(mart: pd.DataFrame) -> go.Figure:
    data = fights_per_year_data(mart)
    fig = go.Figure(go.Bar(
        x=data["year"], y=data["fights"], marker_color=SINGLE, customdata=data[["events"]],
        hovertemplate="%{x}<br>%{y:,} fights<br>%{customdata[0]} events<extra></extra>",
    ))
    return _layout(fig, y_title="Fights")


def completeness_data(mart: pd.DataFrame) -> pd.DataFrame:
    """Share of fights with each field present, by decade (rows = fields)."""
    present = pd.DataFrame({label: mart[column].notna() for label, column in COMPLETENESS_FIELDS.items()})
    present["decade"] = mart["decade"].astype(str) + "s"
    return present.groupby("decade").mean().T


def completeness_heatmap(mart: pd.DataFrame) -> go.Figure:
    data = completeness_data(mart)
    fig = go.Figure(go.Heatmap(
        z=data.to_numpy(), x=data.columns, y=data.index, zmin=0, zmax=1,
        colorscale=[[0, "#cde2fb"], [1, "#104281"]],  # sequential blue, 100 -> 650
        xgap=2, ygap=2, colorbar={"tickformat": ".0%", "thickness": 10},
        text=data.to_numpy(), texttemplate="%{text:.0%}",
        hovertemplate="%{y}, %{x}: %{z:.0%} present<extra></extra>",
    ))
    fig.update_yaxes(autorange="reversed")
    return _layout(fig)


def red_win_share_data(mart: pd.DataFrame) -> pd.DataFrame:
    decided = mart[mart["outcome"] == "win"]
    return (
        decided.assign(red_won=decided["winner_side"] == "red")
        .groupby("year")["red_won"].agg(red_win_share="mean", fights="size")
        .reset_index()
    )


def red_win_share(mart: pd.DataFrame) -> go.Figure:
    """Corner-bias check: a fair corner assignment would sit near 50%."""
    data = red_win_share_data(mart)
    fig = go.Figure(go.Scatter(
        x=data["year"], y=data["red_win_share"], mode="lines+markers",
        line={"color": SINGLE, "width": 2}, marker={"size": 8},
        customdata=data[["fights"]],
        hovertemplate="%{x}: red won %{y:.0%} of %{customdata[0]} fights<extra></extra>",
    ))
    fig.add_hline(y=0.5, line={"color": MUTED, "dash": "dot", "width": 1},
                  annotation_text="50% (no bias)", annotation_position="bottom right")
    fig.update_yaxes(range=[0, 1])
    return _layout(fig, y_title="Red corner win share", percent_y=True)


# --- Outcomes -------------------------------------------------------------

def method_share_data(mart: pd.DataFrame, by: str) -> pd.DataFrame:
    """Share of fights ending by each method, per value of `by` (long format)."""
    counts = (
        mart.assign(method=_method_bucket(mart))
        .groupby([by, "method"], observed=True).size().rename("fights").reset_index()
    )
    counts["share"] = counts["fights"] / counts.groupby(by, observed=True)["fights"].transform("sum")
    return counts


def method_share_by_year(mart: pd.DataFrame) -> go.Figure:
    data = method_share_data(mart, "year")
    fig = _share_stack(data, "year", "method", METHOD_ORDER, METHOD_COLORS, "fights", horizontal=False)
    return _layout(fig, y_title="Share of fights", percent_y=True, legend=True)


def method_share_by_weight_class(mart: pd.DataFrame) -> go.Figure:
    data = method_share_data(mart, "weight_class")
    fig = _share_stack(data, "weight_class", "method", METHOD_ORDER, METHOD_COLORS, "fights", horizontal=True)
    return _horizontal_by_weight_class(fig, mart, tickformat=".0%", legend=True)


def finish_round_data(mart: pd.DataFrame) -> pd.DataFrame:
    finishes = mart[mart["is_finish"]].assign(method=lambda df: df["method_group"].astype(str))
    return finishes.groupby(["round", "method"]).size().rename("fights").reset_index()


def finish_round(mart: pd.DataFrame) -> go.Figure:
    data = finish_round_data(mart)
    fig = go.Figure()
    for method in ("KO/TKO", "SUB"):
        subset = data[data["method"] == method]
        fig.add_trace(go.Bar(
            name=method, x=subset["round"], y=subset["fights"], marker_color=METHOD_COLORS[method],
            hovertemplate=f"{method}, round %{{x}}: %{{y:,}} finishes<extra></extra>",
        ))
    fig.update_layout(barmode="group")
    fig.update_xaxes(dtick=1)
    return _layout(fig, x_title="Round", y_title="Finishes", legend=True)


def fight_duration_data(mart: pd.DataFrame) -> pd.DataFrame:
    return _distribution_table(mart["fight_seconds"] / 60, mart["weight_class"]).round(1)


def fight_duration_by_weight_class(mart: pd.DataFrame) -> go.Figure:
    return _box_by_weight_class(mart["fight_seconds"] / 60, mart["weight_class"], mart,
                                "Fight length (minutes)", ".1f")


# --- Physicals ------------------------------------------------------------

def physical_by_weight_class_data(mart: pd.DataFrame, metric: str) -> pd.DataFrame:
    column, _ = PHYSICAL_METRICS[metric]
    fighters = fighter_appearances(mart, [column])
    return _distribution_table(fighters[column], fighters["weight_class"])


def physical_by_weight_class(mart: pd.DataFrame, metric: str) -> go.Figure:
    column, tickformat = PHYSICAL_METRICS[metric]
    fighters = fighter_appearances(mart, [column])
    return _box_by_weight_class(fighters[column], fighters["weight_class"], mart, metric, tickformat)


def advantage_win_rate_data(mart: pd.DataFrame, metric: str) -> pd.DataFrame:
    """How often the taller / longer / younger fighter wins, by size of the gap.

    Corner-neutral: it compares the advantaged fighter with the winner, so the
    winner-listed-first bias in red/blue does not leak in.
    """
    column, _ = PHYSICAL_METRICS[metric]
    decided = mart[mart["outcome"] == "win"]
    diff = decided[f"{column}_diff"]
    if column == "age":
        diff = -diff  # the younger fighter has the advantage
    has_gap = diff.notna() & (diff != 0)
    decided, diff = decided[has_gap], diff[has_gap]
    advantaged_won = (diff > 0) == (decided["winner_side"] == "red")
    bins, labels = ADVANTAGE_BINS[column]
    gap = pd.cut(diff.abs(), bins=bins, labels=labels)
    return (
        pd.DataFrame({"gap": gap, "advantaged_won": advantaged_won})
        .groupby("gap", observed=True)["advantaged_won"].agg(win_rate="mean", fights="size")
        .reset_index()
    )


def advantage_win_rate(mart: pd.DataFrame, metric: str) -> go.Figure:
    data = advantage_win_rate_data(mart, metric)
    who = {"height_in": "taller", "reach_in": "longer-reach", "age": "younger"}[PHYSICAL_METRICS[metric][0]]
    fig = go.Figure(go.Bar(
        x=data["gap"].astype(str), y=data["win_rate"], marker_color=SINGLE, customdata=data[["fights"]],
        hovertemplate=f"Gap %{{x}}: {who} fighter won %{{y:.0%}} of %{{customdata[0]:,}} fights<extra></extra>",
    ))
    _fifty_percent_line(fig)
    fig.update_yaxes(range=[0, 1])
    return _layout(fig, x_title=f"{metric.split(' (')[0]} gap", y_title=f"Win rate of {who} fighter", percent_y=True)


# --- Striking & grappling -------------------------------------------------

def _rate_fighters(mart: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    fighters = fighter_appearances(mart, columns)
    return fighters[fighters["fight_seconds"] >= MIN_RATE_SECONDS]


def striking_by_weight_class_data(mart: pd.DataFrame, metric: str) -> pd.DataFrame:
    column, _ = STRIKING_METRICS[metric]
    fighters = _rate_fighters(mart, [column])
    return _distribution_table(fighters[column], fighters["weight_class"])


def striking_by_weight_class(mart: pd.DataFrame, metric: str) -> go.Figure:
    column, tickformat = STRIKING_METRICS[metric]
    fighters = _rate_fighters(mart, [column])
    return _box_by_weight_class(fighters[column], fighters["weight_class"], mart, metric, tickformat)


def strike_mix_data(mart: pd.DataFrame, kind: str) -> pd.DataFrame:
    """Pooled share of landed sig. strikes by target or position, per weight class."""
    parts = STRIKE_MIX[kind]
    columns = [f"{part.lower()}_landed" for part in parts]
    fighters = fighter_appearances(mart, columns)
    totals = fighters.groupby("weight_class", observed=True)[columns].sum().astype(float)
    totals.columns = parts
    totals = totals[totals.sum(axis=1) > 0]  # weight classes with no strike data
    data = totals.reset_index().melt(id_vars="weight_class", var_name="part", value_name="strikes")
    data["share"] = data["strikes"] / data.groupby("weight_class", observed=True)["strikes"].transform("sum")
    return data


def strike_mix(mart: pd.DataFrame, kind: str) -> go.Figure:
    data = strike_mix_data(mart, kind)
    fig = _share_stack(data, "weight_class", "part", STRIKE_MIX[kind], MIX_COLORS, "strikes", horizontal=True)
    return _horizontal_by_weight_class(fig, mart, tickformat=".0%", legend=True)


def stat_leader_win_rate_data(mart: pd.DataFrame) -> pd.DataFrame:
    """For each stat, how often the fighter who led it won (fights where it was not tied)."""
    decided = mart[mart["outcome"] == "win"]
    red_won = decided["winner_side"] == "red"
    rows = []
    for label, column in LEADER_STATS.items():
        diff = decided[f"{column}_diff"]
        led = diff.notna() & (diff != 0)
        leader_won = (diff[led] > 0) == red_won[led]
        rows.append({"stat": label, "win_rate": leader_won.mean(), "fights": int(led.sum())})
    return pd.DataFrame(rows).sort_values("win_rate", ignore_index=True)


def stat_leader_win_rate(mart: pd.DataFrame) -> go.Figure:
    data = stat_leader_win_rate_data(mart)
    fig = go.Figure(go.Bar(
        x=data["win_rate"], y=data["stat"], orientation="h", marker_color=SINGLE, customdata=data[["fights"]],
        hovertemplate="Led in %{y}: won %{x:.0%} of %{customdata[0]:,} fights<extra></extra>",
    ))
    _fifty_percent_line(fig, horizontal=True)
    _layout(fig, x_title="Win rate of the fighter who led the stat")
    fig.update_xaxes(range=[0, 1], tickformat=".0%", showgrid=True, gridcolor=GRID)
    fig.update_yaxes(showgrid=False)
    return fig


def control_by_takedowns_data(mart: pd.DataFrame, max_takedowns: int = 6) -> pd.DataFrame:
    fighters = _rate_fighters(mart, ["td_landed", "control_time_seconds"]).dropna(
        subset=["td_landed", "control_time_seconds"]
    )
    takedowns = fighters["td_landed"].clip(upper=max_takedowns).astype(int)
    labels = takedowns.astype(str).where(takedowns < max_takedowns, f"{max_takedowns}+")
    return (
        (fighters["control_time_seconds"] / 60)
        .groupby(pd.Categorical(labels, categories=[*map(str, range(max_takedowns)), f"{max_takedowns}+"]),
                 observed=True)
        .agg(median_control_minutes="median", fighters="size")
        .rename_axis("takedowns").reset_index()
    )


def control_by_takedowns(mart: pd.DataFrame) -> go.Figure:
    data = control_by_takedowns_data(mart)
    fig = go.Figure(go.Bar(
        x=data["takedowns"].astype(str), y=data["median_control_minutes"], marker_color=SINGLE,
        customdata=data[["fighters"]],
        hovertemplate="%{x} takedowns: median %{y:.1f} min control (%{customdata[0]:,} fighters)<extra></extra>",
    ))
    fig.update_xaxes(type="category")
    return _layout(fig, x_title="Takedowns landed in the fight", y_title="Median control time (min)")


# --- Fighter explorer -----------------------------------------------------

RESULT_COLORS = {"W": SERIES[0], "L": SERIES[1], "D": MUTED, "NC": MUTED}
RESULT_LABELS = {"W": "Win", "L": "Loss", "D": "Draw", "NC": "No contest"}

# Radar axes, all oriented so higher is better for the fighter.
CAREER_METRICS = [
    "Sig. strikes landed / min", "Sig. strike accuracy", "Sig. strike defence",
    "Takedowns / 15 min", "Takedown accuracy", "Takedown defence",
    "Control share", "Finish rate (wins)",
]
MIN_PEER_FIGHTS = 3


def _pooled(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    return numerator / denominator.where(denominator > 0)


def career_stats(fighter_fights: pd.DataFrame) -> pd.DataFrame:
    """Career totals turned into rates, one row per fighter (pooled over fights with stats)."""
    with_stats = fighter_fights.dropna(subset=["fighter_sig_str_landed", "fight_seconds"])
    wins = with_stats["result"] == "W"
    totals = with_stats.assign(
        minutes=with_stats["fight_seconds"] / 60,
        win=wins,
        win_finish=wins & with_stats["is_finish"],
    ).groupby("fighter_id")[[
        "minutes", "fight_seconds", "win", "win_finish",
        "fighter_sig_str_landed", "fighter_sig_str_attempted",
        "opponent_sig_str_landed", "opponent_sig_str_attempted",
        "fighter_td_landed", "fighter_td_attempted", "opponent_td_landed", "opponent_td_attempted",
        "fighter_control_time_seconds",
    ]].sum()

    stats = pd.DataFrame({
        "Sig. strikes landed / min": _pooled(totals["fighter_sig_str_landed"], totals["minutes"]),
        "Sig. strike accuracy": _pooled(totals["fighter_sig_str_landed"], totals["fighter_sig_str_attempted"]),
        "Sig. strike defence": 1 - _pooled(totals["opponent_sig_str_landed"], totals["opponent_sig_str_attempted"]),
        "Takedowns / 15 min": 15 * _pooled(totals["fighter_td_landed"], totals["minutes"]),
        "Takedown accuracy": _pooled(totals["fighter_td_landed"], totals["fighter_td_attempted"]),
        "Takedown defence": 1 - _pooled(totals["opponent_td_landed"], totals["opponent_td_attempted"]),
        "Control share": _pooled(totals["fighter_control_time_seconds"], totals["fight_seconds"]),
        "Finish rate (wins)": _pooled(totals["win_finish"], totals["win"]),
        "fights": with_stats.groupby("fighter_id").size(),
    })
    main_class = fighter_fights.groupby("fighter_id")["weight_class"].agg(
        lambda s: s.astype(str).mode().iloc[0]
    )
    return stats.join(main_class.rename("main_weight_class"))


def fighter_record(fighter_fights: pd.DataFrame, fighter_id: int) -> dict:
    fights = fighter_fights[fighter_fights["fighter_id"] == fighter_id]
    counts = fights["result"].value_counts()
    wins = fights[fights["result"] == "W"]
    return {
        "name": fights["fighter_name"].iloc[0],
        "record": f"{counts.get('W', 0)}-{counts.get('L', 0)}-{counts.get('D', 0)}"
                  + (f" ({counts['NC']} NC)" if counts.get("NC", 0) else ""),
        "fights": len(fights),
        "finish_rate": wins["is_finish"].mean() if len(wins) else np.nan,
        "first_fight": fights["event_date"].min(),
        "last_fight": fights["event_date"].max(),
        "main_weight_class": fights["weight_class"].astype(str).mode().iloc[0],
    }


def fighter_history(fighter_fights: pd.DataFrame, fighter_id: int) -> pd.DataFrame:
    """The fighter's fights in date order with a running net record (wins - losses)."""
    fights = fighter_fights[fighter_fights["fighter_id"] == fighter_id].sort_values("event_date")
    net = fights["result"].map({"W": 1, "L": -1}).fillna(0).cumsum()
    return fights.assign(net_record=net.astype(int), fight_number=range(1, len(fights) + 1))


HISTORY_TABLE_COLUMNS = {
    "event_date": "Date", "event_name": "Event", "opponent_name": "Opponent", "result": "Result",
    "method": "Method", "round": "Round", "time": "Time", "weight_class": "Weight class",
    "fighter_sig_str_landed": "Sig. str landed", "opponent_sig_str_landed": "Sig. str absorbed",
    "fighter_td_landed": "TD", "fighter_control_time_seconds": "Control (s)", "fighter_knockdowns": "KD",
}


def fighter_history_table(history: pd.DataFrame) -> pd.DataFrame:
    table = history[list(HISTORY_TABLE_COLUMNS)].rename(columns=HISTORY_TABLE_COLUMNS)
    table["Date"] = table["Date"].dt.date
    return table.iloc[::-1].reset_index(drop=True)  # most recent first


def career_timeline(history: pd.DataFrame) -> go.Figure:
    fig = go.Figure(go.Scatter(
        x=history["event_date"], y=history["net_record"], mode="lines",
        line={"color": MUTED, "width": 2, "shape": "hv"}, hoverinfo="skip", showlegend=False,
    ))
    for result, label in RESULT_LABELS.items():
        fights = history[history["result"] == result]
        if fights.empty:
            continue
        fig.add_trace(go.Scatter(
            x=fights["event_date"], y=fights["net_record"], mode="markers", name=label,
            marker={"color": RESULT_COLORS[result], "size": 10, "line": {"color": "white", "width": 2},
                    "symbol": "circle" if result in ("W", "L") else "diamond"},
            customdata=fights[["opponent_name", "method", "round", "time", "event_name"]],
            hovertemplate="%{x|%d %b %Y}<br>" + label + " vs %{customdata[0]}<br>"
                          "%{customdata[1]}, R%{customdata[2]} %{customdata[3]}<br>"
                          "%{customdata[4]}<extra></extra>",
        ))
    fig.add_hline(y=0, line={"color": MUTED, "width": 1, "dash": "dot"})
    return _layout(fig, y_title="Net record (wins - losses)", legend=True)


def fight_strikes(history: pd.DataFrame) -> go.Figure:
    """Sig. strikes landed vs absorbed in each fight."""
    fights = history.dropna(subset=["fighter_sig_str_landed"])
    labels = fights["event_date"].dt.strftime("%b %Y") + "<br>" + fights["opponent_name"]
    fig = go.Figure()
    for column, name, color in (
        ("fighter_sig_str_landed", "Landed", SERIES[0]),
        ("opponent_sig_str_landed", "Absorbed", SERIES[1]),
    ):
        fig.add_trace(go.Bar(
            x=fights["fight_number"], y=fights[column], name=name, marker_color=color,
            customdata=np.stack([labels, fights["result"]], axis=-1),
            hovertemplate=f"{name}: %{{y}}<br>%{{customdata[0]}} (%{{customdata[1]}})<extra></extra>",
        ))
    fig.update_layout(barmode="group", bargap=0.25)
    fig.update_xaxes(dtick=max(1, len(fights) // 15))
    return _layout(fig, x_title="Fight number", y_title="Sig. strikes", legend=True)


def radar_data(careers: pd.DataFrame, fighter_id: int) -> pd.DataFrame:
    """Fighter's career rates vs their main weight class, as percentiles (50 = class median)."""
    fighter = careers.loc[fighter_id]
    peers = careers[(careers["main_weight_class"] == fighter["main_weight_class"])
                    & (careers["fights"] >= MIN_PEER_FIGHTS)]
    rows = []
    for metric in CAREER_METRICS:
        values = peers[metric].dropna()
        value = fighter[metric]
        percentile = 100 * (values < value).mean() if pd.notna(value) and len(values) else np.nan
        rows.append({"metric": metric, "fighter": value, "class_median": values.median(),
                     "percentile": percentile, "peers": len(values)})
    return pd.DataFrame(rows)


def career_radar(data: pd.DataFrame, fighter_name: str, weight_class: str) -> go.Figure:
    closed = pd.concat([data, data.iloc[[0]]])  # repeat the first axis to close the shape
    closed["metric"] = closed["metric"].str.replace(r"^(\S+ \S+) ", r"\1<br>", regex=True)
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=[50] * len(closed), theta=closed["metric"], name=f"{weight_class} median",
        mode="lines", line={"color": MUTED, "dash": "dot", "width": 1.5}, hoverinfo="skip",
    ))
    fig.add_trace(go.Scatterpolar(
        r=closed["percentile"], theta=closed["metric"], name=fighter_name, mode="lines+markers",
        line={"color": SERIES[0], "width": 2}, marker={"size": 8}, fill="toself",
        fillcolor="rgba(42, 120, 214, 0.15)",
        customdata=closed[["fighter", "class_median"]],
        hovertemplate="%{theta}: %{r:.0f}th percentile<br>"
                      "Fighter %{customdata[0]:.2f} vs median %{customdata[1]:.2f}<extra></extra>",
    ))
    fig.update_layout(polar={"radialaxis": {"range": [0, 100], "tickvals": [25, 50, 75], "angle": 90,
                                            "gridcolor": GRID, "tickfont": {"color": MUTED, "size": 10}},
                             "angularaxis": {"gridcolor": GRID, "rotation": 90}})
    _layout(fig, legend=True)
    fig.update_layout(height=480, margin={"l": 80, "r": 80, "t": 60, "b": 40})
    return fig
