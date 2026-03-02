"""
Generate calendar-indexed (daily and weekly) Cumulative Flow Diagrams
to contrast with the event-indexed CFD.

Conventions match the event-indexed CFD:
- Purple/violet for A(T) arrivals
- Green for D(T) departures
- Grey shaded area between them (presence mass)
"""

from datetime import datetime, timedelta

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd

# Load the event-indexed data
df = pd.read_csv(
    "/sessions/friendly-sleepy-archimedes/mnt/uploads/event_indexed_metrics-e23f0140.csv"
)
df["timestamp"] = pd.to_datetime(df["timestamp"])
df = df.sort_values("timestamp")

# Colors matching the event-indexed CFD
arrival_line_color = "#B0A0D0"  # lavender line for A(T)
arrival_dot_color = "#6060C0"  # purple-blue dots for arrivals
departure_line_color = "#90C090"  # light green line for D(T)
departure_dot_color = "#408040"  # green dots for departures
fill_color = "#D0D0D0"  # grey fill for presence mass
fill_alpha = 0.5
marker_size = 5


def get_metric_at(t, events_df, col):
    """Get metric value at time t by finding the last event at or before t."""
    prior = events_df[events_df["timestamp"] <= t]
    if len(prior) == 0:
        return 0
    return int(prior.iloc[-1][col])


# Date range
start_date = df["timestamp"].min().normalize()
end_date = df["timestamp"].max().normalize() + timedelta(days=1)

# --- Daily calendar boundaries ---
daily_dates = pd.date_range(start=start_date, end=end_date, freq="D")
daily_arrivals = [get_metric_at(t, df, "A(T)") for t in daily_dates]
daily_departures = [get_metric_at(t, df, "D(T)") for t in daily_dates]

# --- Weekly calendar boundaries (Monday) ---
first_monday = start_date - timedelta(days=start_date.weekday())
weekly_dates = pd.date_range(
    start=first_monday, end=end_date + timedelta(days=7), freq="W-MON"
)
weekly_dates = weekly_dates[
    (weekly_dates >= start_date - timedelta(days=7))
    & (weekly_dates <= end_date + timedelta(days=7))
]
weekly_arrivals = [get_metric_at(t, df, "A(T)") for t in weekly_dates]
weekly_departures = [get_metric_at(t, df, "D(T)") for t in weekly_dates]


def style_ax(ax, max_count):
    ax.set_ylabel("Count")
    ax.set_xlabel("Calendar Time")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    ax.tick_params(axis="x", rotation=45)
    ax.spines["top"].set_visible(True)
    ax.spines["right"].set_visible(True)
    for spine in ax.spines.values():
        spine.set_linewidth(0.5)
    ax.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    ax.set_ylim(-0.5, max_count + 1)
    ax.grid(False)


max_count = int(max(df["A(T)"]))

# --- Create combined figure (vertical layout) ---
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=False)

# --- Daily CFD ---
ax1.fill_between(
    daily_dates,
    daily_departures,
    daily_arrivals,
    color=fill_color,
    alpha=fill_alpha,
    zorder=1,
)
ax1.plot(
    daily_dates,
    daily_arrivals,
    color=arrival_line_color,
    linewidth=1.0,
    marker="o",
    markersize=marker_size,
    markerfacecolor=arrival_dot_color,
    markeredgecolor=arrival_dot_color,
    markeredgewidth=0.5,
    zorder=3,
    label="A(T) — Cumulative arrivals",
)
ax1.plot(
    daily_dates,
    daily_departures,
    color=departure_line_color,
    linewidth=1.0,
    marker="o",
    markersize=marker_size,
    markerfacecolor=departure_dot_color,
    markeredgecolor=departure_dot_color,
    markeredgewidth=0.5,
    zorder=3,
    label="D(T) — Cumulative departures",
)
ax1.set_title("Calendar-Indexed Cumulative Flow Diagram (Daily Sampling)", fontsize=11)
ax1.legend(fontsize=8, loc="upper left")
style_ax(ax1, max_count)

# --- Weekly CFD ---
ax2.fill_between(
    weekly_dates,
    weekly_departures,
    weekly_arrivals,
    color=fill_color,
    alpha=fill_alpha,
    zorder=1,
)
ax2.plot(
    weekly_dates,
    weekly_arrivals,
    color=arrival_line_color,
    linewidth=1.0,
    marker="o",
    markersize=marker_size,
    markerfacecolor=arrival_dot_color,
    markeredgecolor=arrival_dot_color,
    markeredgewidth=0.5,
    zorder=3,
    label="A(T) — Cumulative arrivals",
)
ax2.plot(
    weekly_dates,
    weekly_departures,
    color=departure_line_color,
    linewidth=1.0,
    marker="o",
    markersize=marker_size,
    markerfacecolor=departure_dot_color,
    markeredgecolor=departure_dot_color,
    markeredgewidth=0.5,
    zorder=3,
    label="D(T) — Cumulative departures",
)
ax2.set_title("Calendar-Indexed Cumulative Flow Diagram (Weekly Sampling)", fontsize=11)
ax2.legend(fontsize=8, loc="upper left")
style_ax(ax2, max_count)

plt.tight_layout()
plt.savefig(
    "/sessions/friendly-sleepy-archimedes/mnt/samplepath/docs/assets/calendar-indexed-cfd.png",
    dpi=150,
    bbox_inches="tight",
    facecolor="white",
)
plt.close()

print("Chart saved to docs/assets/calendar-indexed-cfd.png")
