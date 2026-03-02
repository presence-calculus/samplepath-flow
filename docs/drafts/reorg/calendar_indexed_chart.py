"""
Generate calendar-indexed (daily and weekly) cumulative arrivals charts
to contrast with the event-indexed step chart.
"""

from datetime import datetime, timedelta

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd

# Load the event-indexed data
df = pd.read_csv(
    "/sessions/friendly-sleepy-archimedes/mnt/uploads/event_indexed_metrics.csv"
)
df["timestamp"] = pd.to_datetime(df["timestamp"])

events = df[["timestamp", "A(T)"]].copy()
events = events.sort_values("timestamp")


def get_arrival_count_at(t, events_df):
    """Get A(T) at time t by finding the last event at or before t."""
    prior = events_df[events_df["timestamp"] <= t]
    if len(prior) == 0:
        return 0
    return int(prior.iloc[-1]["A(T)"])


# Style matching the existing chart exactly
line_color = "#B0A0D0"  # lavender line from reference
dot_color = "#6060C0"  # purple-blue dots from reference
marker_size = 5

# Date range
start_date = events["timestamp"].min().normalize()
end_date = events["timestamp"].max().normalize() + timedelta(days=1)

# --- Daily calendar boundaries ---
daily_dates = pd.date_range(start=start_date, end=end_date, freq="D")
daily_values = [get_arrival_count_at(t, events) for t in daily_dates]

# --- Weekly calendar boundaries (Monday) ---
first_monday = start_date - timedelta(days=start_date.weekday())
weekly_dates = pd.date_range(
    start=first_monday, end=end_date + timedelta(days=7), freq="W-MON"
)
weekly_dates = weekly_dates[
    (weekly_dates >= start_date - timedelta(days=7))
    & (weekly_dates <= end_date + timedelta(days=7))
]
weekly_values = [get_arrival_count_at(t, events) for t in weekly_dates]


def style_ax(ax):
    ax.set_ylabel("Count")
    ax.set_xlabel("Calendar Time")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    ax.tick_params(axis="x", rotation=45)
    ax.spines["top"].set_visible(True)
    ax.spines["right"].set_visible(True)
    for spine in ax.spines.values():
        spine.set_linewidth(0.5)
    ax.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    ax.set_ylim(-0.5, int(max(events["A(T)"])) + 1)
    ax.grid(False)


# --- Create combined figure ---
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7), sharex=False)

# Daily chart
ax1.plot(
    daily_dates,
    daily_values,
    color=line_color,
    linewidth=1.0,
    marker="o",
    markersize=marker_size,
    markerfacecolor=dot_color,
    markeredgecolor=dot_color,
    markeredgewidth=0.5,
    zorder=3,
)
ax1.set_title("A(T) — Cumulative Arrivals (Daily Sampling)", fontsize=11)
style_ax(ax1)

# Weekly chart
ax2.plot(
    weekly_dates,
    weekly_values,
    color=line_color,
    linewidth=1.0,
    marker="o",
    markersize=marker_size,
    markerfacecolor=dot_color,
    markeredgecolor=dot_color,
    markeredgewidth=0.5,
    zorder=3,
)
ax2.set_title("A(T) — Cumulative Arrivals (Weekly Sampling)", fontsize=11)
style_ax(ax2)

plt.tight_layout()
plt.savefig(
    "/sessions/friendly-sleepy-archimedes/mnt/samplepath/docs/assets/calendar-indexed-arrivals.png",
    dpi=150,
    bbox_inches="tight",
    facecolor="white",
)
plt.close()

print("Chart saved to docs/assets/calendar-indexed-arrivals.png")
