# -*- coding: utf-8 -*-
# Copyright (c) 2025 Krishna Kumar
# SPDX-License-Identifier: MIT
from dataclasses import dataclass
from typing import Any, List, Optional, Sequence, Tuple

from matplotlib import pyplot as plt
from matplotlib.axes import Axes
import matplotlib.dates as mdates
from matplotlib.figure import Figure
import numpy as np
import pandas as pd
from pandas.tseries.frequencies import to_offset

from samplepath.plots.chart_config import ChartConfig, ColorConfig


@dataclass
class ScatterOverlay:
    """A scatter series to overlay on a chart.

    Attributes
    ----------
    x : List[pd.Timestamp]
        X-coordinates (timestamps) for scatter points.
    y : List[float]
        Y-coordinates for scatter points.
    color : str
        Color for the scatter points and drop lines.
    label : str
        Legend label for this series.
    drop_lines : bool
        If True, draw vertical lines from each point to the x-axis.
    """

    x: List[pd.Timestamp]
    y: List[float]
    color: str
    label: str
    drop_lines: bool = False


def add_caption(fig: Figure, text: str) -> None:
    """Add a caption below the x-axis."""
    fig.subplots_adjust(bottom=0.28)
    fig.text(
        0.5,
        0.005,
        text,
        ha="center",
        va="bottom",
        fontsize=9,
    )


def resolve_caption(filter_result: Optional[Any]) -> Optional[str]:
    """Build a caption string from a filter result-like object."""
    if filter_result and getattr(filter_result, "label", None):
        return getattr(filter_result, "display", None)
    return "Filters: None"


_WEEKDAY_MAP = {
    "MON": mdates.MONDAY,
    "TUE": mdates.TUESDAY,
    "WED": mdates.WEDNESDAY,
    "THU": mdates.THURSDAY,
    "FRI": mdates.FRIDAY,
    "SAT": mdates.SATURDAY,
    "SUN": mdates.SUNDAY,
}


def _calendar_tick_config(
    unit: str,
) -> Optional[
    Tuple[mdates.DateLocator, mdates.DateFormatter, Optional[mdates.DateLocator]]
]:
    """Map a pandas frequency alias to (locator, formatter, minor_locator | None).

    Returns None for unrecognised frequencies.
    """
    try:
        offset = to_offset(unit)
    except (ValueError, TypeError):
        return None

    if isinstance(offset, pd.tseries.offsets.Day):
        locator = mdates.AutoDateLocator()
        formatter = mdates.ConciseDateFormatter(locator)
        return locator, formatter, None

    if isinstance(offset, pd.tseries.offsets.Week):
        anchor_name = offset.weekday  # 0=MON … 6=SUN
        weekday_const = list(_WEEKDAY_MAP.values())[anchor_name]
        locator = mdates.WeekdayLocator(byweekday=weekday_const)
        formatter = mdates.DateFormatter("%b %d")
        return locator, formatter, None

    if isinstance(offset, pd.tseries.offsets.MonthBegin):
        locator = mdates.MonthLocator()
        formatter = mdates.DateFormatter("%b\n%Y")
        return locator, formatter, None

    if isinstance(offset, pd.tseries.offsets.QuarterBegin):
        start_month = offset.startingMonth
        months = sorted([(start_month + 3 * i - 1) % 12 + 1 for i in range(4)])
        locator = mdates.MonthLocator(bymonth=months)
        formatter = mdates.DateFormatter("%b\n%Y")
        return locator, formatter, None

    if isinstance(offset, pd.tseries.offsets.YearBegin):
        locator = mdates.YearLocator()
        formatter = mdates.DateFormatter("%Y")
        return locator, formatter, None

    return None


def apply_calendar_ticks(ax: Axes, unit: Optional[str]) -> None:
    """Apply calendar-aware tick locators and formatters to the x-axis.

    No-op when *unit* is ``None`` or when ``_calendar_tick_config`` returns
    ``None`` (e.g. for ``"timestamp"`` or unrecognised frequencies).
    """
    if unit is None:
        return
    config = _calendar_tick_config(unit)
    if config is None:
        return
    locator, formatter, minor = config
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(formatter)
    if minor is not None:
        ax.xaxis.set_minor_locator(minor)


def format_date_axis(ax: Axes, unit: str = "timestamp") -> None:
    """Format the x-axis for dates if possible."""
    apply_calendar_ticks(ax, unit)
    ax.set_xlabel(ChartConfig.freq_display_label(unit))
    try:
        ax.figure.autofmt_xdate()
    except Exception:
        pass


def apply_gridlines(ax: Axes, enabled: bool = True) -> None:
    """Enable minor ticks and gridlines on an axis when requested."""
    if not enabled:
        return
    if not hasattr(ax, "minorticks_on") or not hasattr(ax, "grid"):
        return
    ax.minorticks_on()
    ax.grid(
        True,
        which="major",
        axis="both",
        linestyle="--",
        linewidth=0.6,
        alpha=0.4,
    )
    ax.grid(
        True,
        which="minor",
        axis="both",
        linestyle=":",
        linewidth=0.4,
        alpha=0.25,
    )


def format_axis(
    ax: Axes, title: str, unit: str, ylabel: str, grid_lines: bool = True
) -> None:
    """Set axis labels, title, and legend with date formatting."""
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.legend()
    format_date_axis(ax, unit=unit)
    apply_gridlines(ax, enabled=grid_lines)


def format_fig(caption: Optional[str], fig: Figure) -> None:
    """Finalize figure with optional caption and layout adjustment."""
    fig.tight_layout()
    if caption:
        add_caption(fig, caption)


def format_and_save(
    fig: Figure,
    ax: Axes,
    title: str,
    ylabel: str,
    unit: str,
    caption: Optional[str],
    out_path: str,
    grid_lines: bool = True,
) -> None:
    """Format the axis, add optional caption, save the figure, and close it."""
    format_axis(ax, title, unit, ylabel, grid_lines=grid_lines)
    format_fig(caption, fig)
    fig.savefig(out_path)
    plt.close(fig)


def init_fig_ax(figsize: Tuple[float, float] = (10.0, 3.4)) -> Tuple[Figure, Axes]:
    fig, ax = plt.subplots(figsize=figsize)
    return fig, ax


def _clip_axis_to_percentile(
    ax: Axes,
    times: List[pd.Timestamp],
    values: Sequence[float],
    upper_p: Optional[float] = None,
    lower_p: Optional[float] = None,
    warmup_seconds: float = 0.0,
) -> None:
    """Clip y-axis limits based on percentiles of the data.

    Parameters
    ----------
    ax : Axes
        Matplotlib axes to modify.
    times : List[pd.Timestamp]
        X-coordinates (used for warmup filtering).
    values : Sequence[float]
        Y-coordinates to compute percentiles from.
    upper_p : optional float
        Upper percentile for y-axis limit (e.g., 99.5).
    lower_p : optional float
        Lower percentile for y-axis limit.
    warmup_seconds : float
        Seconds from start to exclude from percentile calculation.
    """
    if upper_p is None and lower_p is None:
        return
    vals = np.asarray(values, dtype=float)
    if vals.size == 0:
        return
    mask = np.isfinite(vals)
    if warmup_seconds and times:
        t0 = times[0]
        ages_seconds = np.array([(t - t0).total_seconds() for t in times])
        mask &= ages_seconds >= float(warmup_seconds)
    data = vals[mask]
    if data.size == 0 or not np.isfinite(data).any():
        return
    top = np.nanpercentile(data, upper_p) if upper_p is not None else np.nanmax(data)
    bottom = np.nanpercentile(data, lower_p) if lower_p is not None else 0.0
    if not np.isfinite(top) or not np.isfinite(bottom) or top <= bottom:
        return
    ax.set_ylim(float(bottom), float(top))


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Core render functions (render to existing Axes)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def render_step_chart(
    ax: Axes,
    times: Sequence[pd.Timestamp],
    values: Sequence[float],
    *,
    label: str = "N(t)",
    color: str = "tab:blue",
    fill: bool = False,
    fill_color: Optional[str] = None,
    linewidth: float = 1.0,
    overlays: Optional[List[ScatterOverlay]] = None,
    sampling_frequency: Optional[str] = None,
) -> None:
    """Render a step chart to an existing axes.

    Parameters
    ----------
    ax : Axes
        Matplotlib axes to render to.
    times : Sequence[pd.Timestamp]
        X-coordinates (timestamps).
    values : Sequence[float]
        Y-coordinates.
    label : str
        Legend label for the step series.
    color : str
        Line and fill color.
    fill : bool
        If True, shade the area under the step curve.
    linewidth : float
        Line width for the step curve.
    overlays : optional List[ScatterOverlay]
        Scatter series to overlay on the chart.
    sampling_frequency : optional str
        When set, keep the original color even with overlays present
        (rug-plot mode — overlays sit at y=0, so de-emphasis is unnecessary).
    """
    effective_color = color
    fill_alpha = 0.15 if overlays and sampling_frequency is None else 0.3
    ax.step(
        times,
        values,
        where="post",
        label=label,
        color=effective_color,
        linewidth=linewidth,
        alpha=0.4 if overlays and sampling_frequency is None else 1.0,
    )
    if fill:
        resolved_fill_color = fill_color or effective_color
        ax.fill_between(
            times,
            values,
            step="post",
            alpha=fill_alpha,
            color=resolved_fill_color,
        )
    if overlays:
        for i, overlay in enumerate(overlays):
            if not overlay.x:
                continue
            overlay_scatter_alpha = (
                1.0 if overlay.color == ColorConfig.departure_color else 1.0
            )
            overlay_vline_alpha = (
                0.7 if overlay.color == ColorConfig.departure_color else 0.5
            )
            if overlay.drop_lines:
                ax.vlines(
                    overlay.x,
                    0,
                    overlay.y,
                    colors=overlay.color,
                    linewidths=0.5,
                    alpha=overlay_vline_alpha,
                    zorder=4 + i,
                )
            ax.scatter(
                overlay.x,
                overlay.y,
                color=overlay.color,
                s=2,
                alpha=overlay_scatter_alpha,
                zorder=5 + i,
                label=overlay.label,
            )


def render_line_chart(
    ax: Axes,
    times: Sequence[pd.Timestamp],
    values: Sequence[float],
    *,
    label: str,
    color: str = "tab:blue",
    linewidth: float = 1.0,
    overlays: Optional[List[ScatterOverlay]] = None,
    sampling_frequency: Optional[str] = None,
) -> None:
    """Render a line chart to an existing axes.

    Parameters
    ----------
    ax : Axes
        Matplotlib axes to render to.
    times : Sequence[pd.Timestamp]
        X-coordinates (timestamps).
    values : Sequence[float]
        Y-coordinates.
    label : str
        Legend label for the line series.
    color : str
        Line color.
    linewidth : float
        Line width.
    overlays : optional List[ScatterOverlay]
        Scatter series to overlay on the chart.
    sampling_frequency : optional str
        When set, add markers to make individual data points visible.
    """
    effective_color = color
    plot_kwargs: dict[str, Any] = dict(
        label=label,
        color=effective_color,
        linewidth=linewidth,
        alpha=0.4 if overlays and sampling_frequency is None else 1.0,
    )
    if sampling_frequency is not None:
        plot_kwargs["marker"] = "o"
        plot_kwargs["markersize"] = 4
    ax.plot(times, values, **plot_kwargs)
    if overlays:
        for i, overlay in enumerate(overlays):
            if not overlay.x:
                continue
            overlay_scatter_alpha = (
                1.0 if overlay.color == ColorConfig.departure_color else 1.0
            )
            overlay_vline_alpha = (
                0.7 if overlay.color == ColorConfig.departure_color else 0.5
            )
            if overlay.drop_lines:
                ax.vlines(
                    overlay.x,
                    0,
                    overlay.y,
                    colors=overlay.color,
                    linewidths=0.5,
                    alpha=overlay_vline_alpha,
                    zorder=4 + i,
                )
            ax.scatter(
                overlay.x,
                overlay.y,
                color=overlay.color,
                s=2,
                alpha=overlay_scatter_alpha,
                zorder=5 + i,
                label=overlay.label,
            )


def render_scatter_chart(
    ax: Axes,
    x_vals: Sequence[float],
    y_vals: Sequence[float],
    *,
    label: Optional[str] = None,
    color: str = "tab:blue",
    alpha: float = 0.7,
    size: float = 18,
    overlays: Optional[List[ScatterOverlay]] = None,
    drop_lines: str = "none",
    drop_line_alpha: float = 0.25,
    drop_line_color: Optional[str] = None,
) -> None:
    """Render a scatter chart to an existing axes.

    Parameters
    ----------
    ax : Axes
        Matplotlib axes to render to.
    x_vals : Sequence[float]
        X-coordinates.
    y_vals : Sequence[float]
        Y-coordinates.
    label : optional str
        Legend label for the scatter series.
    color : str
        Color for points.
    alpha : float
        Alpha for points.
    size : float
        Marker size.
    overlays : optional List[ScatterOverlay]
        Scatter series to overlay on the chart.
    drop_lines : str
        One of "none", "vertical", "both" to control base drop lines.
    drop_line_alpha : float
        Alpha for drop lines.
    drop_line_color : optional str
        Color for drop lines (defaults to point color).
    """
    ax.scatter(
        x_vals,
        y_vals,
        s=size,
        alpha=alpha,
        color=color,
        label=label,
    )
    if drop_lines in ("vertical", "both"):
        ax.vlines(
            x_vals,
            0,
            y_vals,
            colors=drop_line_color or color,
            linewidths=0.5,
            alpha=drop_line_alpha,
        )
    if drop_lines == "both":
        ax.hlines(
            y_vals,
            0,
            x_vals,
            colors=drop_line_color or color,
            linewidths=0.5,
            alpha=drop_line_alpha,
        )
    if overlays:
        for i, overlay in enumerate(overlays):
            if not overlay.x:
                continue
            if overlay.drop_lines:
                ax.vlines(
                    overlay.x,
                    0,
                    overlay.y,
                    colors=overlay.color,
                    linewidths=0.5,
                    alpha=0.5,
                    zorder=4 + i,
                )
            ax.scatter(
                overlay.x,
                overlay.y,
                color=overlay.color,
                s=2,
                zorder=5 + i,
                label=overlay.label,
            )


def render_lambda_chart(
    ax: Axes,
    times: Sequence[pd.Timestamp],
    values: Sequence[float],
    *,
    label: str = "Λ(T)",
    color: str = "tab:blue",
    linewidth: float = 1.0,
    pctl_upper: Optional[float] = None,
    pctl_lower: Optional[float] = None,
    warmup_seconds: float = 0.0,
) -> None:
    """Render Λ(T) chart with optional percentile-based y-axis clipping.

    Parameters
    ----------
    ax : Axes
        Matplotlib axes to render to.
    times : Sequence[pd.Timestamp]
        X-coordinates (timestamps).
    values : Sequence[float]
        Y-coordinates (Lambda values).
    label : str
        Legend label for the line series.
    color : str
        Line color.
    linewidth : float
        Line width.
    pctl_upper : optional float
        Upper percentile for y-axis clipping (e.g., 99.5).
    pctl_lower : optional float
        Lower percentile for y-axis clipping.
    warmup_seconds : float
        Seconds to exclude from percentile calculation (warmup period).
    """
    ax.plot(times, values, label=label, color=color, linewidth=linewidth)
    if pctl_upper is not None or pctl_lower is not None:
        _clip_axis_to_percentile(
            ax, list(times), values, pctl_upper, pctl_lower, warmup_seconds
        )


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Overlay helpers
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def build_event_overlays(
    times: Sequence[pd.Timestamp],
    values: Sequence[float],
    arrival_times: Optional[List[pd.Timestamp]],
    departure_times: Optional[List[pd.Timestamp]],
    drop_lines_for_arrivals: bool = True,
    drop_lines_for_departures: bool = True,
    calendar_mode: bool = False,
) -> Optional[List[ScatterOverlay]]:
    """Build arrival/departure overlays by mapping event x-values to series y-values.

    The x-values come from the point events (arrivals/departures).
    The y-values are looked up from the series at those x positions.
    Events with no matching x-value in the series are excluded.

    When *calendar_mode* is True the series timestamps won't match event
    timestamps, so all events are kept at y=0 (rug plot) with drop lines
    disabled.

    Parameters
    ----------
    times : Sequence[pd.Timestamp]
        X-coordinates of the series.
    values : Sequence[float]
        Y-coordinates of the series.
    arrival_times : optional List[pd.Timestamp]
        Timestamps of arrival events.
    departure_times : optional List[pd.Timestamp]
        Timestamps of departure events.
    drop_lines_for_arrivals : bool
        If True, draw vertical lines for arrivals.
    drop_lines_for_departures : bool
        If True, draw vertical lines for departures.
    calendar_mode : bool
        If True, place all events at y=0 (rug plot) with no drop lines.

    Returns
    -------
    Optional[List[ScatterOverlay]]
        List of overlays, or None if no event times provided.
    """
    if arrival_times is None or departure_times is None:
        return None

    if calendar_mode:
        arrival_x = list(arrival_times)
        arrival_y = [0.0] * len(arrival_x)
        departure_x = list(departure_times)
        departure_y = [0.0] * len(departure_x)
        drop_lines_for_arrivals = False
        drop_lines_for_departures = False
    else:
        time_to_idx = {t: i for i, t in enumerate(times)}
        arrival_x = [t for t in arrival_times if t in time_to_idx]
        arrival_y = [float(values[time_to_idx[t]]) for t in arrival_x]
        departure_x = [t for t in departure_times if t in time_to_idx]
        departure_y = [float(values[time_to_idx[t]]) for t in departure_x]

    return [
        ScatterOverlay(
            x=arrival_x,
            y=arrival_y,
            color=ColorConfig.arrival_color,
            label="Arrival",
            drop_lines=drop_lines_for_arrivals,
        ),
        ScatterOverlay(
            x=departure_x,
            y=departure_y,
            color=ColorConfig.departure_color,
            label="Departure",
            drop_lines=drop_lines_for_departures,
        ),
    ]


def compute_flow_cloud_positions(
    timestamps: List[pd.Timestamp],
    window_fraction: float = 0.05,
    vertical_range: Tuple[float, float] = (0.35, 0.65),
) -> np.ndarray:
    """Compute vertical positions for flow cloud visualization.

    For each timestamp, count neighbors within time window. Assign vertical
    offset based on local density + sequential pattern. Dense regions spread
    vertically; sparse regions cluster near center.

    Parameters
    ----------
    timestamps : List[pd.Timestamp]
        Event timestamps in chronological order.
    window_fraction : float
        Sliding window size as fraction of total duration (default 0.05 = 5%).
    vertical_range : Tuple[float, float]
        Min and max vertical positions (default (0.35, 0.65)).

    Returns
    -------
    np.ndarray
        Vertical positions for each timestamp in range [min, max].
    """
    if not timestamps:
        return np.array([])

    if len(timestamps) == 1:
        return np.array([0.5])

    # Convert to numeric timestamps (nanoseconds since epoch)
    ts_numeric = np.array([t.value for t in timestamps])

    # Calculate window size
    total_duration = ts_numeric[-1] - ts_numeric[0]
    if total_duration == 0:
        return np.full(len(timestamps), 0.5)

    window_size = total_duration * window_fraction

    # For each point, compute local density (neighbor count in window)
    densities = np.zeros(len(timestamps))
    for i, ts_val in enumerate(ts_numeric):
        lower_bound = ts_val - window_size / 2
        upper_bound = ts_val + window_size / 2
        density = np.sum((ts_numeric >= lower_bound) & (ts_numeric <= upper_bound))
        densities[i] = density

    # Normalize densities to [0, 1]
    max_density = densities.max()
    if max_density > 0:
        density_norm = densities / max_density
    else:
        density_norm = np.zeros(len(timestamps))

    # Compute vertical positions with adaptive spreading based on density
    positions = np.zeros(len(timestamps))

    # Adaptive strategy: sparse regions cluster tight, dense regions spread wide
    # Density threshold: points below this stay tight, above spread out
    density_threshold = 0.4

    y_min, y_max = vertical_range
    y_center = (y_min + y_max) / 2
    y_range = y_max - y_min

    for i, dens in enumerate(density_norm):
        # Use timestamp-based alternation (not index-based) for consistency
        # across arrival and departure lists when they're close in time
        ts_val = ts_numeric[i]
        alternation = 1 if (ts_val // 1_000_000_000) % 2 == 0 else -1

        if dens < density_threshold:
            # Sparse: cluster tightly around center (smaller offset)
            tight_amplitude = y_range * 0.25
            offset = alternation * (dens / density_threshold) * tight_amplitude
            positions[i] = y_center + offset
        else:
            # Dense: spread toward edges of the assigned range
            density_fraction = (dens - density_threshold) / (1.0 - density_threshold)
            # Spread increases as density increases
            wide_amplitude = y_range * 0.4
            offset = alternation * (density_fraction + 0.2) * wide_amplitude
            positions[i] = y_center + offset

        # Always clamp to assigned vertical range
        positions[i] = np.clip(positions[i], y_min, y_max)

    return positions


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Chart recipes (complete chart definitions with consistent styling)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def render_N_chart(
    ax: Axes,
    times: Sequence[pd.Timestamp],
    N_vals: Sequence[float],
    *,
    arrival_times: Optional[List[pd.Timestamp]] = None,
    departure_times: Optional[List[pd.Timestamp]] = None,
    with_event_marks: bool = False,
    show_title: bool = True,
) -> None:
    """Render the complete N(t) sample path chart with consistent styling.

    This is the single source of truth for how N(t) charts appear everywhere
    in the application - standalone or in composite layouts.

    Parameters
    ----------
    ax : Axes
        Matplotlib axes to render to.
    times : Sequence[pd.Timestamp]
        Observation times.
    N_vals : Sequence[float]
        N(t) values at each observation time.
    arrival_times : optional List[pd.Timestamp]
        Timestamps of arrival events (for overlay markers).
    departure_times : optional List[pd.Timestamp]
        Timestamps of departure events (for overlay markers).
    with_event_marks : bool
        If True and arrival/departure times provided, show event markers.
    show_title : bool
        If True, set the axis title. Set False for composite layouts
        that manage titles separately.
    """
    # Build overlays if requested
    overlays = (
        build_event_overlays(times, N_vals, arrival_times, departure_times)
        if with_event_marks
        else None
    )

    render_step_chart(ax, times, N_vals, label="N(t)", fill=True, overlays=overlays)

    if show_title:
        ax.set_title("N(t) — Sample Path")
    ax.set_ylabel("N(t)")
    ax.legend()


def render_LT_chart(
    ax: Axes,
    times: Sequence[pd.Timestamp],
    L_vals: Sequence[float],
    *,
    arrival_times: Optional[List[pd.Timestamp]] = None,
    departure_times: Optional[List[pd.Timestamp]] = None,
    with_event_marks: bool = False,
    show_title: bool = True,
) -> None:
    """Render the complete L(T) time-average chart with consistent styling.

    This is the single source of truth for how L(T) charts appear everywhere
    in the application - standalone or in composite layouts.

    Parameters
    ----------
    ax : Axes
        Matplotlib axes to render to.
    times : Sequence[pd.Timestamp]
        Observation times.
    L_vals : Sequence[float]
        L(T) values at each observation time.
    arrival_times : optional List[pd.Timestamp]
        Timestamps of arrival events (for overlay markers).
    departure_times : optional List[pd.Timestamp]
        Timestamps of departure events (for overlay markers).
    with_event_marks : bool
        If True and arrival/departure times provided, show event markers.
    show_title : bool
        If True, set the axis title. Set False for composite layouts
        that manage titles separately.
    """
    # Build overlays if requested
    overlays = (
        build_event_overlays(times, L_vals, arrival_times, departure_times)
        if with_event_marks
        else None
    )

    render_line_chart(ax, times, L_vals, label="L(T)", overlays=overlays)

    if show_title:
        ax.set_title("L(T) — Time-Average of N(t)")
    ax.set_ylabel("L(T)")
    ax.legend()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Standalone draw functions (create figure, render, save)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def draw_N_chart(
    times: List[pd.Timestamp],
    N_vals: Sequence[float],
    out_path: str,
    *,
    arrival_times: Optional[List[pd.Timestamp]] = None,
    departure_times: Optional[List[pd.Timestamp]] = None,
    with_event_marks: bool = False,
    unit: str = "timestamp",
    caption: Optional[str] = None,
) -> None:
    """Draw a standalone N(t) chart and save to file.

    This uses render_N_chart internally, ensuring the same visual
    appearance as N(t) charts in composite layouts.
    """
    fig, ax = init_fig_ax()
    render_N_chart(
        ax,
        times,
        N_vals,
        arrival_times=arrival_times,
        departure_times=departure_times,
        with_event_marks=with_event_marks,
    )
    format_date_axis(ax, unit=unit)
    format_fig(caption, fig)
    fig.savefig(out_path)
    plt.close(fig)


def draw_LT_chart(
    times: List[pd.Timestamp],
    L_vals: Sequence[float],
    out_path: str,
    *,
    arrival_times: Optional[List[pd.Timestamp]] = None,
    departure_times: Optional[List[pd.Timestamp]] = None,
    with_event_marks: bool = False,
    unit: str = "timestamp",
    caption: Optional[str] = None,
) -> None:
    """Draw a standalone L(T) chart and save to file.

    This uses render_LT_chart internally, ensuring the same visual
    appearance as L(T) charts in composite layouts.
    """
    fig, ax = init_fig_ax()
    render_LT_chart(
        ax,
        times,
        L_vals,
        arrival_times=arrival_times,
        departure_times=departure_times,
        with_event_marks=with_event_marks,
    )
    format_date_axis(ax, unit=unit)
    format_fig(caption, fig)
    fig.savefig(out_path)
    plt.close(fig)


def plot_series(
    ax: Axes,
    times: Sequence[pd.Timestamp],
    values: Sequence[float],
    label: str,
    style: str = "line",
    where: str = "post",
) -> None:
    if style == "step":
        ax.step(times, values, where=where, label=label)
    else:
        ax.plot(times, values, label=label)


def draw_series_chart(
    times: Sequence[pd.Timestamp],
    values: Sequence[float],
    title: str,
    ylabel: str,
    out_path: str,
    unit: str = "timestamp",
    caption: Optional[str] = None,
    style: str = "line",
    figsize: Tuple[float, float] = (10.0, 3.4),
) -> None:
    fig, ax = init_fig_ax(figsize=figsize)
    plot_series(ax, times, values, label=ylabel, style=style)
    format_and_save(fig, ax, title, ylabel, unit, caption, out_path)


def draw_line_chart(
    times: List[pd.Timestamp],
    values: np.ndarray,
    title: str,
    ylabel: str,
    out_path: str,
    unit: str = "timestamp",
    caption: Optional[str] = None,
) -> None:
    """Draw a standalone line chart and save to file."""
    fig, ax = init_fig_ax()
    render_line_chart(ax, times, values, label=ylabel)
    format_and_save(fig, ax, title, ylabel, unit, caption, out_path)


def draw_step_chart(
    times: List[pd.Timestamp],
    values: np.ndarray,
    title: str,
    ylabel: str,
    out_path: str,
    unit: str = "timestamp",
    caption: Optional[str] = None,
    color: str = "tab:blue",
    fill: bool = False,
    overlays: Optional[List[ScatterOverlay]] = None,
) -> None:
    """Draw a standalone step chart and save to file.

    Parameters
    ----------
    overlays : optional List[ScatterOverlay]
        Scatter series to overlay on the chart. Each overlay specifies
        x/y coordinates, color, label, and whether to draw drop lines.
    """
    fig, ax = init_fig_ax()
    render_step_chart(
        ax, times, values, label=ylabel, color=color, fill=fill, overlays=overlays
    )
    format_and_save(fig, ax, title, ylabel, unit, caption, out_path)
