# -*- coding: utf-8 -*-
# Copyright (c) 2025 Krishna Kumar
# SPDX-License-Identifier: MIT
from typing import Optional, Tuple, Sequence, List

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure


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


def format_date_axis(ax: Axes, unit: str = "timestamp") -> None:
    """Format the x-axis for dates if possible."""
    ax.set_xlabel(f"Date ({unit})")
    try:
        ax.figure.autofmt_xdate()
    except Exception:
        pass


def format_axis(ax: Axes, title: str, unit: str, ylabel: str) -> None:
    """Set axis labels, title, and legend with date formatting."""
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.legend()
    format_date_axis(ax, unit=unit)


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
) -> None:
    """Format the axis, add optional caption, save the figure, and close it."""
    format_axis(ax, title, unit, ylabel)
    format_fig(caption, fig)
    fig.savefig(out_path)
    plt.close(fig)


def init_fig_ax(figsize: Tuple[float, float] = (10.0, 3.4)) -> Tuple[Figure, Axes]:
    fig, ax = plt.subplots(figsize=figsize)
    return fig, ax


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
    draw_series_chart(
        times, values, title, ylabel, out_path, unit=unit, caption=caption, style="line"
    )


def draw_step_chart(
    times: List[pd.Timestamp],
    values: np.ndarray,
    title: str,
    ylabel: str,
    out_path: str,
    unit: str = "timestamp",
    caption: Optional[str] = None,
) -> None:
    draw_series_chart(
        times, values, title, ylabel, out_path, unit=unit, caption=caption, style="step"
    )


def _clip_axis_to_percentile(ax: plt.Axes,
                             times: List[pd.Timestamp],
                             values: np.ndarray,
                             upper_p: Optional[float] = None,
                             lower_p: Optional[float] = None,
                             warmup_hours: float = 0.0) -> None:
    if upper_p is None and lower_p is None:
        return
    vals = np.asarray(values, dtype=float)
    if vals.size == 0:
        return
    mask = np.isfinite(vals)
    if warmup_hours and times:
        t0 = times[0]
        ages_hr = np.array([(t - t0).total_seconds() / 3600.0 for t in times])
        mask &= (ages_hr >= float(warmup_hours))
    data = vals[mask]
    if data.size == 0 or not np.isfinite(data).any():
        return
    top = np.nanpercentile(data, upper_p) if upper_p is not None else np.nanmax(data)
    bottom = np.nanpercentile(data, lower_p) if lower_p is not None else 0.0
    if not np.isfinite(top) or not np.isfinite(bottom) or top <= bottom:
        return
    ax.set_ylim(float(bottom), float(top))
