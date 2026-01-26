# -*- coding: utf-8 -*-
# Copyright (c) 2025 Krishna Kumar
# SPDX-License-Identifier: MIT

from __future__ import annotations

import os
from typing import List, Optional

from matplotlib import pyplot as plt
import numpy as np
import pandas as pd

from samplepath.filter import FilterResult
from samplepath.metrics import FlowMetricsResult
from samplepath.plots.helpers import (
    format_date_axis,
    render_lambda_chart,
    render_line_chart,
    render_LT_chart,
    render_N_chart,
)


def draw_five_panel_column(
    times: List[pd.Timestamp],
    N_vals: np.ndarray,
    L_vals: np.ndarray,
    Lam_vals: np.ndarray,
    w_vals: np.ndarray,
    H_vals: np.ndarray,
    title: str,
    out_path: str,
    scatter_times: Optional[List[pd.Timestamp]] = None,
    scatter_values: Optional[np.ndarray] = None,
    scatter_label: str = "Item time in system",
    lambda_pctl_upper: Optional[float] = None,
    lambda_pctl_lower: Optional[float] = None,
    lambda_warmup_hours: Optional[float] = None,
    arrival_times: Optional[List[pd.Timestamp]] = None,
    departure_times: Optional[List[pd.Timestamp]] = None,
    with_event_marks: bool = False,
) -> None:
    """Draw a 5-panel vertical stack: N(t), L(T), Λ(T), w(T), H(T)."""
    fig, axes = plt.subplots(5, 1, figsize=(12, 14), sharex=True)

    render_N_chart(
        axes[0],
        times,
        N_vals,
        arrival_times=arrival_times,
        departure_times=departure_times,
        with_event_marks=with_event_marks,
    )

    render_LT_chart(
        axes[1],
        times,
        L_vals,
        arrival_times=arrival_times,
        departure_times=departure_times,
        with_event_marks=with_event_marks,
    )

    render_lambda_chart(
        axes[2],
        times,
        Lam_vals,
        label="Λ(T) [1/hr]",
        pctl_upper=lambda_pctl_upper,
        pctl_lower=lambda_pctl_lower,
        warmup_hours=lambda_warmup_hours if lambda_warmup_hours else 0.0,
    )
    axes[2].set_title("Λ(T) — Cumulative Arrival Rate")
    axes[2].set_ylabel("Λ(T) [1/hr]")
    axes[2].legend()

    render_line_chart(axes[3], times, w_vals, label="w(T) [hrs]")
    if (
        scatter_times is not None
        and scatter_values is not None
        and len(scatter_times) > 0
    ):
        axes[3].scatter(
            scatter_times,
            scatter_values,
            s=16,
            alpha=0.6,
            marker="o",
            label=scatter_label,
        )
    axes[3].set_title("w(T) — Average Residence Time")
    axes[3].set_ylabel("w(T) [hrs]")
    axes[3].legend()

    render_line_chart(axes[4], times, H_vals, label="H(T) [hrs·items]")
    axes[4].set_title("H(T) — Cumulative Presence Mass ∫N(t)dt")
    axes[4].set_ylabel("H(T) [hrs·items]")
    axes[4].set_xlabel("Date")
    axes[4].legend()

    for ax in axes:
        format_date_axis(ax)

    fig.suptitle(title)
    plt.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out_path)
    plt.close(fig)


def draw_five_panel_column_with_scatter(
    times: List[pd.Timestamp],
    N_vals: np.ndarray,
    L_vals: np.ndarray,
    Lam_vals: np.ndarray,
    w_vals: np.ndarray,
    title: str,
    out_path: str,
    scatter_times: Optional[List[pd.Timestamp]] = None,
    scatter_values: Optional[np.ndarray] = None,
    scatter_label: str = "Item time in system",
    lambda_pctl_upper: Optional[float] = None,
    lambda_pctl_lower: Optional[float] = None,
    lambda_warmup_hours: Optional[float] = None,
    arrival_times: Optional[List[pd.Timestamp]] = None,
    departure_times: Optional[List[pd.Timestamp]] = None,
    with_event_marks: bool = False,
) -> None:
    """Draw a 5-panel stack: N(t), L(T), Λ(T), w(T) plain, w(T)+scatter."""
    fig, axes = plt.subplots(5, 1, figsize=(12, 14), sharex=True)

    render_N_chart(
        axes[0],
        times,
        N_vals,
        arrival_times=arrival_times,
        departure_times=departure_times,
        with_event_marks=with_event_marks,
    )

    render_LT_chart(
        axes[1],
        times,
        L_vals,
        arrival_times=arrival_times,
        departure_times=departure_times,
        with_event_marks=with_event_marks,
    )

    render_lambda_chart(
        axes[2],
        times,
        Lam_vals,
        label="Λ(T) [1/hr]",
        pctl_upper=lambda_pctl_upper,
        pctl_lower=lambda_pctl_lower,
        warmup_hours=lambda_warmup_hours if lambda_warmup_hours else 0.0,
    )
    axes[2].set_title("Λ(T) — Cumulative Arrival Rate")
    axes[2].set_ylabel("Λ(T) [1/hr]")
    axes[2].legend()

    render_line_chart(axes[3], times, w_vals, label="w(T) [hrs]")
    axes[3].set_title("w(T) — Average Residence Time (plain, own scale)")
    axes[3].set_ylabel("w(T) [hrs]")
    axes[3].legend()

    render_line_chart(axes[4], times, w_vals, label="w(T) [hrs]")
    if (
        scatter_times is not None
        and scatter_values is not None
        and len(scatter_values) > 0
    ):
        axes[4].scatter(
            scatter_times,
            scatter_values,
            s=16,
            alpha=0.6,
            marker="o",
            label=scatter_label,
        )
    axes[4].set_title("w(T) — with per-item durations (scatter, combined scale)")
    axes[4].set_ylabel("w(T) [hrs]")
    axes[4].set_xlabel("Date")
    axes[4].legend()

    try:
        w_min = np.nanmin(w_vals)
        w_max = np.nanmax(w_vals)
        if np.isfinite(w_min) and np.isfinite(w_max):
            pad = 0.05 * max(w_max - w_min, 1.0)
            axes[3].set_ylim(w_min - pad, w_max + pad)
        if scatter_values is not None and len(scatter_values) > 0:
            s_min = np.nanmin(scatter_values)
            s_max = np.nanmax(scatter_values)
            cmin = np.nanmin([w_min, s_min])
            cmax = np.nanmax([w_max, s_max])
        else:
            cmin, cmax = w_min, w_max
        if np.isfinite(cmin) and np.isfinite(cmax):
            pad2 = 0.05 * max(cmax - cmin, 1.0)
            axes[4].set_ylim(cmin - pad2, cmax + pad2)
    except Exception:
        pass

    for ax in axes:
        format_date_axis(ax)

    fig.suptitle(title)
    plt.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out_path)
    plt.close(fig)


def plot_five_column_stacks(df, args, filter_result, metrics, out_dir):
    t_scatter_times = df["start_ts"].tolist()
    t_scatter_vals = df["duration_hr"].to_numpy()
    written = []

    col_ts5 = os.path.join(out_dir, "misc/timestamp_stack_with_H.png")
    draw_five_panel_column(
        metrics.times,
        metrics.N,
        metrics.Lambda,
        metrics.Lambda,
        metrics.w,
        metrics.H,
        f"Finite-window metrics incl. H(T) (timestamp, {filter_result.label})",
        col_ts5,
        scatter_times=t_scatter_times,
        scatter_values=t_scatter_vals,
        lambda_pctl_upper=args.lambda_pctl,
        lambda_pctl_lower=args.lambda_lower_pctl,
        lambda_warmup_hours=args.lambda_warmup,
    )
    written.append(col_ts5)

    col_ts5s = os.path.join(out_dir, "misc/timestamp_stack_with_scatter.png")
    draw_five_panel_column_with_scatter(
        metrics.times,
        metrics.N,
        metrics.L,
        metrics.Lambda,
        metrics.w,
        f"Finite-window metrics with w(T) plain + w(T)+scatter (timestamp, {filter_result.label})",
        col_ts5s,
        scatter_times=t_scatter_times,
        scatter_values=t_scatter_vals,
        lambda_pctl_upper=args.lambda_pctl,
        lambda_pctl_lower=args.lambda_lower_pctl,
        lambda_warmup_hours=args.lambda_warmup,
    )
    written.append(col_ts5s)

    return written


def plot_misc_charts(
    df: pd.DataFrame,
    args,
    filter_result: Optional[FilterResult],
    metrics: FlowMetricsResult,
    out_dir: str,
) -> List[str]:
    # 5-panel stacks including scatter
    return plot_five_column_stacks(df, args, filter_result, metrics, out_dir)
