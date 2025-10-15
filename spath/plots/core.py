# -*- coding: utf-8 -*-
# Copyright (c) 2025 Krishna Kumar
# SPDX-License-Identifier: MIT
import os
from typing import List, Optional

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from spath.filter import FilterResult
from spath.metrics import FlowMetricsResult

from spath.plots.helpers import init_fig_ax, format_and_save, add_caption, _clip_axis_to_percentile, format_date_axis, \
    draw_step_chart, draw_line_chart


def draw_lambda_chart(
    times: List[pd.Timestamp],
    values: np.ndarray,
    title: str,
    ylabel: str,
    out_path: str,
    lambda_pctl_upper: Optional[float] = None,
    lambda_pctl_lower: Optional[float] = None,
    lambda_warmup_hours: Optional[float] = None,
    unit: str = "timestamp",
    caption: Optional[str] = None,
) -> None:
    """Line chart with optional percentile-based y-limits and warmup exclusion."""
    fig, ax = init_fig_ax(figsize=(10.0, 3.6))
    ax.plot(times, values, label=ylabel)

    # Inline percentile clipping
    vals = np.asarray(values, dtype=float)
    if vals.size > 0:
        mask = np.isfinite(vals)
        if lambda_warmup_hours and times:
            t0 = times[0]
            ages_hr = np.array([(t - t0).total_seconds() / 3600.0 for t in times])
            mask &= ages_hr >= float(lambda_warmup_hours)
        data = vals[mask]
        if data.size > 0 and np.isfinite(data).any():
            top = (
                np.nanpercentile(data, lambda_pctl_upper)
                if lambda_pctl_upper is not None
                else np.nanmax(data)
            )
            bottom = (
                np.nanpercentile(data, lambda_pctl_lower)
                if lambda_pctl_lower is not None
                else 0.0
            )
            if np.isfinite(top) and np.isfinite(bottom) and top > bottom:
                ax.set_ylim(float(bottom), float(top))

    format_and_save(fig, ax, title, ylabel, unit, caption, out_path)


def draw_L_vs_Lambda_w(
    times: List[pd.Timestamp],          # kept for symmetry with other draw_* funcs (not used here)
    L_vals: np.ndarray,
    Lambda_vals: np.ndarray,
    w_vals: np.ndarray,
    title: str,
    out_path: str,
    caption: Optional[str] = None,
) -> None:
    """
    Scatter plot of L(T) vs Λ(T)·w(T) with x=y reference line.
    All valid (finite) points should lie on the x=y line per the finite version of Little's Law
    This chart is a visual consistency check for the calculations.
    """
    # Prepare data and drop non-finite points
    x = np.asarray(L_vals, dtype=float)
    y = np.asarray(Lambda_vals, dtype=float) * np.asarray(w_vals, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]

    # Build figure (square so the x=y line is at 45°)
    fig: Figure
    ax: Axes
    fig, ax = plt.subplots(figsize=(6.0, 6.0))

    # Scatter with slightly larger markers to reveal clusters
    ax.scatter(x, y, s=18, alpha=0.7)

    # Reference x=y line across the data range with small padding
    if x.size and y.size:
        mn = float(np.nanmin([x.min(), y.min()]))
        mx = float(np.nanmax([x.max(), y.max()]))
        pad = 0.03 * (mx - mn if mx > mn else 1.0)
        lo, hi = mn - pad, mx + pad
        ax.plot([lo, hi], [lo, hi], linestyle="--")  # reference line
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)

    # Make axes comparable visually
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, linewidth=0.5, alpha=0.4)

    # Labels and title
    ax.set_xlabel("L(T)")
    ax.set_ylabel("Λ(T)·w(T)")
    ax.set_title(title)

    # Layout + optional caption (bottom)
    if caption:
        add_caption(fig, caption)  # uses the helper you already have
    fig.tight_layout(rect=(0.05, 0, 1, 1))
    fig.savefig(out_path)
    plt.close(fig)


def plot_core_sample_path_analysis_stack(args, filter_result, metrics, out_dir):
    four_col_stack = os.path.join(out_dir, 'sample_path_flow_metrics.png')
    draw_four_panel_column(metrics.times, metrics.N, metrics.L, metrics.Lambda, metrics.w,
                           f'Sample Path Flow Metrics', four_col_stack, args.lambda_pctl,
                           args.lambda_lower_pctl, args.lambda_warmup, caption=f"{filter_result.display}")
    return four_col_stack


def draw_four_panel_column(times: List[pd.Timestamp],
                           N_vals: np.ndarray,
                           L_vals: np.ndarray,
                           Lam_vals: np.ndarray,
                           w_vals: np.ndarray,
                           title: str,
                           out_path: str,
                           lambda_pctl_upper: Optional[float] = None,
                           lambda_pctl_lower: Optional[float] = None,
                           lambda_warmup_hours: Optional[float] = None,
                           caption:Optional[str] = None
                           ) -> None:
    fig, axes = plt.subplots(4, 1, figsize=(12, 11), sharex=True)

    axes[0].step(times, N_vals, where='post', label='N(t)')
    axes[0].set_title('N(t) — Sample Path')
    axes[0].set_ylabel('N(t)')
    axes[0].legend()

    axes[1].plot(times, L_vals, label='L(T)')
    axes[1].set_title('L(T) — Time-Average of N(t)')
    axes[1].set_ylabel('L(T)')
    axes[1].legend()

    axes[2].plot(times, Lam_vals, label='Λ(T) [1/hr]')
    axes[2].set_title('Λ(T) — Cumulative Arrival Rate')
    axes[2].set_ylabel('Λ(T) [1/hr]')
    axes[2].legend()
    _clip_axis_to_percentile(axes[2], times, Lam_vals,
                             upper_p=lambda_pctl_upper,
                             lower_p=lambda_pctl_lower,
                             warmup_hours=lambda_warmup_hours)

    axes[3].plot(times, w_vals, label='w(T) [hrs]')
    axes[3].set_title('w(T) — Average Residence Time')
    axes[3].set_ylabel('w(T) [hrs]')
    axes[3].set_xlabel('Date')
    axes[3].legend()

    for ax in axes:
        format_date_axis(ax)

    plt.tight_layout(rect=(0, 0, 1, 0.90))
    fig.suptitle(title, fontsize=14, y=0.97)  # larger main title
    if caption:
        fig.text(0.5, 0.945, caption,  # small gray subtitle just below title
                 ha="center", va="top")




    fig.savefig(out_path)
    plt.close(fig)


def draw_five_panel_column(times: List[pd.Timestamp],
                           N_vals: np.ndarray,
                           L_vals: np.ndarray,
                           Lam_vals: np.ndarray,
                           w_vals: np.ndarray,
                           A_vals: np.ndarray,
                           title: str,
                           out_path: str,
                           scatter_times: Optional[List[pd.Timestamp]] = None,
                           scatter_values: Optional[np.ndarray] = None,
                           scatter_label: str = "Item time in system",
                           lambda_pctl_upper: Optional[float] = None,
                           lambda_pctl_lower: Optional[float] = None,
                           lambda_warmup_hours: Optional[float] = None
                           ) -> None:
    fig, axes = plt.subplots(5, 1, figsize=(12, 14), sharex=True)

    axes[0].step(times, N_vals, where='post', label='N(t)')
    axes[0].set_title('N(t) — Sample Path')
    axes[0].set_ylabel('N(t)')
    axes[0].legend()

    axes[1].plot(times, L_vals, label='L(T)')
    axes[1].set_title('L(T) — Time-Average of N(t)')
    axes[1].set_ylabel('L(T)')
    axes[1].legend()

    axes[2].plot(times, Lam_vals, label='Λ(T) [1/hr]')
    axes[2].set_title('Λ(T) — Cumulative Arrival Rate')
    axes[2].set_ylabel('Λ(T) [1/hr]')
    axes[2].legend()
    _clip_axis_to_percentile(axes[2], times, Lam_vals,
                             upper_p=lambda_pctl_upper,
                             lower_p=lambda_pctl_lower,
                             warmup_hours=lambda_warmup_hours)

    axes[3].plot(times, w_vals, label='w(T) [hrs]')
    if scatter_times is not None and scatter_values is not None and len(scatter_times) > 0:
        axes[3].scatter(scatter_times, scatter_values, s=16, alpha=0.6, marker='o', label=scatter_label)
    axes[3].set_title('w(T) — Average Residence Time')
    axes[3].set_ylabel('w(T) [hrs]')
    axes[3].legend()

    axes[4].plot(times, A_vals, label='A(T) [hrs·items]')
    axes[4].set_title('A(T) — cumulative area ∫N(t)dt')
    axes[4].set_ylabel('A(T) [hrs·items]')
    axes[4].set_xlabel('Date')
    axes[4].legend()

    for ax in axes:
        format_date_axis(ax)

    fig.suptitle(title)
    plt.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out_path)
    plt.close(fig)


def draw_five_panel_column_with_scatter(times: List[pd.Timestamp],
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
                                        lambda_warmup_hours: Optional[float] = None
                                        ) -> None:
    fig, axes = plt.subplots(5, 1, figsize=(12, 14), sharex=True)

    axes[0].step(times, N_vals, where='post', label='N(t)')
    axes[0].set_title('N(t) — Sample Path')
    axes[0].set_ylabel('N(t)')
    axes[0].legend()

    axes[1].plot(times, L_vals, label='L(T)')
    axes[1].set_title('L(T) — Time-Average of N(t)')
    axes[1].set_ylabel('L(T)')
    axes[1].legend()

    axes[2].plot(times, Lam_vals, label='Λ(T) [1/hr]')
    axes[2].set_title('Λ(T) — Cumulative Arrival Rate')
    axes[2].set_ylabel('Λ(T) [1/hr]')
    axes[2].legend()
    _clip_axis_to_percentile(axes[2], times, Lam_vals,
                             upper_p=lambda_pctl_upper,
                             lower_p=lambda_pctl_lower,
                             warmup_hours=lambda_warmup_hours)

    axes[3].plot(times, w_vals, label='w(T) [hrs]')
    axes[3].set_title('w(T) — Average Residence Time (plain, own scale)')
    axes[3].set_ylabel('w(T) [hrs]')
    axes[3].legend()

    axes[4].plot(times, w_vals, label='w(T) [hrs]')
    if scatter_times is not None and scatter_values is not None and len(scatter_values) > 0:
        axes[4].scatter(scatter_times, scatter_values, s=16, alpha=0.6, marker='o', label=scatter_label)
    axes[4].set_title('w(T) — with per-item durations (scatter, combined scale)')
    axes[4].set_ylabel('w(T) [hrs]')
    axes[4].set_xlabel('Date')
    axes[4].legend()

    try:
        w_min = np.nanmin(w_vals); w_max = np.nanmax(w_vals)
        if np.isfinite(w_min) and np.isfinite(w_max):
            pad = 0.05 * max(w_max - w_min, 1.0)
            axes[3].set_ylim(w_min - pad, w_max + pad)
        if scatter_values is not None and len(scatter_values) > 0:
            s_min = np.nanmin(scatter_values); s_max = np.nanmax(scatter_values)
            cmin = np.nanmin([w_min, s_min]); cmax = np.nanmax([w_max, s_max])
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


def plot_core_flow_metrics_charts(
    df: pd.DataFrame,
    args,
    filter_result: Optional[FilterResult],
    metrics: FlowMetricsResult,
    out_dir: str,
) -> List[str]:
    core_panels_dir = os.path.join(out_dir, "core")
    filter_label = filter_result.label if filter_result else ""
    note = f"Filters: {filter_label}"

    path_N = os.path.join(core_panels_dir, "sample_path_N.png")
    draw_step_chart(
        metrics.times, metrics.N, "N(t) — Sample Path", "N(t)", path_N, caption=note
    )

    path_L = os.path.join(core_panels_dir, "time_average_N_L.png")
    draw_line_chart(
        metrics.times,
        metrics.L,
        "L(T) — Time-average N(t)",
        "L(T)",
        path_L,
        caption=note,
    )

    path_Lam = os.path.join(core_panels_dir, "cumulative_arrival_rate_Lambda.png")
    draw_lambda_chart(
        metrics.times,
        metrics.Lambda,
        "Λ(T) — Cumulative arrival rate",
        "Λ(T) [1/hr]",
        path_Lam,
        lambda_pctl_upper=args.lambda_pctl,
        lambda_pctl_lower=args.lambda_lower_pctl,
        lambda_warmup_hours=args.lambda_warmup,
        caption=note,
    )

    path_w = os.path.join(core_panels_dir, "average_residence_time_w.png")
    draw_line_chart(
        metrics.times,
        metrics.w,
        "w(T) — Average residence time",
        "w(T) [hrs]",
        path_w,
        caption=note,
    )

    path_invariant = os.path.join(core_panels_dir, "littles_law_invariant.png")
    draw_L_vs_Lambda_w(
        metrics.times,
        metrics.L,
        metrics.Lambda,
        metrics.w,
        title="L(T) vs Λ(T).w(T)",
        out_path=path_invariant,
        caption=note
    )
    # soujourn time scatter plot
    path_w_scatter = plot_sojourn_time_scatter(args, df, filter_result, metrics, out_dir)

    # Vertical stacks (4×1)
    path_sample_path_analysis = plot_core_sample_path_analysis_stack(args, filter_result, metrics, out_dir)
    return [path_N, path_L, path_Lam, path_w, path_invariant, path_sample_path_analysis, path_w_scatter]


def plot_sojourn_time_scatter(args, df, filter_result, metrics,out_dir) -> str:
    t_scatter_times: List[pd.Timestamp] = []
    t_scatter_vals = np.array([])
    written = []
    if args.incomplete:
        if len(metrics.times) > 0:
            t_end = metrics.times[-1]
            t_scatter_times = df["start_ts"].tolist()
            t_scatter_vals = ((t_end - df["start_ts"]).dt.total_seconds() / 3600.0).to_numpy()

    else:
        df_c = df[df["end_ts"].notna()].copy()
        if not df_c.empty:
            t_scatter_times = df_c["end_ts"].tolist()
            t_scatter_vals = df_c["duration_hr"].to_numpy()

    if len(t_scatter_times) > 0:
        ts_w_scatter = os.path.join(out_dir, "convergence/panels/residence_time_sojourn_time_scatter.png")
        label = "age" if args.incomplete else "sojourn time"
        draw_line_chart_with_scatter(metrics.times, metrics.w,
                                     f"Element {label} vs Average residence time",
                                     f"Time [hrs]", ts_w_scatter, t_scatter_times, t_scatter_vals, scatter_label=f"element {label}",
                                      caption=f"{filter_result.label}")



    return ts_w_scatter


def draw_line_chart_with_scatter(times: List[pd.Timestamp],
                                 values: np.ndarray,
                                 title: str,
                                 ylabel: str,
                                 out_path: str,
                                 scatter_times: List[pd.Timestamp],
                                 scatter_values: np.ndarray,
                                 line_label: str = 'Average Residence Time',
                                 scatter_label: str = "element sojourn time",
                                 unit: str = "timestamp",
                                 caption: Optional[str] = None
                                 ) -> None:
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(times, values, label=line_label)
    if scatter_times is not None and scatter_values is not None and len(scatter_times) > 0:
        ax.scatter(scatter_times, scatter_values, s=16, alpha=0.6, marker='o', label=scatter_label)

    format_and_save(fig, ax, title, ylabel, unit, caption, out_path)
