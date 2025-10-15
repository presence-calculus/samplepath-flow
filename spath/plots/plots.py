# -*- coding: utf-8 -*-
# Copyright (c) 2025 Krishna Kumar
# SPDX-License-Identifier: MIT
import os
from typing import List, Optional, Tuple, Sequence

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from spath.file_utils import ensure_output_dirs
from spath.filter import FilterResult
from spath.metrics import FlowMetricsResult, compute_elementwise_empirical_metrics, compute_tracking_errors, \
    compute_coherence_score, compute_end_effect_series, compute_total_active_age_series, ElementWiseEmpiricalMetrics


def _add_caption(fig: Figure, text: str) -> None:
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


def _format_date_axis(ax: Axes, unit: str = "timestamp") -> None:
    """Format the x-axis for dates if possible."""
    ax.set_xlabel(f"Date ({unit})")
    try:
        ax.figure.autofmt_xdate()
    except Exception:
        pass


def _format_axis(ax: Axes, title: str, unit: str, ylabel: str) -> None:
    """Set axis labels, title, and legend with date formatting."""
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.legend()
    _format_date_axis(ax, unit=unit)


def _format_fig(caption: Optional[str], fig: Figure) -> None:
    """Finalize figure with optional caption and layout adjustment."""
    fig.tight_layout()
    if caption:
        _add_caption(fig, caption)


def _format_and_save(
    fig: Figure,
    ax: Axes,
    title: str,
    ylabel: str,
    unit: str,
    caption: Optional[str],
    out_path: str,
) -> None:
    """Format the axis, add optional caption, save the figure, and close it."""
    _format_axis(ax, title, unit, ylabel)
    _format_fig(caption, fig)
    fig.savefig(out_path)
    plt.close(fig)


# ── Common plotting engines (de-duplicated) ───────────────────────────────────


def _init_fig_ax(figsize: Tuple[float, float] = (10.0, 3.4)) -> Tuple[Figure, Axes]:
    fig, ax = plt.subplots(figsize=figsize)
    return fig, ax


def _plot_series(
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
    fig, ax = _init_fig_ax(figsize=figsize)
    _plot_series(ax, times, values, label=ylabel, style=style)
    _format_and_save(fig, ax, title, ylabel, unit, caption, out_path)


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
    fig, ax = _init_fig_ax(figsize=(10.0, 3.6))
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

    _format_and_save(fig, ax, title, ylabel, unit, caption, out_path)


# ── Higher-level plotting functions (unchanged except captions fixed) ─────────


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

def plot_convergence_charts(
    df: pd.DataFrame,
    args,
    filter_result: Optional[FilterResult],
    metrics: FlowMetricsResult,
    empirical_metrics: ElementWiseEmpiricalMetrics,
    out_dir: str,
) -> List[str]:
    written = []

    written += plot_arrival_rate_convergence(args, filter_result, metrics, empirical_metrics, out_dir)

    written += plot_residence_time_sojourn_time_coherence_charts(df, args, filter_result, metrics, out_dir)

    written += plot_residence_vs_sojourn_stack(df, args, filter_result, metrics, out_dir)

    written += plot_sample_path_convergence(df, args, filter_result, metrics, out_dir)

    return written

def plot_stability_charts(
    df: pd.DataFrame,
    args,
    filter_result: Optional[FilterResult],
    metrics: FlowMetricsResult,
    out_dir: str,
) -> List[str]:
    written = []
    written += plot_rate_stability_charts(df, args, filter_result, metrics, out_dir)
    return written

def plot_advanced_charts(
    df: pd.DataFrame,
    args,
    filter_result: Optional[FilterResult],
    metrics: FlowMetricsResult,
    out_dir: str,
) -> List[str]:
    written = []
    written += plot_llaw_manifold_3d(df, metrics, out_dir)
    return written

def plot_misc_charts(df: pd.DataFrame,
    args,
    filter_result: Optional[FilterResult],
    metrics: FlowMetricsResult,
    out_dir: str,
) -> List[str]:
    # 5-panel stacks including scatter
    return  plot_five_column_stacks(df, args, filter_result, metrics, out_dir)

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

    _format_and_save(fig, ax, title, ylabel, unit, caption, out_path)



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
        _add_caption(fig, caption)  # uses the helper you already have
    fig.tight_layout(rect=(0.05, 0, 1, 1))
    fig.savefig(out_path)
    plt.close(fig)



def draw_residence_time_convergence_panel(times: List[pd.Timestamp],
                                          w_vals: np.ndarray,
                                          W_star: np.ndarray,
                                          title: str,
                                          out_path: str,
                                          caption: Optional[str] = None
                                          ) -> None:
    fig, ax = plt.subplots(figsize=(12, 6.5))

    ax.plot(times, w_vals, label='w(T) [hrs]')
    ax.plot(times, W_star, linestyle='--', label='W*(t) [hrs] (completed ≤ t)')
    ax.set_title('w(T) vs W*(t)')
    ax.set_ylabel('hours')
    ax.legend()

    _format_date_axis(ax)

    fig.suptitle(title)
    if caption:
        _add_caption(fig, caption)  # uses the helper
    fig.tight_layout(rect=(0, 0, 1, 1))

    fig.savefig(out_path)
    plt.close(fig)


def draw_cumulative_arrival_rate_convergence_panel(times: List[pd.Timestamp],
                                   w_vals: np.ndarray,
                                   lam_vals: np.ndarray,
                                   W_star: np.ndarray,
                                   lam_star: np.ndarray,
                                   title: str,
                                   out_path: str,
                                   lambda_pctl_upper: Optional[float] = None,
                                   lambda_pctl_lower: Optional[float] = None,
                                   lambda_warmup_hours: Optional[float] = None,
                                   caption: Optional[str] = None
                                   ) -> None:
    fig, ax = plt.subplots(figsize=(12, 6.5))

    ax.plot(times, lam_vals, label='Λ(T) [1/hr]')
    ax.plot(times, lam_star, linestyle='--', label='λ*(t) [1/hr] (arrivals ≤ t)')
    ax.set_title('Λ(T) vs λ*(t)  — arrival rate')
    ax.set_ylabel('1/hr')
    ax.set_xlabel('Date')
    ax.legend()

    _clip_axis_to_percentile(ax, times, lam_vals,
                             upper_p=lambda_pctl_upper,
                             lower_p=lambda_pctl_lower,
                             warmup_hours=lambda_warmup_hours)

    _format_date_axis(ax)

    fig.suptitle(title)
    if caption:
        _add_caption(fig, caption)  # uses the helper you already have
    fig.tight_layout(rect=(0, 0, 1, 1))

    fig.savefig(out_path)
    plt.close(fig)


def draw_dynamic_convergence_panel_with_errors(times: List[pd.Timestamp],
                                               w_vals: np.ndarray,
                                               lam_vals: np.ndarray,
                                               W_star: np.ndarray,
                                               lam_star: np.ndarray,
                                               eW: np.ndarray,
                                               eLam: np.ndarray,
                                               epsilon: Optional[float],
                                               title: str,
                                               out_path: str,
                                               lambda_pctl_upper: Optional[float] = None,
                                               lambda_pctl_lower: Optional[float] = None,
                                               lambda_warmup_hours: Optional[float] = None
                                               ) -> None:
    fig, axes = plt.subplots(3, 1, figsize=(12, 9.2), sharex=True)

    axes[0].plot(times, w_vals, label='w(T) [hrs]')
    axes[0].plot(times, W_star, linestyle='--', label='W*(t) [hrs] (completed ≤ t)')
    axes[0].set_title('w(T) vs W*(t) — dynamic')
    axes[0].set_ylabel('hours')
    axes[0].legend()

    axes[1].plot(times, lam_vals, label='Λ(T) [1/hr]')
    axes[1].plot(times, lam_star, linestyle='--', label='λ*(t) [1/hr] (arrivals ≤ t)')
    axes[1].set_title('Λ(T) vs λ*(t) — dynamic')
    axes[1].set_ylabel('1/hr')
    axes[1].legend()
    _clip_axis_to_percentile(axes[1], times, lam_vals,
                             upper_p=lambda_pctl_upper,
                             lower_p=lambda_pctl_lower,
                             warmup_hours=lambda_warmup_hours)

    axes[2].plot(times, eW, label='rel. error e_W')
    axes[2].plot(times, eLam, label='rel. error e_λ')
    if epsilon is not None:
        axes[2].axhline(epsilon, linestyle='--', label=f'ε = {epsilon:g}')
    axes[2].set_title('Relative tracking errors')
    axes[2].set_ylabel('relative error')
    axes[2].set_xlabel('Date')
    axes[2].legend()

    err = np.concatenate([eW[np.isfinite(eW)], eLam[np.isfinite(eLam)]])
    if err.size > 0:
        ub = float(np.nanpercentile(err, 99.5))
        axes[2].set_ylim(0.0, max(ub, (epsilon if epsilon is not None else 0.0) * 1.5 + 1e-6))

    for ax in axes:
        _format_date_axis(ax)

    fig.suptitle(title)
    plt.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out_path)
    plt.close(fig)


def draw_dynamic_convergence_panel_with_errors_and_endeffects(times: List[pd.Timestamp],
                                                              w_vals: np.ndarray,
                                                              lam_vals: np.ndarray,
                                                              W_star: np.ndarray,
                                                              lam_star: np.ndarray,
                                                              eW: np.ndarray,
                                                              eLam: np.ndarray,
                                                              rA: np.ndarray,
                                                              rB: np.ndarray,
                                                              rho: np.ndarray,
                                                              epsilon: Optional[float],
                                                              title: str,
                                                              out_path: str,
                                                              lambda_pctl_upper: Optional[float] = None,
                                                              lambda_pctl_lower: Optional[float] = None,
                                                              lambda_warmup_hours: Optional[float] = None
                                                              ) -> None:
    """Four-row dynamic convergence view with end-effect metrics."""
    fig, axes = plt.subplots(4, 1, figsize=(12, 12), sharex=True)

    axes[0].plot(times, w_vals, label='w(T) [hrs]')
    axes[0].plot(times, W_star, linestyle='--', label='W*(t) [hrs] (completed ≤ t)')
    axes[0].set_title('w(T) vs W*(t) — dynamic')
    axes[0].set_ylabel('hours')
    axes[0].legend()

    axes[1].plot(times, lam_vals, label='Λ(T) [1/hr]')
    axes[1].plot(times, lam_star, linestyle='--', label='λ*(t) [1/hr] (arrivals ≤ t)')
    axes[1].set_title('Λ(T) vs λ*(t) — dynamic')
    axes[1].set_ylabel('1/hr')
    axes[1].legend()
    _clip_axis_to_percentile(axes[1], times, lam_vals,
                             upper_p=lambda_pctl_upper,
                             lower_p=lambda_pctl_lower,
                             warmup_hours=lambda_warmup_hours)

    axes[2].plot(times, eW, label='rel. error e_W')
    axes[2].plot(times, eLam, label='rel. error e_λ')
    if epsilon is not None:
        axes[2].axhline(epsilon, linestyle='--', label=f'ε = {epsilon:g}')
    axes[2].set_title('Relative tracking errors')
    axes[2].set_ylabel('relative error')
    axes[2].legend()

    err = np.concatenate([eW[np.isfinite(eW)], eLam[np.isfinite(eLam)]])
    if err.size > 0:
        ub = float(np.nanpercentile(err, 99.5))
        axes[2].set_ylim(0.0, max(ub, (epsilon if epsilon is not None else 0.0) * 1.5 + 1e-6))

    axes[3].plot(times, rA, label='r_A(T) = E/A', alpha=0.9)
    axes[3].plot(times, rB, label='r_B(T) = B/starts', alpha=0.9)
    axes[3].set_title('End-effects: mass share and boundary share')
    axes[3].set_ylabel('share [0–1]')
    axes[3].set_ylim(0.0, 1.0)
    ax2 = axes[3].twinx()
    ax2.plot(times, rho, linestyle='--', label='ρ(T)=T/W*(t)', alpha=0.7)
    ax2.set_ylabel('ρ (window / duration)')
    lines1, labels1 = axes[3].get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    axes[3].legend(lines1 + lines2, labels1 + labels2, loc='upper right')

    for ax in axes:
        _format_date_axis(ax)

    fig.suptitle(title)
    plt.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out_path)
    plt.close(fig)


# ── Arrival departure convergence: two-panel stack ───────────────────

def draw_arrival_departure_convergence_stack(
    times: List[pd.Timestamp],
    arrivals_cum: np.ndarray,           # A(t)  = metrics.Arrivals
    departures_cum: np.ndarray,         # D(t)  = metrics.Departures
    lambda_cum_rate: np.ndarray,        # Λ(T)  = metrics.Lambda [1/hr]
    title: str,
    out_path: str,
    lambda_pctl_upper: Optional[float] = None,
    lambda_pctl_lower: Optional[float] = None,
    lambda_warmup_hours: Optional[float] = None,
    caption: Optional[str] = None,
) -> None:
    """
    Two stacked charts sharing the x-axis:

      (1) Cumulative Arrivals vs Cumulative Departures
          Stability: A(t) and D(t) grow together; gap A-D = N(t) stays bounded.

      (2) Λ(T) vs θ(T)  (arrival rate vs throughput rate)
          θ(T) := D(T) / (T - t0) [1/hr], masked after the last departure to avoid the
          idle-tail artifact where the ratio would decay toward 0.
    """
    # ---- Compute elapsed hours and throughput rate θ(T) ----------------------
    n = len(times)
    if n > 0:
        t0 = times[0]
        elapsed_h = np.array([(t - t0).total_seconds() / 3600.0 for t in times], dtype=float)
    else:
        elapsed_h = np.array([], dtype=float)

    with np.errstate(divide="ignore", invalid="ignore"):
        theta_rate = np.where(elapsed_h > 0.0, departures_cum / elapsed_h, np.nan)

    # Find last departure index (last time D(t) increases)
    last_dep_idx = -1
    if len(departures_cum) > 0:
        d = np.asarray(departures_cum, dtype=float)
        inc = np.flatnonzero(np.diff(d, prepend=d[0]) > 0)
        if inc.size > 0:
            last_dep_idx = int(inc.max())

    # Mask θ(T) after the last departure (prevents idle tail from misleading viewers)
    if last_dep_idx >= 0 and last_dep_idx + 1 < n:
        theta_rate[last_dep_idx + 1:] = np.nan

    # ---- Figure ----------------------------------------------------------------
    fig, axes = plt.subplots(2, 1, figsize=(12, 6.5), sharex=True)

    # Panel 1: cumulative counts (step plots)
    axes[0].step(times, arrivals_cum, where='post', label='A(t): cumulative arrivals')
    axes[0].step(times, departures_cum, where='post', label='D(t): cumulative departures')
    axes[0].set_title('Cumulative Arrivals vs Cumulative Departures')
    axes[0].set_ylabel('count')
    axes[0].legend(loc='best')
    _format_date_axis(axes[0])

    # Panel 2: rates (Λ vs θ), with house percentile clipping on Λ
    axes[1].plot(times, lambda_cum_rate, label='Λ(T) [1/hr]')
    axes[1].plot(times, theta_rate, label='θ(T) = D(T)/elapsed [1/hr]')
    axes[1].set_title('Arrival Rate Λ(T) vs Departure Rate θ(T)')
    axes[1].set_ylabel('1/hr')
    axes[1].legend(loc='best')
    _clip_axis_to_percentile(
        axes[1], times, lambda_cum_rate,
        upper_p=lambda_pctl_upper,
        lower_p=lambda_pctl_lower,
        warmup_hours=(lambda_warmup_hours or 0.0),
    )
    _format_date_axis(axes[1])

    fig.suptitle(title)
    if caption:
        fig.text(0.5, 0.01, caption, ha='center', va='bottom', fontsize=9)
        plt.tight_layout(rect=(0, 0.03, 1, 0.96))
    else:
        plt.tight_layout(rect=(0, 0, 1, 0.96))

    fig.savefig(out_path)
    plt.close(fig)


def plot_arrival_rate_convergence(
    args,
    filter_result: Optional[FilterResult],
    metrics: FlowMetricsResult,
    empirical_metrics: ElementWiseEmpiricalMetrics,
    out_dir: str,
) -> List[str]:
    """
    Inputs expected from FlowMetricsResult:
      - metrics.times                : List[pd.Timestamp]
      - metrics.Arrivals             : cumulative arrivals A(t)
      - metrics.Departures           : cumulative departures D(t)
      - metrics.Lambda               : Cumulative Arrival Rate Λ(T) [1/hr]
    Returns list of written image paths.
    """

    caption = (filter_result.display if filter_result else None)

    pctl_upper = getattr(args, "lambda_pctl", None)
    pctl_lower = getattr(args, "lambda_lower_pctl", None)
    warmup_hrs = getattr(args, "lambda_warmup", None)

    eq_path = os.path.join(out_dir, "convergence/arrival_departure_equilibrium.png")
    draw_arrival_departure_convergence_stack(
        metrics.times,
        metrics.Arrivals,
        metrics.Departures,
        metrics.Lambda,
        title="Flow Equilibrium: Arrival/Departure Convergence",
        out_path=eq_path,
        lambda_pctl_upper=pctl_upper,
        lambda_pctl_lower=pctl_lower,
        lambda_warmup_hours=warmup_hrs,
        caption=caption,
    )
    lambda_path = os.path.join(out_dir, "convergence/panels/arrival_rate_convergence.png")
    draw_cumulative_arrival_rate_convergence_panel(
        metrics.times,
        metrics.w,
        metrics.Lambda,
        empirical_metrics.W_star,
        empirical_metrics.lam_star,
        title="Flow Equilibrium: Arrival/Departure Convergence",
        out_path=lambda_path,
        lambda_pctl_upper=pctl_upper,
        lambda_pctl_lower=pctl_lower,
        lambda_warmup_hours=warmup_hrs,
        caption=caption,
    )
    return [eq_path, lambda_path]

# ── Residence vs Sojourn: two-panel stack ────────────────────────────────────

def draw_residence_vs_sojourn_stack(
    times: List[pd.Timestamp],
    w_series_hours: np.ndarray,         # w(T) aligned to `times` (avg residence time, hours)
    df: pd.DataFrame,                   # original events with start_ts / end_ts
    title: str,
    out_path: str,
    caption: Optional[str] = None,
) -> None:
    """
    Top panel:  w(T) vs W*(t)  — Average Residence Time vs empirical average sojourn time
    Bottom:     Sojourn-time scatter vs w(T)

    Assumptions:
      • df has 'start_ts' and 'end_ts' columns (tz-aware OK).
      • W*(t) is computed via your existing helper:
          compute_dynamic_empirical_series(df, times) -> (W_star, lam_star)
        We use only W_star here.
      • w_series_hours is aligned to `times` and in HOURS.
    """
    # --- Compute W*(t) aligned to `times`
    if len(times) > 0:
        W_star_hours, _ = compute_elementwise_empirical_metrics(df, times).as_tuple()
    else:
        W_star_hours = np.array([])

    # --- Build scatter (completed items only)
    df_c = df[df["end_ts"].notna()].copy()
    if not df_c.empty:
        soj_times = df_c["end_ts"].tolist()
        soj_vals_h = ((df_c["end_ts"] - df_c["start_ts"]).dt.total_seconds() / 3600.0).to_numpy()
    else:
        soj_times, soj_vals_h = [], np.array([])

    # --- Figure
    fig, axes = plt.subplots(2, 1, figsize=(12, 6.5), sharex=True)

    # Panel 1: w(T) vs W*(t)
    axes[0].plot(times, w_series_hours, label='w(T) [hrs]')
    axes[0].plot(times, W_star_hours, linestyle='--', label='W*(t) [hrs] (completed ≤ t)')
    axes[0].set_title('w(T) vs W*(t) — residence vs sojourn')
    axes[0].set_ylabel('hours')
    axes[0].legend(loc='best')
    _format_date_axis(axes[0])

    # Panel 2: Sojourn scatter vs w(T)
    if len(soj_times) > 0:
        axes[1].scatter(soj_times, soj_vals_h, s=18, alpha=0.55, label='element sojourn time')
    axes[1].plot(times, w_series_hours, label='Average Residence Time', zorder=3)
    axes[1].set_title('Element sojourn time vs Average residence time')
    axes[1].set_ylabel('Time [hrs]')
    axes[1].legend(loc='best')
    _format_date_axis(axes[1])

    fig.suptitle(title)
    if caption:
        fig.text(0.5, 0.01, caption, ha='center', va='bottom', fontsize=9)
        plt.tight_layout(rect=(0, 0.03, 1, 0.96))
    else:
        plt.tight_layout(rect=(0, 0, 1, 0.96))

    fig.savefig(out_path)
    plt.close(fig)


def plot_residence_vs_sojourn_stack(
    df: pd.DataFrame,
    args,
    filter_result: Optional[FilterResult],
    metrics: FlowMetricsResult,
    out_dir: str,
) -> List[str]:
    """
    Orchestrator mirroring your other plot_* wrappers.

    Expects from FlowMetricsResult:
      • metrics.times              : List[pd.Timestamp]
      • metrics.w                  : np.ndarray (Average Residence Time series in HOURS)

    Uses compute_dynamic_empirical_series(df, metrics.times) for W*(t).
    Writes: timestamp_residence_vs_sojourn_stack.png
    """
    
    caption = (filter_result.display if filter_result else None)

    out_path = os.path.join(out_dir, "convergence/residence_sojourn_coherence.png")
    draw_residence_vs_sojourn_stack(
        metrics.times,
        metrics.w,            # w(T) [hrs] aligned to times
        df,
        title="Flow Coherence: Residence Time/Sojourn Time Convergence",
        out_path=out_path,
        caption=caption,
    )
    return [out_path]

# ── Little's Law scatter coherence: L(T) vs λ*(t)·W*(t) ──────────────────────

def draw_ll_scatter_coherence(
    L_vals: np.ndarray,                 # metrics.L (avg number-in-system)
    lam_star: np.ndarray,               # λ*(t) [1/hr] aligned to metrics.times
    W_star_hours: np.ndarray,           # W*(t) [hrs] aligned to metrics.times
    times: List[pd.Timestamp],          # metrics.times
    epsilon: float,                     # tolerance for "within band"
    out_png: str,
    title: str = "Sample Path Coherence",
    caption: Optional[str] = None,
    horizon_days: float = 0.0,         # require elapsed >= horizon_days
) -> Tuple[float, int, int]:
    """
    Scatter points x=L(T) vs y=λ*(t)·W*(t), draw x=y and an ε relative band:
        (1-ε)·x  <=  y  <=  (1+ε)·x
    Return (score, ok_count, total_count) using only points with elapsed >= horizon_hours.

    Notes:
      • Relative band makes the tolerance scale with magnitude.
      • Also robust to zeros by ignoring points where x <= 0 or y is NaN/inf.
    """
    # Build y = λ*(t)·W*(t). W* is in HOURS, λ* in 1/hr so product is dimensionless (matches L).
    y_vals = lam_star * W_star_hours
    x_vals = np.asarray(L_vals, dtype=float)

    n = len(times)
    if n > 0:
        t0 = times[0]
        elapsed_h = np.array([(t - t0).total_seconds() / 3600.0 for t in times], dtype=float)
    else:
        elapsed_h = np.array([], dtype=float)

    # Mask: finite and past horizon
    finite_mask = np.isfinite(x_vals) & np.isfinite(y_vals) & (x_vals > 0.0)
    horizon_hours = float(horizon_days) * 24.0
    if horizon_hours and horizon_hours > 0.0:
        finite_mask &= elapsed_h >= float(horizon_hours)

    X = x_vals[finite_mask]
    Y = y_vals[finite_mask]

    # Coherence test: inside relative epsilon tube around x=y
    # Equivalent to |Y/X - 1| <= ε  (avoid division-by-zero via x>0 mask above)
    rel_err = np.abs(Y / X - 1.0)
    ok_mask = rel_err <= float(epsilon)
    ok_count = int(np.count_nonzero(ok_mask))
    total_count = int(X.size)
    score = (ok_count / total_count) if total_count > 0 else float("nan")

    # ---- Plot
    fig, ax = _init_fig_ax(figsize=(7.5, 6.0))

    # Scatter (only evaluated points)
    ax.scatter(X, Y, s=16, alpha=0.7, label='Points: (L(T), λ*(t)·W*(t))')

    # Diagonal x=y
    if X.size > 0:
        x_min, x_max = float(np.nanmin(X)), float(np.nanmax(X))
        pad = 0.05 * (x_max - x_min if x_max > x_min else 1.0)
        x_line = np.linspace(max(0.0, x_min - pad), x_max + pad, 200)
        ax.plot(x_line, x_line, linewidth=1.2, label='x = y')

        # ε-band: (1-ε)x .. (1+ε)x
        ax.plot(x_line, (1.0 + float(epsilon)) * x_line, linewidth=0.8, linestyle='--', label=f'y=(1+ε)x')
        ax.plot(x_line, (1.0 - float(epsilon)) * x_line, linewidth=0.8, linestyle='--', label=f'y=(1−ε)x')
        ax.fill_between(
            x_line,
            (1.0 - float(epsilon)) * x_line,
            (1.0 + float(epsilon)) * x_line,
            alpha=0.08
        )

    ax.set_xlabel('L(T)  (average number in system)')
    ax.set_ylabel('λ*(t)·W*(t)')
    ax.set_title(title)
    ax.legend(loc='best')

    # annotate score
    if total_count > 0:
        ax.text(
            0.02, 0.98,
            f'ε={epsilon:.3g}, horizon={horizon_days:.1f}d:  {ok_count}/{total_count}  ({score*100:.1f}%)',
            ha='left', va='top', transform=ax.transAxes, fontsize=9
        )

    if caption:
        fig.text(0.5, 0.01, caption, ha='center', va='bottom', fontsize=9)
        plt.tight_layout(rect=(0, 0.03, 1, 0.96))
    else:
        plt.tight_layout()

    fig.savefig(out_png)
    plt.close(fig)

    return score, ok_count, total_count


def plot_sample_path_convergence(
    df: pd.DataFrame,
    args,
    filter_result: Optional[FilterResult],
    metrics: FlowMetricsResult,
    out_dir: str,
) -> List[str]:
    """

    Uses:
      • metrics.L                : L(T) series (dimensionless)
      • metrics.times            : timestamps
      • compute_dynamic_empirical_series(df, metrics.times) -> (W*, λ*)
        - W* in HOURS, λ* in 1/hr

    CLI-style knobs (optional in args):
      • args.epsilon   : default 0.05
      • args.horizon_day : horizon in days (default 28)

    Outputs:
      • PNG: timestamp_sample_path_coherence.png

    """

    caption = (filter_result.display if filter_result else None)

    # derive W*(t), λ*(t) aligned to times
    if len(metrics.times) > 0:
        W_star_hours, lam_star = compute_elementwise_empirical_metrics(df, metrics.times).as_tuple()
    else:
        W_star_hours = np.array([])
        lam_star = np.array([])

    epsilon = getattr(args, "epsilon", 0.05)
    horizon_days = getattr(args, "horizon_days", 28)


    png_path = os.path.join(out_dir, "sample_path_convergence.png")
    score, ok_count, total_count = draw_ll_scatter_coherence(
        metrics.L,
        lam_star,
        W_star_hours,
        metrics.times,
        epsilon=epsilon,
        out_png=png_path,
        title="Sample Path Convergence: L(T) vs λ*(t)·W*(t)",
        caption=caption,
        horizon_days=horizon_days,
    )

    # wri
    if np.isnan(score):
        print(f"Sample Path Convergence: ε={epsilon}, H={horizon_days}d -> n/a (no valid points)\n")
    else:
        print(f"Sample Path Convergence: ε={epsilon}, H={horizon_days}d -> "
                f"{ok_count}/{total_count} ({score*100:.1f}%)\n")
    return [png_path]

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
        _format_date_axis(ax)

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
        _format_date_axis(ax)

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
        _format_date_axis(ax)

    fig.suptitle(title)
    plt.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out_path)
    plt.close(fig)


def plot_residence_time_sojourn_time_coherence_charts(df, args, filter_result, metrics, out_dir):
    # Empirical targets & dynamic baselines
    horizon_days = args.horizon_days
    epsilon = args.epsilon
    lambda_pctl_upper = args.lambda_pctl
    lambda_pctl_lower = args.lambda_lower_pctl
    lambda_warmup_hours = args.lambda_warmup
    mode_label = filter_result.label

    written: List[str] = []

    if len(metrics.times) > 0:
        W_star_ts, lam_star_ts = compute_elementwise_empirical_metrics(df, metrics.times).as_tuple()
    else:
        W_star_ts = np.array([])
        lam_star_ts = np.array([])
    # Relative errors & coherence
    eW_ts, eLam_ts, elapsed_ts = compute_tracking_errors(metrics.times, metrics.w, metrics.Lambda, W_star_ts,
                                                         lam_star_ts)
    coh_summary_lines: List[str] = []
    if epsilon is not None and horizon_days is not None:
        h_hrs = float(horizon_days) * 24.0
        sc_ts, ok_ts, tot_ts = compute_coherence_score(eW_ts, eLam_ts, elapsed_ts, float(epsilon), h_hrs)
        coh_summary_lines.append(
            f"Coherence (timestamp): eps={epsilon:g}, H={horizon_days:g}d -> {ok_ts}/{tot_ts} ({(sc_ts * 100 if sc_ts == sc_ts else 0):.1f}%)")
    # Convergence diagnostics (timestamp)
    if len(metrics.times) > 0:
        ts_conv_dyn = os.path.join(out_dir, 'convergence/panels/residence_time_convergence.png')
        draw_residence_time_convergence_panel(metrics.times, metrics.w, W_star_ts,
                                               title=f"Residence Time Convergence", out_path=ts_conv_dyn,
                                              caption=filter_result.display)
        written.append(ts_conv_dyn)

        ts_conv_dyn3 = os.path.join(out_dir, 'advanced/residence_convergence_errors.png')
        draw_dynamic_convergence_panel_with_errors(metrics.times, metrics.w, metrics.Lambda, W_star_ts, lam_star_ts,
                                                   eW_ts, eLam_ts, epsilon,
                                                   f'Residence time convergence + errors (timestamp, {mode_label})',
                                                   ts_conv_dyn3, lambda_pctl_upper=lambda_pctl_upper,
                                                   lambda_pctl_lower=lambda_pctl_lower,
                                                   lambda_warmup_hours=lambda_warmup_hours)
        written.append(ts_conv_dyn3)
    # --- End-effect diagnostics ---
    rA_ts, rB_ts, rho_ts = compute_end_effect_series(df, metrics.times, metrics.A, W_star_ts) if len(
        metrics.times) > 0 else (np.array([]), np.array([]), np.array([]))
    if len(metrics.times) > 0:
        ts_conv_dyn4 = os.path.join(out_dir, 'advanced/residence_time_convergence_errors_endeffects.png')
        draw_dynamic_convergence_panel_with_errors_and_endeffects(
            metrics.times, metrics.w, metrics.Lambda, W_star_ts, lam_star_ts, eW_ts, eLam_ts,
            rA_ts, rB_ts, rho_ts, epsilon,
            f'Residence time convergence + errors + end-effects (timestamp, {mode_label})', ts_conv_dyn4,
            lambda_pctl_upper=lambda_pctl_upper, lambda_pctl_lower=lambda_pctl_lower,
            lambda_warmup_hours=lambda_warmup_hours)
        written.append(ts_conv_dyn4)

    return written


def plot_core_sample_path_analysis_stack(args, filter_result, metrics, out_dir):
    four_col_stack = os.path.join(out_dir, 'sample_path_flow_metrics.png')
    draw_four_panel_column(metrics.times, metrics.N, metrics.L, metrics.Lambda, metrics.w,
                           f'Sample Path Flow Metrics', four_col_stack, args.lambda_pctl,
                           args.lambda_lower_pctl, args.lambda_warmup, caption=f"{filter_result.display}")
    return four_col_stack


def plot_five_column_stacks(df, args, filter_result, metrics, out_dir):
    t_scatter_times = df["start_ts"].tolist()
    t_scatter_vals = df["duration_hr"].to_numpy()
    written = []

    col_ts5 = os.path.join(out_dir, 'misc/timestamp_stack_with_A.png')
    draw_five_panel_column(metrics.times, metrics.N, metrics.Lambda, metrics.Lambda, metrics.w, metrics.A,
                           f'Finite-window metrics incl. A(T) (timestamp, {filter_result.label})', col_ts5,
                           scatter_times=t_scatter_times, scatter_values=t_scatter_vals,
                           lambda_pctl_upper=args.lambda_pctl, lambda_pctl_lower=args.lambda_lower_pctl,
                           lambda_warmup_hours=args.lambda_warmup)
    written.append(col_ts5)


    col_ts5s = os.path.join(out_dir, 'misc/timestamp_stack_with_scatter.png')
    draw_five_panel_column_with_scatter(metrics.times, metrics.N, metrics.L, metrics.Lambda, metrics.w,
                                        f'Finite-window metrics with w(T) plain + w(T)+scatter (timestamp, {filter_result.label})',
                                        col_ts5s,
                                        scatter_times=t_scatter_times, scatter_values=t_scatter_vals,
                                        lambda_pctl_upper=args.lambda_pctl,
                                        lambda_pctl_lower=args.lambda_lower_pctl,
                                        lambda_warmup_hours=args.lambda_warmup)
    written.append(col_ts5s)

    return written

def plot_rate_stability_charts(
    df: pd.DataFrame,
    args,                 # kept for signature consistency
    filter_result,        # may provide .title_prefix and .display
    metrics,              # FlowMetricsResult with .times, .N, .t0, .w
    out_dir: str,
) -> List[str]:
    """
    Produce:
      - timestamp_rate_stability_n.png          (N(T)/T)
      - timestamp_rate_stability_r.png          (R(T)/T)
      - timestamp_rate_stability_stack.png      (4-row stack: N/T, R/T, λ*(T), W-coherence)

    The stacked figure has suptitle "Equilibrium and Coherence" and a caption with the filter display.
    """
    written: List[str] = []

    # Observation grid
    times = [pd.Timestamp(t) for t in metrics.times]
    if not times:
        return written

    # Elapsed hours since t0
    t0 = metrics.t0 if hasattr(metrics, "t0") and pd.notna(metrics.t0) else times[0]
    elapsed_h = np.array([(t - t0).total_seconds() / 3600.0 for t in times], dtype=float)
    denom = np.where(elapsed_h > 0.0, elapsed_h, np.nan)

    # Core rate series
    N_raw = np.asarray(metrics.N, dtype=float)
    R_raw = compute_total_active_age_series(df, times)  # hours

    with np.errstate(divide="ignore", invalid="ignore"):
        N_over_T = N_raw / denom
        R_over_T = R_raw / denom

    # Dynamic empirical series (for λ* and W*)
    W_star_ts, lam_star_ts = compute_elementwise_empirical_metrics(df, times).as_tuple()
    w_ts = np.asarray(metrics.w, dtype=float)

    # Optional display bits
    title_prefix = getattr(filter_result, "title_prefix", None)
    caption_text = getattr(filter_result, "display", None)

    # --------------------- Chart 1: N(t) sample path + N(T)/T ---------------------
    out_path_N = os.path.join(out_dir, "stability/panels/wip_growth_rate.png")
    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(10, 7.8), sharex=True)

    # Top: N(t) sample path (step plot)
    ax_top = axes[0]
    ax_top.step(times, N_raw, where='post', label='N(t)', linewidth=1.5)
    ax_top.set_ylabel('count')
    ax_top.set_title('N(t) — Sample Path')
    ax_top.legend(loc='best')
    _format_date_axis(ax_top)

    # Bottom: WIP Growth Rate N(T)/T
    ax = axes[1]
    ax.plot(times, N_over_T, label='N(t)/T', linewidth=1.9, zorder=3)
    ax.axhline(0.0, linewidth=0.8, alpha=0.6, zorder=1)
    ax.axhline(1.0, linewidth=1.0, alpha=1.0, linestyle=':', zorder=1)
    _format_date_axis(ax)
    ax.set_xlabel('time')
    ax.set_ylabel('rate')
    ax.set_title(f"{title_prefix}: WIP Growth Rate - N(t)/T" if title_prefix else 'WIP Growth Rate - N(t)/T')
    ax.legend(loc='best')

    finite_vals_N = N_over_T[np.isfinite(N_over_T)]
    if finite_vals_N.size:
        top = float(np.nanpercentile(finite_vals_N, 99.5))
        bot = float(np.nanmin(finite_vals_N))
        ax.set_ylim(bottom=min(0.0, bot * 1.05), top=top * 1.05)

    fig.tight_layout()
    fig.savefig(out_path_N, dpi=200)
    plt.close(fig)
    written.append(out_path_N)

    # --------------------- Chart 2: R(t) sample path + R(T)/T ---------------------
    out_path_R = os.path.join(out_dir, "stability/panels/total_age_growth_rate.png")
    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(10, 7.8), sharex=True)

    # Top: R(t) — total age of WIP at time t
    ax_top = axes[0]
    ax_top.plot(times, R_raw, label='R(t) [hours]', linewidth=1.5, zorder=3)
    ax_top.set_ylabel('hours')
    ax_top.set_title('R(t) — Total age of WIP')
    ax_top.legend(loc='best')
    _format_date_axis(ax_top)

    # Bottom: Total Age Growth Rate R(T)/T
    ax = axes[1]
    ax.plot(times, R_over_T, label="R(T)/T", linewidth=1.9, zorder=3)
    ax.axhline(0.0, linewidth=0.8, alpha=0.6, zorder=1)
    ax.axhline(1.0, linewidth=1.0, alpha=1.0, linestyle=":", zorder=1)  # reference guide

    _format_date_axis(ax)
    ax.set_xlabel("time")
    ax.set_ylabel("rate")
    ax.set_title(
        f"{title_prefix}: Total Age Growth Rate - R(T)/T" if title_prefix else "Total Age Growth Rate - R(T)/T")
    ax.legend(loc="best")

    finite_vals_R = R_over_T[np.isfinite(R_over_T)]
    if finite_vals_R.size:
        top = float(np.nanpercentile(finite_vals_R, 99.5))
        bot = float(np.nanmin(finite_vals_R))
        ax.set_ylim(bottom=min(0.0, bot * 1.05), top=top * 1.05)

    fig.tight_layout()
    fig.savefig(out_path_R, dpi=200)
    plt.close(fig)
    written.append(out_path_R)

    # --------------------- 4-row stack: Equilibrium and Coherence --------------
    out_path_stack = os.path.join(out_dir, "stability/rate_stability.png")
    fig, axes = plt.subplots(nrows=4, ncols=1, figsize=(12, 13.5), sharex=True)

    # Panel 1: N(T)/T
    axN = axes[0]
    axN.plot(times, N_over_T, label="N(T)/T", linewidth=1.9, zorder=3)
    axN.axhline(0.0, linewidth=0.8, alpha=0.6, zorder=1)
    axN.axhline(1.0, linewidth=1.0, alpha=1.0, linestyle=":", zorder=1)
    _format_date_axis(axN)
    axN.set_ylabel("rate")
    axN.set_title("WIP Growth Rate: N(T)/T")
    axN.legend(loc="best")
    if finite_vals_N.size:
        topN = float(np.nanpercentile(finite_vals_N, 99.5))
        botN = float(np.nanmin(finite_vals_N))
        axN.set_ylim(bottom=min(0.0, botN * 1.05), top=topN * 1.05)

    # Panel 2: R(T)/T
    axR = axes[1]
    axR.plot(times, R_over_T, label="R(T)/T", linewidth=1.9, zorder=3)
    axR.axhline(0.0, linewidth=0.8, alpha=0.6, zorder=1)
    axR.axhline(1.0, linewidth=1.0, alpha=1.0, linestyle=":", zorder=1)
    _format_date_axis(axR)
    axR.set_ylabel("rate")
    axR.set_title("Total Age Growth Rate: R(T)/T")
    axR.legend(loc="best")
    if finite_vals_R.size:
        topR = float(np.nanpercentile(finite_vals_R, 99.5))
        botR = float(np.nanmin(finite_vals_R))
        axR.set_ylim(bottom=min(0.0, botR * 1.05), top=topR * 1.05)

    # Panel 3: λ*(T)
    axLam = axes[2]
    axLam.plot(times, lam_star_ts, label="λ*(T) [1/hr]", linewidth=1.9, zorder=3)
    axLam.axhline(0.0, linewidth=0.8, alpha=0.6, zorder=1)
    _format_date_axis(axLam)
    axLam.set_ylabel("[1/hr]")
    axLam.set_title("λ*(T) — running arrival rate")
    axLam.legend(loc="best")
    # Clip like other charts
    try:
        _clip_axis_to_percentile(
            axLam, times, lam_star_ts,
            upper_p=getattr(args, "lambda_pctl", None),
            lower_p=getattr(args, "lambda_lower_pctl", None),
            warmup_hours=float(getattr(args, "lambda_warmup", 0.0) or 0.0),
        )
    except Exception:
        pass

    # Panel 4: W-coherence overlay
    axW = axes[3]
    axW.plot(times, w_ts,        label="w(T) [hrs] (finite-window)", linewidth=1.9, zorder=3)
    axW.plot(times, W_star_ts,   label="W*(T) [hrs] (completed mean)", linewidth=1.9, linestyle="--", zorder=3)
    axW.axhline(0.0, linewidth=0.8, alpha=0.6, zorder=1)
    _format_date_axis(axW)
    axW.set_xlabel("time")
    axW.set_ylabel("hours")
    axW.set_title("w(T) vs W*(T) — coherence")
    axW.legend(loc="best")

    # Subtitle + caption
    fig.suptitle("Equilibrium and Coherence", fontsize=14, y=0.98)
    try:
        if caption_text:
            _add_caption(fig, caption_text)
    except Exception:
        pass

    fig.tight_layout(rect=(0, 0.06, 1, 0.96))
    fig.savefig(out_path_stack, dpi=200)
    plt.close(fig)
    written.append(out_path_stack)

    return written

def plot_llaw_manifold_3d(
    df,
    metrics,                           # FlowMetricsResult
    out_dir: str,
    title: str = "Manifold view: L = Λ · w (log-space plane z = x + y)",
    caption: Optional[str] = None,
    figsize: Tuple[int, int] = (9, 7),
    elev: float = 28.0,
    azim: float = -135.0,
    alpha_surface: float = 0.22,       # kept in signature; not used explicitly
    wireframe_stride: int = 6,
    point_size: int = 16,
) -> List[str]:
    """Log-linear manifold: z = x + y with x=log Λ, y=log w, z=log L.
    Plots only the finite-time trajectory on a filled plane (no empirical series).
    """
    import os
    import numpy as np
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    # ---- helpers ------------------------------------------------------------
    def _safe_log(a: np.ndarray) -> np.ndarray:
        """Natural log with nan for non-positive entries."""
        out = np.full_like(a, np.nan, dtype=float)
        m = np.isfinite(a) & (a > 0)
        out[m] = np.log(a[m])
        return out

    def _finite(*arrs):
        m = np.ones_like(arrs[0], dtype=bool)
        for a in arrs:
            m &= np.isfinite(a)
        return m

    def _pad_range(v: np.ndarray, pad: float = 0.05) -> Tuple[float, float]:
        v = v[np.isfinite(v)]
        if v.size == 0:
            return (-1.0, 1.0)
        lo, hi = float(np.nanmin(v)), float(np.nanmax(v))
        span = hi - lo
        if not np.isfinite(span) or span <= 0:
            span = 1.0
        return lo - pad * span, hi + pad * span

    # ---- finite-time series (on-plane in log space) -------------------------
    T = metrics.times  # not used here but kept for signature compatibility
    L_vals = np.asarray(metrics.L, dtype=float)
    Lam_vals = np.asarray(metrics.Lambda, dtype=float)
    w_vals = np.asarray(metrics.w, dtype=float)

    # Logs for the finite trio
    x_fin = _safe_log(Lam_vals)   # log Λ
    y_fin = _safe_log(w_vals)     # log w
    z_fin = _safe_log(L_vals)     # log L
    mask_fin = _finite(x_fin, y_fin, z_fin)

    # ---- figure / axes ------------------------------------------------------
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection="3d")
    ax.grid(False)
    try:
        ax.set_proj_type("ortho")  # orthographic for honest distances
    except Exception:
        pass

    # Plane extents from finite trajectory only
    x_lo, x_hi = _pad_range(x_fin[mask_fin] if mask_fin.any() else np.array([0.0]))
    y_lo, y_hi = _pad_range(y_fin[mask_fin] if mask_fin.any() else np.array([0.0]))

    # Build grid for plane z = x + y
    nx = max(10, 2 * wireframe_stride)
    ny = max(10, 2 * wireframe_stride)
    Xg = np.linspace(x_lo, x_hi, nx)
    Yg = np.linspace(y_lo, y_hi, ny)
    X, Y = np.meshgrid(Xg, Yg)
    Z = X + Y

    # Filled plane: darker grey, semi-opaque; no mesh lines
    ax.plot_surface(
        X, Y, Z,
        color="dimgray",
        alpha=0.5,
        linewidth=0,
        antialiased=True
    )

    # ---- finite-time trajectory (lies on the plane) -------------------------
    if mask_fin.any():
        ax.plot(x_fin[mask_fin], y_fin[mask_fin], z_fin[mask_fin],
                lw=1.6, label="(log Λ, log w, log L)")

    # ---- labels / view / legend / caption ----------------------------------
    ax.set_xlabel("log Λ")
    ax.set_ylabel("log w")
    ax.set_zlabel("log L")
    ax.view_init(elev=elev, azim=azim)
    ax.set_title(title)
    try:
        ax.legend(loc="upper left")
    except Exception:
        pass

    if caption:
        fig.text(0.01, 0.01, caption, ha="left", va="bottom", fontsize=9)

    # z-limits: include plane and line
    if np.isfinite(Z).any():
        z_hi = np.nanmax([np.nanmax(Z), np.nanmax(z_fin[mask_fin]) if mask_fin.any() else np.nanmin(Z)])
        z_lo = np.nanmin([np.nanmin(Z), np.nanmin(z_fin[mask_fin]) if mask_fin.any() else np.nanmax(Z)])
        ax.set_zlim(z_lo, z_hi)

    plt.tight_layout()
    out_path = os.path.join(out_dir, "advanced/invariant_manifold3D_log.png")
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)

    return [out_path]


def produce_all_charts(df, csv_path, args, filter_result, metrics, empirical_metrics):
    out_dir = ensure_output_dirs(csv_path, output_dir=args.output_dir, clean=args.clean)
    written: List[str] = []
    # create plots
    written += plot_core_flow_metrics_charts(df, args, filter_result, metrics, out_dir)
    written += plot_convergence_charts(df, args, filter_result, metrics, empirical_metrics, out_dir)
    written += plot_stability_charts(df, args, filter_result, metrics, out_dir)
    written += plot_advanced_charts(df, args, filter_result, metrics, out_dir)
    written += plot_misc_charts(df, args, filter_result, metrics, out_dir)
    return written
