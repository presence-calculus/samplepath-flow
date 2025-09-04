# -*- coding: utf-8 -*-
# Copyright (c) 2025 Krishna Kumar
# SPDX-License-Identifier: MIT
import os
from typing import List, Optional

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes

from spath.filter import FilterResult
from spath.metrics import FlowMetricsResult, compute_dynamic_empirical_series, compute_tracking_errors, \
    compute_coherence_score, compute_end_effect_series

def _add_caption(fig: Figure, text: str) -> None:
    """Add a caption below the x-axis."""
    fig.subplots_adjust(bottom=0.31)  # adjust bottom margin
    fig.text(
        0.5, 0.02, text,
        ha="center", va="bottom",
        fontsize=9, color="gray"
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
    _format_date_axis(ax)

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
        out_path: str
) -> None:
    """Format the axis, add optional caption, save the figure, and close it."""
    _format_axis(ax, title, unit, ylabel)
    _format_fig(caption, fig)
    fig.savefig(out_path)
    plt.close(fig)

def draw_line_chart(times: List[pd.Timestamp], values: np.ndarray, title: str, ylabel: str, out_path: str) -> None:
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(times, values, label=ylabel)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xlabel("Date")
    ax.legend()
    _format_date_axis(ax)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def draw_lambda_chart(times: List[pd.Timestamp],
                      values: np.ndarray,
                      title: str,
                      ylabel: str,
                      out_path: str,
                      lambda_pctl_upper: Optional[float] = None,
                      lambda_pctl_lower: Optional[float] = None,
                      lambda_warmup_hours: Optional[float] = None
                      ) -> None:

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(times, values, label=ylabel)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xlabel("Date")
    ax.legend()
    _clip_axis_to_percentile(ax, times, values,
                             upper_p=lambda_pctl_upper,
                             lower_p=lambda_pctl_lower,
                             warmup_hours=lambda_warmup_hours)
    _format_date_axis(ax)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def draw_line_chart_with_scatter(times: List[pd.Timestamp],
                                 values: np.ndarray,
                                 title: str,
                                 ylabel: str,
                                 out_path: str,
                                 scatter_times: List[pd.Timestamp],
                                 scatter_values: np.ndarray,
                                 scatter_label: str = "Item time in system") -> None:
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(times, values, label=ylabel)
    if scatter_times is not None and scatter_values is not None and len(scatter_times) > 0:
        ax.scatter(scatter_times, scatter_values, s=16, alpha=0.6, marker='o', label=scatter_label)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xlabel("Date")
    ax.legend()
    _format_date_axis(ax)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def draw_step_chart(times: List[pd.Timestamp], values: np.ndarray, title: str, ylabel: str, out_path: str, unit: str = 'Timestamp', caption:str = "") -> None:
    fig, ax = plt.subplots(figsize=(10, 3.2))
    ax.step(times, values, where='post', label=ylabel)

    _format_and_save(fig, ax, title, ylabel, unit, caption, out_path)





def draw_bar_chart(times: List[pd.Timestamp], values: np.ndarray, title: str, ylabel: str, out_path: str) -> None:
    fig, ax = plt.subplots(figsize=(10, 3.2))
    ax.bar(times, values, label=ylabel)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xlabel("Date")
    ax.legend()
    _format_date_axis(ax)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def draw_convergence_panel(times: List[pd.Timestamp],
                           w_vals: np.ndarray,
                           lam_vals: np.ndarray,
                           W_emp: float,
                           lam_emp: float,
                           title: str,
                           out_path: str,
                           lambda_pctl_upper: Optional[float] = None,
                           lambda_pctl_lower: Optional[float] = None,
                           lambda_warmup_hours: Optional[float] = None
                           ) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(12, 6.5), sharex=True)

    axes[0].plot(times, w_vals, label='w(T) [hrs]')
    if np.isfinite(W_emp):
        axes[0].axhline(W_emp, linestyle='--', label=f'W* ≈ {W_emp:.2f} h')
    axes[0].set_title('w(T) vs W*  — residence time convergence')
    axes[0].set_ylabel('hours')
    axes[0].legend()

    axes[1].plot(times, lam_vals, label='Λ(T) [1/hr]')
    if np.isfinite(lam_emp):
        axes[1].axhline(lam_emp, linestyle='--', label=f'λ* ≈ {lam_emp:.4f} 1/hr')
    axes[1].set_title('Λ(T) vs λ*  — arrival rate convergence')
    axes[1].set_ylabel('1/hr')
    axes[1].set_xlabel('Date')
    axes[1].legend()

    _clip_axis_to_percentile(axes[1], times, lam_vals,
                             upper_p=lambda_pctl_upper,
                             lower_p=lambda_pctl_lower,
                             warmup_hours=lambda_warmup_hours)

    for ax in axes:
        _format_date_axis(ax)

    fig.suptitle(title)
    plt.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out_path)
    plt.close(fig)


def draw_dynamic_convergence_panel(times: List[pd.Timestamp],
                                   w_vals: np.ndarray,
                                   lam_vals: np.ndarray,
                                   W_star: np.ndarray,
                                   lam_star: np.ndarray,
                                   title: str,
                                   out_path: str,
                                   lambda_pctl_upper: Optional[float] = None,
                                   lambda_pctl_lower: Optional[float] = None,
                                   lambda_warmup_hours: Optional[float] = None
                                   ) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(12, 6.5), sharex=True)

    axes[0].plot(times, w_vals, label='w(T) [hrs]')
    axes[0].plot(times, W_star, linestyle='--', label='W*(t) [hrs] (completed ≤ t)')
    axes[0].set_title('w(T) vs W*(t)  — dynamic')
    axes[0].set_ylabel('hours')
    axes[0].legend()

    axes[1].plot(times, lam_vals, label='Λ(T) [1/hr]')
    axes[1].plot(times, lam_star, linestyle='--', label='λ*(t) [1/hr] (arrivals ≤ t)')
    axes[1].set_title('Λ(T) vs λ*(t)  — arrival rate (dynamic)')
    axes[1].set_ylabel('1/hr')
    axes[1].set_xlabel('Date')
    axes[1].legend()

    _clip_axis_to_percentile(axes[1], times, lam_vals,
                             upper_p=lambda_pctl_upper,
                             lower_p=lambda_pctl_lower,
                             warmup_hours=lambda_warmup_hours)

    for ax in axes:
        _format_date_axis(ax)

    fig.suptitle(title)
    plt.tight_layout(rect=(0, 0, 1, 0.96))
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




def plot_core_flow_metrics(
        args,
        filter_result:Optional[FilterResult],
        metrics: FlowMetricsResult,
        out_dir: str
) -> List[str]:

    out_dir = ensure_output_dir(out_dir)
    filter_label = filter_result.label if filter_result else ""

    path_N = os.path.join(out_dir, "timestamp_N.png")
    draw_step_chart(metrics.times, metrics.N, f"N(t) — Active elements", "N(t)", path_N, caption=f"Filters: {filter_label}")

    path_L = os.path.join(out_dir, "timestamp_L.png")
    draw_line_chart(metrics.times, metrics.L, f"L(T) — Time-average N(t) (timestamp, {filter_label})", "L(T)", path_L)

    path_Lam = os.path.join(out_dir, "timestamp_Lambda.png")
    draw_lambda_chart(metrics.times, metrics.Lambda, f"Λ(T) — Cumulative arrival rate (timestamp, {filter_label})", "Λ(T) [1/hr]",
                      path_Lam, args.lambda_pctl, args.lambda_lower_pctl, args.lambda_warmup)

    path_w = os.path.join(out_dir, "timestamp_w.png")
    draw_line_chart(metrics.times, metrics.w, f"w(T) — Average residence time  (timestamp, {filter_label})",
                    "w(T) [hrs]", path_w)

    return [path_N, path_L, path_Lam, path_w]


def plot_sojourn_time_scatter(args, df, filter_result, metrics,out_dir) -> List[str]:
    t_scatter_times: List[pd.Timestamp] = []
    t_scatter_vals = np.array([])
    written = []
    if args.incomplete:
        if len(metrics.times) > 0:
            t_scatter_times = df["start_ts"].tolist()
            t_scatter_vals = df["duration_hr"].to_numpy()

    else:
        df_c = df[df["end_ts"].notna()].copy()
        if not df_c.empty:
            t_scatter_times = df_c["end_ts"].tolist()
            t_scatter_vals = df_c["duration_hr"].to_numpy()

    if len(t_scatter_times) > 0:
        ts_w_scatter = os.path.join(out_dir, "timestamp_w_with_scatter.png")
        label = "Item age at sweep end" if args.incomplete else "Item time in system"
        draw_line_chart_with_scatter(metrics.times, metrics.w,
                                     f"w(T) — average residence time (timestamp, {filter_result.label})",
                                     "w(T) [hrs]", ts_w_scatter, t_scatter_times, t_scatter_vals, scatter_label=label)

        written += [ts_w_scatter]

    return written


def draw_four_panel_column(times: List[pd.Timestamp],
                           N_vals: np.ndarray,
                           L_vals: np.ndarray,
                           Lam_vals: np.ndarray,
                           w_vals: np.ndarray,
                           title: str,
                           out_path: str,
                           lambda_pctl_upper: Optional[float] = None,
                           lambda_pctl_lower: Optional[float] = None,
                           lambda_warmup_hours: Optional[float] = None
                           ) -> None:
    fig, axes = plt.subplots(4, 1, figsize=(12, 11), sharex=True)

    axes[0].step(times, N_vals, where='post', label='N(t)')
    axes[0].set_title('N(t) — active elements')
    axes[0].set_ylabel('N(t)')
    axes[0].legend()

    axes[1].plot(times, L_vals, label='L(T)')
    axes[1].set_title('L(T) — time-average of N(t)')
    axes[1].set_ylabel('L(T)')
    axes[1].legend()

    axes[2].plot(times, Lam_vals, label='Λ(T) [1/hr]')
    axes[2].set_title('Λ(T) — cumulative arrival rate')
    axes[2].set_ylabel('Λ(T) [1/hr]')
    axes[2].legend()
    _clip_axis_to_percentile(axes[2], times, Lam_vals,
                             upper_p=lambda_pctl_upper,
                             lower_p=lambda_pctl_lower,
                             warmup_hours=lambda_warmup_hours)

    axes[3].plot(times, w_vals, label='w(T) [hrs]')
    axes[3].set_title('w(T) — average residence time')
    axes[3].set_ylabel('w(T) [hrs]')
    axes[3].set_xlabel('Date')
    axes[3].legend()

    for ax in axes:
        _format_date_axis(ax)

    fig.suptitle(title)
    plt.tight_layout(rect=(0, 0, 1, 0.97))
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
    axes[0].set_title('N(t) — active elements')
    axes[0].set_ylabel('N(t)')
    axes[0].legend()

    axes[1].plot(times, L_vals, label='L(T)')
    axes[1].set_title('L(T) — time-average of N(t)')
    axes[1].set_ylabel('L(T)')
    axes[1].legend()

    axes[2].plot(times, Lam_vals, label='Λ(T) [1/hr]')
    axes[2].set_title('Λ(T) — cumulative arrival rate')
    axes[2].set_ylabel('Λ(T) [1/hr]')
    axes[2].legend()
    _clip_axis_to_percentile(axes[2], times, Lam_vals,
                             upper_p=lambda_pctl_upper,
                             lower_p=lambda_pctl_lower,
                             warmup_hours=lambda_warmup_hours)

    axes[3].plot(times, w_vals, label='w(T) [hrs]')
    if scatter_times is not None and scatter_values is not None and len(scatter_times) > 0:
        axes[3].scatter(scatter_times, scatter_values, s=16, alpha=0.6, marker='o', label=scatter_label)
    axes[3].set_title('w(T) — average residence time')
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
    axes[0].set_title('N(t) — active elements')
    axes[0].set_ylabel('N(t)')
    axes[0].legend()

    axes[1].plot(times, L_vals, label='L(T)')
    axes[1].set_title('L(T) — time-average of N(t)')
    axes[1].set_ylabel('L(T)')
    axes[1].legend()

    axes[2].plot(times, Lam_vals, label='Λ(T) [1/hr]')
    axes[2].set_title('Λ(T) — cumulative arrival rate')
    axes[2].set_ylabel('Λ(T) [1/hr]')
    axes[2].legend()
    _clip_axis_to_percentile(axes[2], times, Lam_vals,
                             upper_p=lambda_pctl_upper,
                             lower_p=lambda_pctl_lower,
                             warmup_hours=lambda_warmup_hours)

    axes[3].plot(times, w_vals, label='w(T) [hrs]')
    axes[3].set_title('w(T) — average residence time (plain, own scale)')
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


def ensure_output_dir(csv_path: str) -> str:
    base = os.path.basename(csv_path)
    stem = os.path.splitext(base)[0]
    out_dir = os.path.join("charts", stem)
    os.makedirs(out_dir, exist_ok=True)
    return out_dir


def plot_coherence_charts(df, args, filter_result, metrics, out_dir):
    # Empirical targets & dynamic baselines
    horizon_days = args.horizon_days
    epsilon = args.epsilon
    lambda_pctl_upper = args.lambda_pctl
    lambda_pctl_lower = args.lambda_lower_pctl
    lambda_warmup_hours = args.lambda_warmup
    mode_label = filter_result.label

    written: List[str] = []

    if len(metrics.times) > 0:
        W_star_ts, lam_star_ts = compute_dynamic_empirical_series(df, metrics.times)
    else:
        W_star_ts = lam_star_ts = np.array([])
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
        ts_conv_dyn = os.path.join(out_dir, 'timestamp_convergence_dynamic.png')
        draw_dynamic_convergence_panel(metrics.times, metrics.w, metrics.Lambda, W_star_ts, lam_star_ts,
                                       f'Dynamic convergence (timestamp, {mode_label})', ts_conv_dyn,
                                       lambda_pctl_upper=lambda_pctl_upper, lambda_pctl_lower=lambda_pctl_lower,
                                       lambda_warmup_hours=lambda_warmup_hours)
        written.append(ts_conv_dyn)

        ts_conv_dyn3 = os.path.join(out_dir, 'timestamp_convergence_dynamic_errors.png')
        draw_dynamic_convergence_panel_with_errors(metrics.times, metrics.w, metrics.Lambda, W_star_ts, lam_star_ts,
                                                   eW_ts, eLam_ts, epsilon,
                                                   f'Dynamic convergence + errors (timestamp, {mode_label})',
                                                   ts_conv_dyn3, lambda_pctl_upper=lambda_pctl_upper,
                                                   lambda_pctl_lower=lambda_pctl_lower,
                                                   lambda_warmup_hours=lambda_warmup_hours)
        written.append(ts_conv_dyn3)
    # --- End-effect diagnostics ---
    rA_ts, rB_ts, rho_ts = compute_end_effect_series(df, metrics.times, metrics.A, W_star_ts) if len(
        metrics.times) > 0 else (np.array([]), np.array([]), np.array([]))
    if len(metrics.times) > 0:
        ts_conv_dyn4 = os.path.join(out_dir, 'timestamp_convergence_dynamic_errors_endeffects.png')
        draw_dynamic_convergence_panel_with_errors_and_endeffects(
            metrics.times, metrics.w, metrics.Lambda, W_star_ts, lam_star_ts, eW_ts, eLam_ts,
            rA_ts, rB_ts, rho_ts, epsilon,
            f'Dynamic convergence + errors + end-effects (timestamp, {mode_label})', ts_conv_dyn4,
            lambda_pctl_upper=lambda_pctl_upper, lambda_pctl_lower=lambda_pctl_lower,
            lambda_warmup_hours=lambda_warmup_hours)
        written.append(ts_conv_dyn4)

        # Write coherence summary (and print)
        if coh_summary_lines:
            txt_path = os.path.join(out_dir, "coherence_summary.txt")
            with open(txt_path, "w") as f:
                for line in coh_summary_lines:
                    f.write(line + "\n")
            print("\n".join(coh_summary_lines))
            written.append(txt_path)

    return written


def plot_core_metrics_stack(args, filter_result, metrics, out_dir):
    four_col_stack = os.path.join(out_dir, 'timestamp_stack.png')
    draw_four_panel_column(metrics.times, metrics.N, metrics.L, metrics.Lambda, metrics.w,
                           f'Finite-window metrics (timestamp, {filter_result.label})', four_col_stack, args.lambda_pctl,
                           args.lambda_lower_pctl, args.lambda_warmup)
    return [four_col_stack]


def plot_five_column_stacks(df, args, filter_result, metrics, out_dir):
    t_scatter_times = df["start_ts"].tolist()
    t_scatter_vals = df["duration_hr"].to_numpy()
    written = []
    if args.with_A:
        col_ts5 = os.path.join(out_dir, 'timestamp_stack_with_A.png')
        draw_five_panel_column(metrics.times, metrics.N, metrics.Lambda, metrics.Lambda, metrics.w, metrics.A,
                               f'Finite-window metrics incl. A(T) (timestamp, {filter_result.label})', col_ts5,
                               scatter_times=t_scatter_times, scatter_values=t_scatter_vals,
                               lambda_pctl_upper=args.lambda_pctl, lambda_pctl_lower=args.lambda_lower_pctl,
                               lambda_warmup_hours=args.lambda_warmup)
        written.append(col_ts5)

    elif args.scatter:
        col_ts5s = os.path.join(out_dir, 'timestamp_stack_with_scatter.png')
        draw_five_panel_column_with_scatter(metrics.times, metrics.N, metrics.L, metrics.Lambda, metrics.w,
                                            f'Finite-window metrics with w(T) plain + w(T)+scatter (timestamp, {filter_result.label})',
                                            col_ts5s,
                                            scatter_times=t_scatter_times, scatter_values=t_scatter_vals,
                                            lambda_pctl_upper=args.lambda_pctl,
                                            lambda_pctl_lower=args.lambda_lower_pctl,
                                            lambda_warmup_hours=args.lambda_warmup)
        written.append(col_ts5s)

    return written
