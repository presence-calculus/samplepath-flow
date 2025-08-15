# -*- coding: utf-8 -*-
# Copyright (c) 2025 Krishna Kumar
# SPDX-License-Identifier: MIT
from typing import List, Optional

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


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


def draw_step_chart(times: List[pd.Timestamp], values: np.ndarray, title: str, ylabel: str, out_path: str) -> None:
    fig, ax = plt.subplots(figsize=(10, 3.2))
    ax.step(times, values, where='post', label=ylabel)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xlabel("Date")
    ax.legend()
    _format_date_axis(ax)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


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
    axes[0].set_title('N(t) — active processes')
    axes[0].set_ylabel('N(t)')
    axes[0].legend()

    axes[1].plot(times, L_vals, label='L(T)')
    axes[1].set_title('L(T) — time-average number')
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
    axes[3].set_title('w(T) — average residence time in window')
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
    axes[0].set_title('N(t) — active processes')
    axes[0].set_ylabel('N(t)')
    axes[0].legend()

    axes[1].plot(times, L_vals, label='L(T)')
    axes[1].set_title('L(T) — time-average number')
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
    axes[3].set_title('w(T) — average residence time in window')
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
    axes[0].set_title('N(t) — active processes')
    axes[0].set_ylabel('N(t)')
    axes[0].legend()

    axes[1].plot(times, L_vals, label='L(T)')
    axes[1].set_title('L(T) — time-average number')
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


def _format_date_axis(ax: plt.Axes) -> None:
    try:
        ax.figure.autofmt_xdate()
    except Exception:
        pass
