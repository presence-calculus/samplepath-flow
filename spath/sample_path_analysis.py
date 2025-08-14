# -*- coding: utf-8 -*-
# Copyright (c) 2025 Krishna Kumar
# SPDX-License-Identifier: MIT

"""
Finite-window flow metrics & convergence diagnostics with end-effect panel.
See README.md for context.
"""

import os
import sys
from typing import List, Optional, Tuple
from argparse import Namespace

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt



import cli
from csv_loader import csv_to_dataframe
from filter import FilterResult, apply_filters

# -------------------------------
# Plot helpers
# -------------------------------
def _format_date_axis(ax: plt.Axes) -> None:
    try:
        ax.figure.autofmt_xdate()
    except Exception:
        pass


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


# === NEW: End-effect metrics ===
def compute_end_effect_series(df: pd.DataFrame,
                              times: List[pd.Timestamp],
                              A_vals: np.ndarray,
                              W_star: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute end-effect diagnostics over [t0, t]:

    Returns arrays aligned to `times`:
      - rA(T) = E(T) / A(T), where E(T) = A(T) - sum(full durations of items fully contained)
      - rB(T) = B(T) / total_items_started_by_t, boundary share
      - rho(T) = T / W*(t), window/typical-duration ratio
    """
    n = len(times)
    rA = np.full(n, np.nan, dtype=float)
    rB = np.full(n, np.nan, dtype=float)
    rho = np.full(n, np.nan, dtype=float)
    if n == 0:
        return rA, rB, rho

    df = df.copy()
    df["duration_h"] = (df["end_ts"] - df["start_ts"]).dt.total_seconds() / 3600.0
    df_sorted_by_end = df.sort_values("end_ts")
    df_sorted_by_start = df.sort_values("start_ts")

    t0 = times[0]

    for i, t in enumerate(times):
        elapsed_h = (t - t0).total_seconds() / 3600.0
        if elapsed_h <= 0:
            continue

        A_T = float(A_vals[i]) if i < len(A_vals) and np.isfinite(A_vals[i]) else np.nan
        if not np.isfinite(A_T) or A_T <= 0:
            continue

        mask_full = df_sorted_by_end["end_ts"].notna() & (df_sorted_by_end["end_ts"] <= t)
        A_full = float(df_sorted_by_end.loc[mask_full, "duration_h"].sum()) if mask_full.any() else 0.0

        E_T = max(A_T - A_full, 0.0)
        rA[i] = E_T / A_T if A_T > 0 else np.nan

        mask_started = (df_sorted_by_start["start_ts"] <= t)
        total_started = int(mask_started.sum())
        mask_incomplete_by_t = mask_started & ((df_sorted_by_start["end_ts"].isna()) | (df_sorted_by_start["end_ts"] > t))
        B_T = int(mask_incomplete_by_t.sum())
        rB[i] = (B_T / total_started) if total_started > 0 else np.nan

        Wstar_t = float(W_star[i]) if i < len(W_star) else float('nan')
        rho[i] = (elapsed_h / Wstar_t) if np.isfinite(Wstar_t) and Wstar_t > 0 else np.nan

    return rA, rB, rho


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


# -------------------------------
# Core computations
# -------------------------------


def ensure_output_dir(csv_path: str) -> str:
    base = os.path.basename(csv_path)
    stem = os.path.splitext(base)[0]
    out_dir = os.path.join("charts", stem)
    os.makedirs(out_dir, exist_ok=True)
    return out_dir


def build_event_stream(df: pd.DataFrame) -> List[Tuple[pd.Timestamp, int, int]]:
    """Return sorted events: (time, deltaN, arrivals_at_time)."""
    events: List[Tuple[pd.Timestamp, int, int]] = []
    for _, row in df.iterrows():
        st = row["start_ts"]
        events.append((st, +1, 1))
        et = row["end_ts"]
        if pd.notna(et):
            events.append((et, -1, 0))
    events.sort(key=lambda x: (x[0], -x[1]))
    return events


def sweep_at_times(events: List[Tuple[pd.Timestamp, int, int]], sample_times: List[pd.Timestamp]) -> Tuple[List[pd.Timestamp], np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute L(T), Λ(T), w(T), N(t), A(T) at given sample times."""
    if not events:
        return sample_times, np.array([]), np.array([]), np.array([]), np.array([]), np.array([])

    events = sorted(events, key=lambda e: e[0])
    sample_times = sorted(sample_times)
    t0 = sample_times[0]

    A = 0.0
    N = 0
    arrivals = 0

    out_L = []
    out_Lam = []
    out_w = []
    out_N = []
    out_A = []

    ev_i = 0
    prev_time = t0

    for t in sample_times:
        while ev_i < len(events) and events[ev_i][0] <= t:
            ev_time, dN, a = events[ev_i]
            dt_h = (ev_time - prev_time).total_seconds() / 3600.0
            if dt_h > 0:
                A += N * dt_h
                prev_time = ev_time
            N += dN
            arrivals += a
            ev_i += 1

        dt_h = (t - prev_time).total_seconds() / 3600.0
        if dt_h > 0:
            A += N * dt_h
            prev_time = t

        elapsed_h = (t - t0).total_seconds() / 3600.0
        L = (A / elapsed_h) if elapsed_h > 0 else np.nan
        Lam = (arrivals / elapsed_h) if elapsed_h > 0 else np.nan
        w = (A / arrivals) if arrivals > 0 else np.nan

        out_L.append(L)
        out_Lam.append(Lam)
        out_w.append(w)
        out_N.append(N)
        out_A.append(A)

    return sample_times, np.array(out_L), np.array(out_Lam), np.array(out_w), np.array(out_N), np.array(out_A)


def sweep_timestamp_series(events: List[Tuple[pd.Timestamp, int, int]]):
    unique_times: List[pd.Timestamp] = sorted({t for t, _, _ in events})
    return sweep_at_times(events, unique_times)


def sweep_daily_series(events: List[Tuple[pd.Timestamp, int, int]]):
    if not events:
        return [], np.array([]), np.array([]), np.array([]), np.array([]), np.array([])
    first = min(t for t, _, _ in events).normalize()
    last = max(t for t, _, _ in events).normalize()
    days = pd.date_range(start=first, end=last, freq="D")
    times = [pd.Timestamp(t) for t in days.to_pydatetime()]
    return sweep_at_times(events, times)


def compute_empirical_targets(df: pd.DataFrame, t0: pd.Timestamp, t_end: pd.Timestamp) -> Tuple[float, float]:
    comp = df[df["end_ts"].notna()].copy()
    if not comp.empty:
        W_emp = float(((comp["end_ts"] - comp["start_ts"]).dt.total_seconds() / 3600.0).mean())
    else:
        W_emp = float('nan')
    arrivals = int((df["start_ts"] <= t_end).sum())
    elapsed_h = (t_end - t0).total_seconds() / 3600.0 if pd.notna(t_end) and pd.notna(t0) else 0.0
    lam_emp = float(arrivals / elapsed_h) if elapsed_h > 0 else float('nan')
    return W_emp, lam_emp


def compute_dynamic_empirical_series(df: pd.DataFrame,
                                     times: List[pd.Timestamp]) -> Tuple[np.ndarray, np.ndarray]:
    """Return W*(t) and λ*(t) aligned to `times`."""
    n = len(times)
    W_star = np.full(n, np.nan, dtype=float)
    lam_star = np.full(n, np.nan, dtype=float)
    if n == 0:
        return W_star, lam_star

    comp = df[df["end_ts"].notna()].copy().sort_values("end_ts")
    comp_durations = ((comp["end_ts"] - comp["start_ts"]).dt.total_seconds() / 3600.0).to_numpy()
    comp_end_times = comp["end_ts"].to_list()

    starts = df["start_ts"].sort_values().to_list()

    j = 0
    count_c = 0
    sum_c = 0.0
    k = 0
    count_starts = 0
    t0 = times[0]

    for i, t in enumerate(times):
        while j < len(comp_end_times) and comp_end_times[j] <= t:
            sum_c += comp_durations[j]
            count_c += 1
            j += 1
        if count_c > 0:
            W_star[i] = sum_c / count_c

        while k < len(starts) and starts[k] <= t:
            count_starts += 1
            k += 1
        elapsed_h = (t - t0).total_seconds() / 3600.0
        if elapsed_h > 0:
            lam_star[i] = count_starts / elapsed_h

    return W_star, lam_star


# -------------------------------
# Coherence diagnostics
# -------------------------------
def compute_tracking_errors(times: List[pd.Timestamp],
                            w_vals: np.ndarray,
                            lam_vals: np.ndarray,
                            W_star: np.ndarray,
                            lam_star: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = len(times)
    if n == 0:
        return np.array([]), np.array([]), np.array([])
    t0 = times[0]
    elapsed_hours = np.array([(t - t0).total_seconds() / 3600.0 for t in times], dtype=float)

    eW = np.full(n, np.nan, dtype=float)
    eLam = np.full(n, np.nan, dtype=float)

    valid_W = np.isfinite(w_vals) & np.isfinite(W_star) & (W_star > 0)
    valid_L = np.isfinite(lam_vals) & np.isfinite(lam_star) & (lam_star > 0)

    eW[valid_W] = np.abs(w_vals[valid_W] - W_star[valid_W]) / W_star[valid_W]
    eLam[valid_L] = np.abs(lam_vals[valid_L] - lam_star[valid_L]) / lam_star[valid_L]

    return eW, eLam, elapsed_hours


def compute_coherence_score(eW: np.ndarray,
                            eLam: np.ndarray,
                            elapsed_hours: np.ndarray,
                            epsilon: float,
                            horizon_hours: float) -> Tuple[float, int, int]:
    ok_idx = np.isfinite(eW) & np.isfinite(eLam) & (elapsed_hours >= horizon_hours)
    total = int(np.sum(ok_idx))
    if total == 0:
        return float('nan'), 0, 0
    coherent = int(np.sum(np.maximum(eW[ok_idx], eLam[ok_idx]) <= epsilon))
    return coherent / total, coherent, total


# -------------------------------
# Orchestration
# -------------------------------
def produce_all_charts(csv_path: str,
                       args: Namespace,
                       completed_only: bool = False,
                       incomplete_only: bool = False,
                       with_A: bool = False,
                       with_daily_breakdown: bool = False,
                       scatter: bool = False,
                       epsilon: Optional[float] = None,
                       horizon_days: Optional[float] = None,
                       lambda_pctl_upper: Optional[float] = None,
                       lambda_pctl_lower: Optional[float] = None,
                       lambda_warmup_hours: Optional[float] = None,
                       ) -> List[str]:

    df = csv_to_dataframe(csv_path)

    filter_result: FilterResult = apply_filters(df, args)
    df = filter_result.df


    # Build events and sweeps
    events = build_event_stream(df)
    t_times, t_L, t_Lam, t_w, t_N, t_A = sweep_timestamp_series(events)
    d_times, d_L, d_Lam, d_w, d_N, d_A = sweep_daily_series(events)

    # Empirical targets & dynamic baselines
    if len(t_times) > 0:
        W_emp_ts, lam_emp_ts = compute_empirical_targets(df, t_times[0], t_times[-1])
        W_star_ts, lam_star_ts = compute_dynamic_empirical_series(df, t_times)
    else:
        W_emp_ts = lam_emp_ts = float('nan')
        W_star_ts = lam_star_ts = np.array([])
    if len(d_times) > 0:
        W_emp_d, lam_emp_d = compute_empirical_targets(df, d_times[0], d_times[-1])
        W_star_d, lam_star_d = compute_dynamic_empirical_series(df, d_times)
    else:
        W_emp_d = lam_emp_d = float('nan')
        W_star_d = lam_star_d = np.array([])

    # Relative errors & coherence
    eW_ts, eLam_ts, elapsed_ts = compute_tracking_errors(t_times, t_w, t_Lam, W_star_ts, lam_star_ts)
    eW_d, eLam_d, elapsed_d = compute_tracking_errors(d_times, d_w, d_Lam, W_star_d, lam_star_d)

    coh_summary_lines: List[str] = []
    if epsilon is not None and horizon_days is not None:
        h_hrs = float(horizon_days) * 24.0
        sc_ts, ok_ts, tot_ts = compute_coherence_score(eW_ts, eLam_ts, elapsed_ts, float(epsilon), h_hrs)
        sc_d, ok_d, tot_d = compute_coherence_score(eW_d, eLam_d, elapsed_d, float(epsilon), h_hrs)
        coh_summary_lines.append(f"Coherence (timestamp): eps={epsilon:g}, H={horizon_days:g}d -> {ok_ts}/{tot_ts} ({(sc_ts*100 if sc_ts==sc_ts else 0):.1f}%)")
        coh_summary_lines.append(f"Coherence (daily):     eps={epsilon:g}, H={horizon_days:g}d -> {ok_d}/{tot_d} ({(sc_d*100 if sc_d==sc_d else 0):.1f}%)")

    # Scatter arrays
    t_scatter_times: List[pd.Timestamp] = []
    t_scatter_vals = np.array([])
    d_scatter_times: List[pd.Timestamp] = []
    d_scatter_vals = np.array([])
    if scatter:
        if incomplete_only:
            if len(t_times) > 0:
                t_end = t_times[-1]
                t_scatter_times = df["start_ts"].tolist()
                t_scatter_vals = ((t_end - df["start_ts"]).dt.total_seconds() / 3600.0).to_numpy()
            if len(d_times) > 0:
                d_end = d_times[-1]
                d_scatter_times = df["start_ts"].tolist()
                d_scatter_vals = ((d_end - df["start_ts"]).dt.total_seconds() / 3600.0).to_numpy()
        else:
            df_c = df[df["end_ts"].notna()].copy()
            if not df_c.empty:
                t_scatter_times = df_c["end_ts"].tolist()
                t_scatter_vals = ((df_c["end_ts"] - df_c["start_ts"]).dt.total_seconds() / 3600.0).to_numpy()
                d_scatter_times = t_scatter_times
                d_scatter_vals = t_scatter_vals

    mode_label = filter_result.label

    out_dir = ensure_output_dir(csv_path)
    written: List[str] = []

    # Timestamp charts
    ts_L = os.path.join(out_dir, "timestamp_L.png")
    ts_Lam = os.path.join(out_dir, "timestamp_Lambda.png")
    ts_w = os.path.join(out_dir, "timestamp_w.png")
    ts_Np = os.path.join(out_dir, "timestamp_N.png")

    draw_line_chart(t_times, t_L, f"L(T) — time-average number (timestamp, {mode_label})", "L(T)", ts_L)
    draw_lambda_chart(t_times, t_Lam, f"Λ(T) — cumulative arrivals per hour (timestamp, {mode_label})", "Λ(T) [1/hr]", ts_Lam, lambda_pctl_upper, lambda_pctl_lower, lambda_warmup_hours)

    if scatter and len(t_scatter_times) > 0:
        label = "Item age at sweep end" if incomplete_only else "Item time in system"
        draw_line_chart_with_scatter(t_times, t_w,
                                     f"w(T) — average residence time in window (timestamp, {mode_label})",
                                     "w(T) [hrs]", ts_w, t_scatter_times, t_scatter_vals, scatter_label=label)
    else:
        draw_line_chart(t_times, t_w, f"w(T) — average residence time in window (timestamp, {mode_label})", "w(T) [hrs]", ts_w)

    draw_step_chart(t_times, t_N, f"N(t) — active processes (timestamp, {mode_label})", "N(t)", ts_Np)
    written += [ts_L, ts_Lam, ts_w, ts_Np]

    # Convergence diagnostics (timestamp)
    if len(t_times) > 0:
        ts_conv = os.path.join(out_dir, 'timestamp_convergence.png')
        draw_convergence_panel(t_times, t_w, t_Lam, W_emp_ts, lam_emp_ts,
                               f'Convergence diagnostics (timestamp, {mode_label})', ts_conv, lambda_pctl_upper=lambda_pctl_upper, lambda_pctl_lower=lambda_pctl_lower, lambda_warmup_hours=lambda_warmup_hours)
        written.append(ts_conv)

        ts_conv_dyn = os.path.join(out_dir, 'timestamp_convergence_dynamic.png')
        draw_dynamic_convergence_panel(t_times, t_w, t_Lam, W_star_ts, lam_star_ts,
                                       f'Dynamic convergence (timestamp, {mode_label})', ts_conv_dyn, lambda_pctl_upper=lambda_pctl_upper, lambda_pctl_lower=lambda_pctl_lower, lambda_warmup_hours=lambda_warmup_hours)
        written.append(ts_conv_dyn)

        ts_conv_dyn3 = os.path.join(out_dir, 'timestamp_convergence_dynamic_errors.png')
        draw_dynamic_convergence_panel_with_errors(t_times, t_w, t_Lam, W_star_ts, lam_star_ts,
                                                   eW_ts, eLam_ts, epsilon,
                                                   f'Dynamic convergence + errors (timestamp, {mode_label})', ts_conv_dyn3, lambda_pctl_upper=lambda_pctl_upper, lambda_pctl_lower=lambda_pctl_lower, lambda_warmup_hours=lambda_warmup_hours)
        written.append(ts_conv_dyn3)

    # Daily charts
    dL = os.path.join(out_dir, "daily_L.png")
    dLam = os.path.join(out_dir, "daily_Lambda.png")
    dw = os.path.join(out_dir, "daily_w.png")
    dNp = os.path.join(out_dir, "daily_N.png")

    draw_line_chart(d_times, d_L, f"L(T) — time-average number (daily, {mode_label})", "L(T)", dL)
    draw_lambda_chart(d_times, d_Lam, f"Λ(T) — cumulative arrivals per hour (daily, {mode_label})", "Λ(T) [1/hr]", dLam, lambda_pctl_upper, lambda_pctl_lower, lambda_warmup_hours)

    if scatter and len(d_scatter_times) > 0:
        draw_line_chart_with_scatter(d_times, d_w,
                                     f"w(T) — average residence time in window (daily, {mode_label})",
                                     "w(T) [hrs]", dw, d_scatter_times, d_scatter_vals,
                                     scatter_label=("Item age at sweep end" if incomplete_only else "Item time in system"))
    else:
        draw_line_chart(d_times, d_w, f"w(T) — average residence time in window (daily, {mode_label})", "w(T) [hrs]", dw)

    draw_step_chart(d_times, d_N, f"N(t) — active processes (daily, {mode_label})", "N(t)", dNp)
    written += [dL, dLam, dw, dNp]

    # Convergence diagnostics (daily)
    if len(d_times) > 0:
        d_conv = os.path.join(out_dir, 'daily_convergence.png')
        draw_convergence_panel(d_times, d_w, d_Lam, W_emp_d, lam_emp_d,
                               f'Convergence diagnostics (daily, {mode_label})', d_conv)
        written.append(d_conv)

        d_conv_dyn = os.path.join(out_dir, 'daily_convergence_dynamic.png')
        draw_dynamic_convergence_panel(d_times, d_w, d_Lam, W_star_d, lam_star_d,
                                       f'Dynamic convergence (daily, {mode_label})', d_conv_dyn)
        written.append(d_conv_dyn)

        d_conv_dyn3 = os.path.join(out_dir, 'daily_convergence_dynamic_errors.png')
        draw_dynamic_convergence_panel_with_errors(d_times, d_w, d_Lam, W_star_d, lam_star_d,
                                                   eW_d, eLam_d, epsilon,
                                                   f'Dynamic convergence + errors (daily, {mode_label})', d_conv_dyn3, lambda_pctl_upper=lambda_pctl_upper, lambda_pctl_lower=lambda_pctl_lower, lambda_warmup_hours=lambda_warmup_hours)
        written.append(d_conv_dyn3)

    # --- End-effect diagnostics ---
    rA_ts, rB_ts, rho_ts = compute_end_effect_series(df, t_times, t_A, W_star_ts) if len(t_times) > 0 else (np.array([]), np.array([]), np.array([]))
    rA_d,  rB_d,  rho_d  = compute_end_effect_series(df, d_times, d_A, W_star_d)  if len(d_times)  > 0 else (np.array([]), np.array([]), np.array([]))

    if len(t_times) > 0:
        ts_conv_dyn4 = os.path.join(out_dir, 'timestamp_convergence_dynamic_errors_endeffects.png')
        draw_dynamic_convergence_panel_with_errors_and_endeffects(
            t_times, t_w, t_Lam, W_star_ts, lam_star_ts, eW_ts, eLam_ts,
            rA_ts, rB_ts, rho_ts, epsilon,
            f'Dynamic convergence + errors + end-effects (timestamp, {mode_label})', ts_conv_dyn4, lambda_pctl_upper=lambda_pctl_upper, lambda_pctl_lower=lambda_pctl_lower, lambda_warmup_hours=lambda_warmup_hours)
        written.append(ts_conv_dyn4)

    if len(d_times) > 0:
        d_conv_dyn4 = os.path.join(out_dir, 'daily_convergence_dynamic_errors_endeffects.png')
        draw_dynamic_convergence_panel_with_errors_and_endeffects(
            d_times, d_w, d_Lam, W_star_d, lam_star_d, eW_d, eLam_d,
            rA_d, rB_d, rho_d, epsilon,
            f'Dynamic convergence + errors + end-effects (daily, {mode_label})', d_conv_dyn4, lambda_pctl_upper=lambda_pctl_upper, lambda_pctl_lower=lambda_pctl_lower, lambda_warmup_hours=lambda_warmup_hours)
        written.append(d_conv_dyn4)

    # Vertical stacks (4×1)
    col_ts = os.path.join(out_dir, 'timestamp_stack.png')
    draw_four_panel_column(t_times, t_N, t_L, t_Lam, t_w, f'Finite-window metrics (timestamp, {mode_label})', col_ts, lambda_pctl_upper, lambda_pctl_lower, lambda_warmup_hours)
    written.append(col_ts)

    col_d = os.path.join(out_dir, 'daily_stack.png')
    draw_four_panel_column(d_times, d_N, d_L, d_Lam, d_w, f'Finite-window metrics (daily, {mode_label})', col_d, lambda_pctl_upper, lambda_pctl_lower, lambda_warmup_hours)
    written.append(col_d)

    # 5-panel stacks including A(T)
    if with_A:
        col_ts5 = os.path.join(out_dir, 'timestamp_stack_with_A.png')
        draw_five_panel_column(t_times, t_N, t_L, t_Lam, t_w, t_A,
                               f'Finite-window metrics incl. A(T) (timestamp, {mode_label})', col_ts5,
                               scatter_times=t_scatter_times, scatter_values=t_scatter_vals, lambda_pctl_upper=lambda_pctl_upper, lambda_pctl_lower=lambda_pctl_lower, lambda_warmup_hours=lambda_warmup_hours)
        written.append(col_ts5)

        col_d5 = os.path.join(out_dir, 'daily_stack_with_A.png')
        draw_five_panel_column(d_times, d_N, d_L, d_Lam, d_w, d_A,
                               f'Finite-window metrics incl. A(T) (daily, {mode_label})', col_d5,
                               scatter_times=d_scatter_times, scatter_values=d_scatter_vals, lambda_pctl_upper=lambda_pctl_upper, lambda_pctl_lower=lambda_pctl_lower, lambda_warmup_hours=lambda_warmup_hours)
        written.append(col_d5)
    elif scatter:
        col_ts5s = os.path.join(out_dir, 'timestamp_stack_with_scatter.png')
        draw_five_panel_column_with_scatter(t_times, t_N, t_L, t_Lam, t_w,
                                            f'Finite-window metrics with w(T) plain + w(T)+scatter (timestamp, {mode_label})',
                                            col_ts5s,
                                            scatter_times=t_scatter_times, scatter_values=t_scatter_vals, lambda_pctl_upper=lambda_pctl_upper, lambda_pctl_lower=lambda_pctl_lower, lambda_warmup_hours=lambda_warmup_hours)
        written.append(col_ts5s)

        col_d5s = os.path.join(out_dir, 'daily_stack_with_scatter.png')
        draw_five_panel_column_with_scatter(d_times, d_N, d_L, d_Lam, d_w,
                                            f'Finite-window metrics with w(T) plain + w(T)+scatter (daily, {mode_label})',
                                            col_d5s,
                                            scatter_times=d_scatter_times, scatter_values=d_scatter_vals, lambda_pctl_upper=lambda_pctl_upper, lambda_pctl_lower=lambda_pctl_lower, lambda_warmup_hours=lambda_warmup_hours)
        written.append(col_d5s)

    # Optional daily breakdowns derived from A(T)
    if with_daily_breakdown:
        delta_A = np.empty(len(d_A)); delta_A[:] = np.nan
        if len(d_A) > 1:
            delta_A[1:] = d_A[1:] - d_A[:-1]
        delta_t_hours = np.empty(len(d_times)); delta_t_hours[:] = np.nan
        if len(d_times) > 1:
            delta_t_hours[1:] = [((d_times[i] - d_times[i-1]).total_seconds() / 3600.0) for i in range(1, len(d_times))]
        avg_WIP_daily = delta_A / delta_t_hours

        daily_wiphours_path = os.path.join(out_dir, 'daily_wip_hours.png')
        draw_bar_chart(d_times, delta_A, f'Daily WIP-hours ΔA (daily, {mode_label})', 'ΔA per day [hrs·items]', daily_wiphours_path)
        written.append(daily_wiphours_path)

        daily_avgwip_path = os.path.join(out_dir, 'daily_avg_WIP.png')
        draw_line_chart(d_times, avg_WIP_daily, f'Daily average WIP (daily, {mode_label})', 'Avg WIP per day', daily_avgwip_path)
        written.append(daily_avgwip_path)

    # Write coherence summary (and print)
    if coh_summary_lines:
        txt_path = os.path.join(out_dir, "coherence_summary.txt")
        with open(txt_path, "w") as f:
            for line in coh_summary_lines:
                f.write(line + "\n")
        print("\n".join(coh_summary_lines))
        written.append(txt_path)

    return written


def main():
    args = cli.parse_args()
    try:
        paths = produce_all_charts(
            args.csv,
            args,
            completed_only=args.completed,
            incomplete_only=args.incomplete,
            with_A=args.with_A,
            with_daily_breakdown=args.with_daily_breakdown,
            scatter=args.scatter,
            epsilon=args.epsilon,
            horizon_days=args.horizon_days,
            lambda_pctl_upper=args.lambda_pctl,
            lambda_pctl_lower=args.lambda_lower_pctl,
            lambda_warmup_hours=float(args.lambda_warmup)
        )
        print("Wrote charts:\n" + "\n".join(paths))
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
