# -*- coding: utf-8 -*-
# Copyright (c) 2025 Krishna Kumar
# SPDX-License-Identifier: MIT
from __future__ import annotations

import os
from typing import List, Optional

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from samplepath.filter import FilterResult
from samplepath.metrics import compute_elementwise_empirical_metrics, FlowMetricsResult
from samplepath.plots.helpers import format_date_axis, _clip_axis_to_percentile, add_caption


def compute_total_active_age_series(
    df: pd.DataFrame,
    times: List[pd.Timestamp]
) -> np.ndarray:
    """
    Return R(T) aligned to `times`: total age (HOURS) of ACTIVE elements at T.

    Numerically safe: all prefix sums are done in time **relative to t0** and
    in float64 HOURS (not ns) to avoid int64 overflows.

    active(T): start <= T and (end > T or end is NaT)
    window clip: ages measured from s' = max(start, t0), so R(t0) = 0.
    """
    n = len(times)
    R = np.zeros(n, dtype=float)
    if n == 0:
        return R

    # Ensure ascending times
    T_seq: List[pd.Timestamp] = [pd.Timestamp(t) for t in times]
    t0 = T_seq[0]
    t0_ns = pd.Timestamp(t0).value  # int64 (ns since epoch)

    # ----------------- Prepare start/end arrays -----------------
    # Starts (absolute for comparisons; sorted)
    starts_dt = pd.to_datetime(df["start_ts"]).sort_values().to_numpy(dtype="datetime64[ns]")

    # Ends: only completed items; keep their corresponding starts for subtraction
    ended = df[df["end_ts"].notna()].copy()
    ended.sort_values("end_ts", inplace=True)
    ends_dt = pd.to_datetime(ended["end_ts"]).to_numpy(dtype="datetime64[ns]")
    ended_starts_dt = pd.to_datetime(ended["start_ts"]).to_numpy(dtype="datetime64[ns]")

    # ----------------- Relative-to-t0 in HOURS (float64) -----------------
    # Convert to ns int for relative deltas, then to hours
    starts_ns = starts_dt.astype("int64")
    starts_rel_h = np.maximum((starts_ns - t0_ns) / 3.6e12, 0.0).astype(np.float64)
    starts_rel_cumsum_h = np.cumsum(starts_rel_h, dtype=np.float64)

    ended_starts_ns = ended_starts_dt.astype("int64")
    ended_starts_rel_h = np.maximum((ended_starts_ns - t0_ns) / 3.6e12, 0.0).astype(np.float64)
    ended_starts_rel_cumsum_h = np.cumsum(ended_starts_rel_h, dtype=np.float64)

    # Pointers over sorted arrays
    S = starts_dt.size
    E = ends_dt.size
    i_s = 0  # count of starts with start <= T
    i_e = 0  # count of ends   with end   <= T

    # ----------------- Main sweep -----------------
    for i, T in enumerate(T_seq):
        T_dt_ns = np.datetime64(T).astype("datetime64[ns]")

        # Advance pointers monotonically
        while i_s < S and starts_dt[i_s] <= T_dt_ns:
            i_s += 1
        while i_e < E and ends_dt[i_e] <= T_dt_ns:
            i_e += 1

        N_active = i_s - i_e
        if N_active <= 0:
            R[i] = 0.0
            continue

        # Sum of clipped (relative) starts up to T, in HOURS
        S_le_T_rel_h = starts_rel_cumsum_h[i_s - 1] if i_s > 0 else 0.0
        S_le_T_ended_rel_h = ended_starts_rel_cumsum_h[i_e - 1] if i_e > 0 else 0.0
        S_active_rel_h = max(S_le_T_rel_h - S_le_T_ended_rel_h, 0.0)

        # R_h = N_active * (T - t0) [hours] - sum_active_clipped_starts [hours]
        T_rel_h = (pd.Timestamp(T).value - t0_ns) / 3.6e12  # ns → hours
        R_h = float(N_active) * T_rel_h - S_active_rel_h

        # Numerical safety: never negative
        R[i] = max(R_h, 0.0)

    return R


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
    format_date_axis(ax_top)

    # Bottom: WIP Growth Rate N(T)/T
    ax = axes[1]
    ax.plot(times, N_over_T, label='N(t)/T', linewidth=1.9, zorder=3)
    ax.axhline(0.0, linewidth=0.8, alpha=0.6, zorder=1)
    ax.axhline(1.0, linewidth=1.0, alpha=1.0, linestyle=':', zorder=1)
    format_date_axis(ax)
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
    format_date_axis(ax_top)

    # Bottom: Total Age Growth Rate R(T)/T
    ax = axes[1]
    ax.plot(times, R_over_T, label="R(T)/T", linewidth=1.9, zorder=3)
    ax.axhline(0.0, linewidth=0.8, alpha=0.6, zorder=1)
    ax.axhline(1.0, linewidth=1.0, alpha=1.0, linestyle=":", zorder=1)  # reference guide

    format_date_axis(ax)
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
    format_date_axis(axN)
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
    format_date_axis(axR)
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
    format_date_axis(axLam)
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
    format_date_axis(axW)
    axW.set_xlabel("time")
    axW.set_ylabel("hours")
    axW.set_title("w(T) vs W*(T) — coherence")
    axW.legend(loc="best")

    # Subtitle + caption
    fig.suptitle("Equilibrium and Coherence", fontsize=14, y=0.98)
    try:
        if caption_text:
            add_caption(fig, caption_text)
    except Exception:
        pass

    fig.tight_layout(rect=(0, 0.06, 1, 0.96))
    fig.savefig(out_path_stack, dpi=200)
    plt.close(fig)
    written.append(out_path_stack)

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
