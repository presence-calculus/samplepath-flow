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

import cli
from csv_loader import csv_to_dataframe
from filter import FilterResult, apply_filters
from metrics import compute_sample_path_metrics, compute_finite_window_flow_metrics, compute_end_effect_series, \
    compute_empirical_targets, compute_dynamic_empirical_series, compute_tracking_errors, compute_coherence_score
from plots import draw_line_chart, draw_lambda_chart, draw_line_chart_with_scatter, draw_step_chart, \
    draw_four_panel_column, draw_five_panel_column, draw_five_panel_column_with_scatter, draw_convergence_panel, \
    draw_dynamic_convergence_panel, draw_dynamic_convergence_panel_with_errors, \
    draw_dynamic_convergence_panel_with_errors_and_endeffects
from point_process import build_arrival_departure_events


# -------------------------------
# Plot helpers
# -------------------------------


# === NEW: End-effect metrics ===


# -------------------------------
# Core computations
# -------------------------------


def ensure_output_dir(csv_path: str) -> str:
    base = os.path.basename(csv_path)
    stem = os.path.splitext(base)[0]
    out_dir = os.path.join("charts", stem)
    os.makedirs(out_dir, exist_ok=True)
    return out_dir


def sweep_timestamp_series(events: List[Tuple[pd.Timestamp, int, int]]):
    unique_times: List[pd.Timestamp] = sorted({t for t, _, _ in events})
    return compute_sample_path_metrics(events, unique_times)


# -------------------------------
# Coherence diagnostics
# -------------------------------


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
    mode_label = filter_result.label

    # Build events and sweeps
    events = build_arrival_departure_events(df)
    # Compute core finite window flow metrics
    metrics = compute_finite_window_flow_metrics(events)

    t_times, t_L, t_Lam, t_w, t_N, t_A = (
        metrics.times,
        metrics.L,
        metrics.Lambda,
        metrics.w,
        metrics.N,
        metrics.A
    )

    out_dir = ensure_output_dir(csv_path)
    written: List[str] = []

    # Timestamp charts
    ts_L = os.path.join(out_dir, "timestamp_L.png")
    ts_Lam = os.path.join(out_dir, "timestamp_Lambda.png")
    ts_w = os.path.join(out_dir, "timestamp_w.png")
    ts_Np = os.path.join(out_dir, "timestamp_N.png")

    draw_line_chart(t_times, t_L, f"L(T) — time-average number (timestamp, {mode_label})", "L(T)", ts_L)
    draw_lambda_chart(t_times, t_Lam, f"Λ(T) — cumulative arrivals per hour (timestamp, {mode_label})", "Λ(T) [1/hr]",
                      ts_Lam, lambda_pctl_upper, lambda_pctl_lower, lambda_warmup_hours)
    draw_line_chart(t_times, t_w, f"w(T) — average residence time in window (timestamp, {mode_label})",
                    "w(T) [hrs]", ts_w)

    draw_step_chart(t_times, t_N, f"N(t) — active processes (timestamp, {mode_label})", "N(t)", ts_Np)
    written += [ts_L, ts_Lam, ts_w, ts_Np]


    # Empirical targets & dynamic baselines
    if len(t_times) > 0:
        W_emp_ts, lam_emp_ts = compute_empirical_targets(df, t_times[0], t_times[-1])
        W_star_ts, lam_star_ts = compute_dynamic_empirical_series(df, t_times)
    else:
        W_emp_ts = lam_emp_ts = float('nan')
        W_star_ts = lam_star_ts = np.array([])

    # Relative errors & coherence
    eW_ts, eLam_ts, elapsed_ts = compute_tracking_errors(t_times, t_w, t_Lam, W_star_ts, lam_star_ts)
    coh_summary_lines: List[str] = []
    if epsilon is not None and horizon_days is not None:
        h_hrs = float(horizon_days) * 24.0
        sc_ts, ok_ts, tot_ts = compute_coherence_score(eW_ts, eLam_ts, elapsed_ts, float(epsilon), h_hrs)
        coh_summary_lines.append(f"Coherence (timestamp): eps={epsilon:g}, H={horizon_days:g}d -> {ok_ts}/{tot_ts} ({(sc_ts*100 if sc_ts==sc_ts else 0):.1f}%)")

    # Scatter arrays
    t_scatter_times: List[pd.Timestamp] = []
    t_scatter_vals = np.array([])
    if scatter:
        if incomplete_only:
            if len(t_times) > 0:
                t_end = t_times[-1]
                t_scatter_times = df["start_ts"].tolist()
                t_scatter_vals = ((t_end - df["start_ts"]).dt.total_seconds() / 3600.0).to_numpy()
        else:
            df_c = df[df["end_ts"].notna()].copy()
            if not df_c.empty:
                t_scatter_times = df_c["end_ts"].tolist()
                t_scatter_vals = ((df_c["end_ts"] - df_c["start_ts"]).dt.total_seconds() / 3600.0).to_numpy()


    if scatter and len(t_scatter_times) > 0:
        label = "Item age at sweep end" if incomplete_only else "Item time in system"
        draw_line_chart_with_scatter(t_times, t_w,
                                     f"w(T) — average residence time in window (timestamp, {mode_label})",
                                     "w(T) [hrs]", ts_w, t_scatter_times, t_scatter_vals, scatter_label=label)


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



    # --- End-effect diagnostics ---
    rA_ts, rB_ts, rho_ts = compute_end_effect_series(df, t_times, t_A, W_star_ts) if len(t_times) > 0 else (np.array([]), np.array([]), np.array([]))

    if len(t_times) > 0:
        ts_conv_dyn4 = os.path.join(out_dir, 'timestamp_convergence_dynamic_errors_endeffects.png')
        draw_dynamic_convergence_panel_with_errors_and_endeffects(
            t_times, t_w, t_Lam, W_star_ts, lam_star_ts, eW_ts, eLam_ts,
            rA_ts, rB_ts, rho_ts, epsilon,
            f'Dynamic convergence + errors + end-effects (timestamp, {mode_label})', ts_conv_dyn4, lambda_pctl_upper=lambda_pctl_upper, lambda_pctl_lower=lambda_pctl_lower, lambda_warmup_hours=lambda_warmup_hours)
        written.append(ts_conv_dyn4)


    # Vertical stacks (4×1)
    col_ts = os.path.join(out_dir, 'timestamp_stack.png')
    draw_four_panel_column(t_times, t_N, t_L, t_Lam, t_w, f'Finite-window metrics (timestamp, {mode_label})', col_ts, lambda_pctl_upper, lambda_pctl_lower, lambda_warmup_hours)
    written.append(col_ts)


    # 5-panel stacks including A(T)
    if with_A:
        col_ts5 = os.path.join(out_dir, 'timestamp_stack_with_A.png')
        draw_five_panel_column(t_times, t_N, t_L, t_Lam, t_w, t_A,
                               f'Finite-window metrics incl. A(T) (timestamp, {mode_label})', col_ts5,
                               scatter_times=t_scatter_times, scatter_values=t_scatter_vals, lambda_pctl_upper=lambda_pctl_upper, lambda_pctl_lower=lambda_pctl_lower, lambda_warmup_hours=lambda_warmup_hours)
        written.append(col_ts5)

    elif scatter:
        col_ts5s = os.path.join(out_dir, 'timestamp_stack_with_scatter.png')
        draw_five_panel_column_with_scatter(t_times, t_N, t_L, t_Lam, t_w,
                                            f'Finite-window metrics with w(T) plain + w(T)+scatter (timestamp, {mode_label})',
                                            col_ts5s,
                                            scatter_times=t_scatter_times, scatter_values=t_scatter_vals, lambda_pctl_upper=lambda_pctl_upper, lambda_pctl_lower=lambda_pctl_lower, lambda_warmup_hours=lambda_warmup_hours)
        written.append(col_ts5s)

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
