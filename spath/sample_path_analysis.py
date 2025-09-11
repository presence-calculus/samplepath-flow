# -*- coding: utf-8 -*-
# Copyright (c) 2025 Krishna Kumar
# SPDX-License-Identifier: MIT

"""
Finite-window flow metrics & convergence diagnostics with end-effect panel.
See README.md for context.
"""

import sys
from typing import List, Optional, Tuple
from argparse import Namespace

import pandas as pd

import cli
from csv_loader import csv_to_dataframe
from filter import FilterResult, apply_filters
from metrics import compute_finite_window_flow_metrics, FlowMetricsResult
from point_process import to_arrival_departure_process
from spath.plots import plot_core_flow_metrics, plot_sojourn_time_scatter, ensure_output_dir, plot_coherence_charts, \
    plot_core_metrics_stack, \
    plot_five_column_stacks, plot_rate_stability_charts, plot_llaw_manifold_3d


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

    # Build arrival departure process
    arrival_departure_process: List[Tuple[pd.Timestamp, int, int]] = to_arrival_departure_process(df)
    # Compute core finite window flow metrics
    metrics: FlowMetricsResult = compute_finite_window_flow_metrics(arrival_departure_process)

    out_dir = ensure_output_dir(csv_path)
    written: List[str] = []

    # create plots
    written += plot_core_flow_metrics(df, args, filter_result, metrics, out_dir)
    # Vertical stacks (4×1)
    written += plot_core_metrics_stack(args, filter_result, metrics, out_dir)

    if scatter:
        written += plot_sojourn_time_scatter(args, df, filter_result, metrics, out_dir)

        # 5-panel stacks including scatter
    plot_five_column_stacks(df, args, filter_result, metrics, out_dir)

    written += plot_coherence_charts(df, args, filter_result, metrics, out_dir)

    written += plot_rate_stability_charts(df, args, filter_result, metrics, out_dir)

    written += plot_llaw_manifold_3d(df, metrics, out_dir)
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


