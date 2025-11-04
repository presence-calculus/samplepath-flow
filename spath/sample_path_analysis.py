# -*- coding: utf-8 -*-
# Copyright (c) 2025 Krishna Kumar
# SPDX-License-Identifier: MIT

"""
Finite-window flow metrics & convergence diagnostics with end-effect panel.
See README.md for context.
"""
from __future__ import annotations

import sys
from argparse import Namespace
from typing import List, Tuple

import pandas as pd

import cli
from csv_loader import csv_to_dataframe
from file_utils import ensure_output_dirs, write_cli_args_to_file, copy_input_csv_to_output
from filter import FilterResult, apply_filters
from metrics import compute_finite_window_flow_metrics, FlowMetricsResult
from point_process import to_arrival_departure_process
from spath.limits import write_limits
from spath.metrics import ElementWiseEmpiricalMetrics, compute_elementwise_empirical_metrics
from spath.plots.advanced import plot_advanced_charts
from spath.plots.convergence import plot_convergence_charts
from spath.plots.core import plot_core_flow_metrics_charts
from spath.plots.misc import plot_misc_charts
from spath.plots.stability import plot_stability_charts


def produce_all_charts(df,  args, filter_result, metrics, empirical_metrics, out_dir):
    written: List[str] = []
    # create plots
    written += plot_core_flow_metrics_charts(df, args, filter_result, metrics, out_dir)
    written += plot_convergence_charts(df, args, filter_result, metrics, empirical_metrics, out_dir)
    written += plot_stability_charts(df, args, filter_result, metrics, out_dir)
    written += plot_advanced_charts(df, args, filter_result, metrics, out_dir)
    written += plot_misc_charts(df, args, filter_result, metrics, out_dir)
    return written

# -------------------------------
# Orchestration
# -------------------------------
def run_analysis(csv_path: str, args: Namespace, out_dir: str) -> List[str]:
    df = csv_to_dataframe(csv_path, args=args)
    filter_result: FilterResult = apply_filters(df, args)
    df = filter_result.df
    # Build arrival departure process
    arrival_departure_process: List[Tuple[pd.Timestamp, int, int]] = to_arrival_departure_process(df)
    # Compute core finite window flow metrics
    metrics: FlowMetricsResult = compute_finite_window_flow_metrics(arrival_departure_process)

    # Compute  ElementWiseMetrics once
    empirical_metrics: ElementWiseEmpiricalMetrics = compute_elementwise_empirical_metrics(df, metrics.times)

    write_limits(metrics, empirical_metrics, out_dir)
    return produce_all_charts(df, args, filter_result, metrics, empirical_metrics, out_dir)


def main():
    parser, args = cli.parse_args()
    out_dir = ensure_output_dirs(args.csv, output_dir=args.output_dir, scenario_dir=args.scenario,  clean=args.clean)
    if args.save_input:
        copy_input_csv_to_output(args.csv, out_dir)

    write_cli_args_to_file(parser, args, out_dir)
    try:
        paths = run_analysis(
            args.csv,
            args,
            out_dir
        )
        print("Wrote charts:\n" + "\n".join(paths))
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()



