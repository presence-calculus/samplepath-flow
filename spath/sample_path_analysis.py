# -*- coding: utf-8 -*-
# Copyright (c) 2025 Krishna Kumar
# SPDX-License-Identifier: MIT

"""
Finite-window flow metrics & convergence diagnostics with end-effect panel.
See README.md for context.
"""

import sys
from argparse import Namespace
from typing import List, Tuple

import pandas as pd

import cli
from csv_loader import csv_to_dataframe
from filter import FilterResult, apply_filters
from metrics import compute_finite_window_flow_metrics, FlowMetricsResult
from point_process import to_arrival_departure_process
from spath.metrics import ElementWiseEmpiricalMetrics, compute_elementwise_empirical_metrics
from spath.plots import produce_all_charts


# -------------------------------
# Orchestration
# -------------------------------
def run_analysis(csv_path: str, args: Namespace) -> List[str]:
    df = csv_to_dataframe(csv_path, args=args)
    filter_result: FilterResult = apply_filters(df, args)
    df = filter_result.df
    # Build arrival departure process
    arrival_departure_process: List[Tuple[pd.Timestamp, int, int]] = to_arrival_departure_process(df)
    # Compute core finite window flow metrics
    metrics: FlowMetricsResult = compute_finite_window_flow_metrics(arrival_departure_process)

    # Compute  ElementWiseMetrics once
    empirical_metrics: ElementWiseEmpiricalMetrics = compute_elementwise_empirical_metrics(df, metrics.times)

    return produce_all_charts(df, csv_path, args, filter_result, metrics, empirical_metrics)


def main():
    args = cli.parse_args()
    try:
        paths = run_analysis(
            args.csv,
            args
        )
        print("Wrote charts:\n" + "\n".join(paths))
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()


