# -*- coding: utf-8 -*-
# Copyright (c) 2025 Krishna Kumar
# SPDX-License-Identifier: MIT

"""
Finite-window flow metrics & convergence diagnostics with end-effect panel.
See README.md for context.
"""
from __future__ import annotations

from argparse import Namespace
from typing import List, Tuple

import pandas as pd

from .csv_loader import csv_to_dataframe
from .filter import FilterResult, apply_filters
from .limits import write_limits
from .metrics import (
    ElementWiseEmpiricalMetrics,
    FlowMetricsResult,
    compute_elementwise_empirical_metrics,
    compute_finite_window_flow_metrics,
)
from .plots import (
    plot_advanced_charts,
    plot_convergence_charts,
    plot_core_flow_metrics_charts,
    plot_misc_charts,
    plot_stability_charts,
)
from .point_process import to_arrival_departure_process


def produce_all_charts(df, args, filter_result, metrics, empirical_metrics, out_dir):
    written: List[str] = []
    # create plots
    written += plot_core_flow_metrics_charts(df, args, filter_result, metrics, out_dir)
    written += plot_convergence_charts(
        df, args, filter_result, metrics, empirical_metrics, out_dir
    )
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
    arrival_departure_process: List[Tuple[pd.Timestamp, int, int]] = (
        to_arrival_departure_process(df)
    )
    # Compute core finite window flow metrics
    metrics: FlowMetricsResult = compute_finite_window_flow_metrics(
        arrival_departure_process
    )

    # Compute  ElementWiseMetrics once
    empirical_metrics: ElementWiseEmpiricalMetrics = (
        compute_elementwise_empirical_metrics(df, metrics.times)
    )

    write_limits(metrics, empirical_metrics, out_dir)
    return produce_all_charts(
        df, args, filter_result, metrics, empirical_metrics, out_dir
    )
