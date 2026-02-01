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
from .data_export import export_data
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
    plot_stability_charts,
)
from .plots.chart_config import ChartConfig
from .point_process import to_arrival_departure_process
from .utils.duration_scale import infer_duration_scale
from .utils.file_utils import ensure_export_dir


def produce_all_charts(df, args, filter_result, metrics, empirical_metrics, out_dir):
    written: List[str] = []
    # create plots
    chart_config = ChartConfig.init_from_args(args)
    if chart_config.duration_scale is None:
        chart_config = chart_config.with_duration_scale(
            infer_duration_scale(pd.Series(metrics.w, dtype=float).to_numpy())
        )
    written += plot_core_flow_metrics_charts(
        metrics, empirical_metrics, filter_result, chart_config, out_dir
    )
    written += plot_convergence_charts(
        metrics, empirical_metrics, filter_result, chart_config, out_dir
    )
    written += plot_stability_charts(df, chart_config, filter_result, metrics, out_dir)
    written += plot_advanced_charts(
        df, args, chart_config, filter_result, metrics, out_dir
    )
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
    anchor = getattr(args, "anchor", None)
    anchor_kwargs = {}
    if anchor is not None:
        anchor_kwargs["week_anchor"] = anchor
        anchor_kwargs["quarter_anchor"] = anchor
        anchor_kwargs["year_anchor"] = anchor
    metrics: FlowMetricsResult = compute_finite_window_flow_metrics(
        arrival_departure_process,
        freq=args.sampling_frequency,
        **anchor_kwargs,
    )

    # Compute  ElementWiseMetrics once
    empirical_metrics: ElementWiseEmpiricalMetrics = (
        compute_elementwise_empirical_metrics(df, metrics.times)
    )

    write_limits(metrics, empirical_metrics, out_dir)

    # Handle data export
    written: List[str] = []
    export_data_flag = getattr(args, "export_data", False)
    export_only_flag = getattr(args, "export_only", False)

    if export_data_flag or export_only_flag:
        export_dir = ensure_export_dir(out_dir)
        sampling_frequency = getattr(args, "sampling_frequency", None)
        export_paths = export_data(
            df, metrics, empirical_metrics, export_dir, sampling_frequency
        )
        written.extend(export_paths)

    # Generate charts unless export_only is set
    if not export_only_flag:
        chart_paths = produce_all_charts(
            df, args, filter_result, metrics, empirical_metrics, out_dir
        )
        written.extend(chart_paths)

    return written
