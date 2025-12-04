# -*- coding: utf-8 -*-
# Copyright (c) 2025 Krishna Kumar
# SPDX-License-Identifier: MIT

from __future__ import annotations

import os
from typing import List, Optional

import pandas as pd

from samplepath.filter import FilterResult
from samplepath.metrics import FlowMetricsResult
from samplepath.plots.core import (
    draw_five_panel_column,
    draw_five_panel_column_with_scatter,
)


def plot_five_column_stacks(df, args, filter_result, metrics, out_dir):
    t_scatter_times = df["start_ts"].tolist()
    t_scatter_vals = df["duration_hr"].to_numpy()
    written = []

    col_ts5 = os.path.join(out_dir, "misc/timestamp_stack_with_A.png")
    draw_five_panel_column(
        metrics.times,
        metrics.N,
        metrics.Lambda,
        metrics.Lambda,
        metrics.w,
        metrics.A,
        f"Finite-window metrics incl. A(T) (timestamp, {filter_result.label})",
        col_ts5,
        scatter_times=t_scatter_times,
        scatter_values=t_scatter_vals,
        lambda_pctl_upper=args.lambda_pctl,
        lambda_pctl_lower=args.lambda_lower_pctl,
        lambda_warmup_hours=args.lambda_warmup,
    )
    written.append(col_ts5)

    col_ts5s = os.path.join(out_dir, "misc/timestamp_stack_with_scatter.png")
    draw_five_panel_column_with_scatter(
        metrics.times,
        metrics.N,
        metrics.L,
        metrics.Lambda,
        metrics.w,
        f"Finite-window metrics with w(T) plain + w(T)+scatter (timestamp, {filter_result.label})",
        col_ts5s,
        scatter_times=t_scatter_times,
        scatter_values=t_scatter_vals,
        lambda_pctl_upper=args.lambda_pctl,
        lambda_pctl_lower=args.lambda_lower_pctl,
        lambda_warmup_hours=args.lambda_warmup,
    )
    written.append(col_ts5s)

    return written


def plot_misc_charts(
    df: pd.DataFrame,
    args,
    filter_result: Optional[FilterResult],
    metrics: FlowMetricsResult,
    out_dir: str,
) -> List[str]:
    # 5-panel stacks including scatter
    return plot_five_column_stacks(df, args, filter_result, metrics, out_dir)
