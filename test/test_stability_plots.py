# -*- coding: utf-8 -*-
# Copyright (c) 2025 Krishna Kumar
# SPDX-License-Identifier: MIT
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd

from samplepath.plots import stability


def test_plot_rate_stability_charts_calls_render_N_panel():
    times = [pd.Timestamp("2024-01-01"), pd.Timestamp("2024-01-02")]
    metrics = SimpleNamespace(
        times=times,
        N=np.array([1.0, 2.0]),
        t0=times[0],
        w=np.array([0.5, 0.6]),
    )
    df = pd.DataFrame({"start_ts": times, "end_ts": times})
    args = SimpleNamespace(lambda_pctl=None, lambda_lower_pctl=None, lambda_warmup=0.0)
    filter_result = SimpleNamespace(title_prefix=None, display=None)
    fig1 = MagicMock()
    fig2 = MagicMock()
    fig3 = MagicMock()
    axes1 = [MagicMock(), MagicMock()]
    axes2 = [MagicMock(), MagicMock()]
    axes3 = [MagicMock() for _ in range(4)]

    with (
        patch(
            "samplepath.plots.stability.compute_total_active_age_series",
            return_value=np.array([0.0, 1.0]),
        ),
        patch(
            "samplepath.plots.stability.compute_elementwise_empirical_metrics",
            return_value=SimpleNamespace(
                as_tuple=lambda: (np.array([1.0, 1.0]), np.array([1.0, 1.0]))
            ),
        ),
        patch("samplepath.plots.stability._clip_axis_to_percentile"),
        patch(
            "samplepath.plots.stability.plt.subplots",
            side_effect=[
                (fig1, axes1),
                (fig2, axes2),
                (fig3, axes3),
            ],
        ),
        patch("samplepath.plots.stability.plt.close"),
        patch("samplepath.plots.stability.NPanel.render") as mock_render,
    ):
        stability.plot_rate_stability_charts(
            df, args, filter_result, metrics, out_dir="/tmp/out"
        )
    assert mock_render.call_count == 1
