# -*- coding: utf-8 -*-
# Copyright (c) 2025 Krishna Kumar
# SPDX-License-Identifier: MIT
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd

from samplepath.plots.convergence import draw_arrival_departure_convergence_stack


def _t(s: str) -> pd.Timestamp:
    return pd.Timestamp(s)


def test_draw_arrival_departure_convergence_stack_uses_render_CFD():
    times = [_t("2024-01-01"), _t("2024-01-02")]
    arrivals = np.array([1.0, 2.0])
    departures = np.array([0.0, 1.0])
    lambda_rate = np.array([0.5, 0.75])

    with (
        patch("samplepath.plots.convergence.render_CFD") as mock_render,
        patch("samplepath.plots.convergence.plt.subplots") as mock_subplots,
        patch("samplepath.plots.convergence.format_date_axis"),
        patch("samplepath.plots.convergence._clip_axis_to_percentile"),
    ):
        fig = MagicMock()
        axes = np.empty(2, dtype=object)
        axes[:] = [MagicMock(), MagicMock()]
        mock_subplots.return_value = (fig, axes)
        draw_arrival_departure_convergence_stack(
            times,
            arrivals,
            departures,
            lambda_rate,
            "Title",
            "out.png",
        )

    mock_render.assert_called_once_with(axes[0], times, arrivals, departures)
