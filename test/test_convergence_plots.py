# -*- coding: utf-8 -*-
# Copyright (c) 2025 Krishna Kumar
# SPDX-License-Identifier: MIT
from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd

from samplepath.metrics import ElementWiseEmpiricalMetrics
from samplepath.plots.chart_config import ChartConfig
from samplepath.plots.convergence import (
    ProcessTimeConvergencePanel,
    draw_arrival_departure_convergence_stack,
    plot_convergence_charts,
)


def _t(s: str) -> pd.Timestamp:
    return pd.Timestamp(s)


def test_draw_arrival_departure_convergence_stack_uses_CFDPanel_render():
    times = [_t("2024-01-01"), _t("2024-01-02")]
    arrivals = np.array([1.0, 2.0])
    departures = np.array([0.0, 1.0])
    lambda_rate = np.array([0.5, 0.75])

    with (
        patch("samplepath.plots.convergence.CFDPanel.render") as mock_render,
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


def test_process_time_convergence_panel_overlays_drop_lines():
    ax = MagicMock()
    times = [_t("2024-01-01")]
    w_vals = np.array([1.0])
    w_prime_vals = np.array([2.0])
    w_star_vals = np.array([3.0])
    arrivals = [times[0]]
    departures = [times[0]]
    with (
        patch(
            "samplepath.plots.convergence.build_event_overlays",
            return_value=["overlay"],
        ) as mock_overlays,
        patch("samplepath.plots.convergence.render_line_chart"),
    ):
        ProcessTimeConvergencePanel(with_event_marks=True).render(
            ax,
            times,
            w_vals,
            w_prime_vals,
            w_star_vals,
            arrival_times=arrivals,
            departure_times=departures,
        )
    assert mock_overlays.call_count == 3
    first_args, first_kwargs = (
        mock_overlays.call_args_list[0].args,
        mock_overlays.call_args_list[0].kwargs,
    )
    assert first_args[2] == arrivals
    assert first_args[3] == []
    assert first_kwargs["drop_lines_for_arrivals"] is True
    assert first_kwargs["drop_lines_for_departures"] is False
    second_args, second_kwargs = (
        mock_overlays.call_args_list[1].args,
        mock_overlays.call_args_list[1].kwargs,
    )
    assert second_args[2] == []
    assert second_args[3] == departures
    assert second_kwargs["drop_lines_for_arrivals"] is False
    assert second_kwargs["drop_lines_for_departures"] is True
    third_args, third_kwargs = (
        mock_overlays.call_args_list[2].args,
        mock_overlays.call_args_list[2].kwargs,
    )
    assert third_args[2] == []
    assert third_args[3] == departures
    assert third_kwargs["drop_lines_for_arrivals"] is False
    assert third_kwargs["drop_lines_for_departures"] is True


def test_process_time_convergence_panel_plot_calls_renderer():
    fig = MagicMock()
    ax = MagicMock()
    metrics = SimpleNamespace(
        times=[_t("2024-01-01")],
        w=np.array([1.0]),
        w_prime=np.array([2.0]),
        arrival_times=[_t("2024-01-01")],
        departure_times=[_t("2024-01-01")],
        freq="D",
    )
    empirical_metrics = ElementWiseEmpiricalMetrics(
        times=metrics.times, W_star=np.array([3.0]), lam_star=np.array([0.5])
    )
    filter_result = SimpleNamespace(display="Filters: test", label="test")
    chart_config = ChartConfig()

    @contextmanager
    def fake_context(out_path=None, **kwargs):
        assert out_path is None
        assert kwargs["out_dir"] == "/tmp/out"
        assert kwargs["subdir"] == "convergence/panels"
        assert kwargs["base_name"] == "process_time_convergence"
        yield fig, ax, "out.png"

    with (
        patch("samplepath.plots.convergence.figure_context", side_effect=fake_context),
        patch(
            "samplepath.plots.convergence.ProcessTimeConvergencePanel.render"
        ) as mock_render,
    ):
        written = ProcessTimeConvergencePanel().plot(
            metrics, empirical_metrics, filter_result, chart_config, "/tmp/out"
        )
    assert written == "out.png"
    mock_render.assert_called_once()


def test_plot_convergence_charts_includes_process_time_panel():
    df = pd.DataFrame()
    args = SimpleNamespace()
    metrics = SimpleNamespace(
        times=[_t("2024-01-01")],
        w=np.array([1.0]),
        w_prime=np.array([2.0]),
        arrival_times=[],
        departure_times=[],
        freq=None,
    )
    empirical_metrics = ElementWiseEmpiricalMetrics(
        times=metrics.times, W_star=np.array([3.0]), lam_star=np.array([0.5])
    )
    filter_result = SimpleNamespace(display="Filters: test", label="test")
    with (
        patch(
            "samplepath.plots.convergence.plot_arrival_rate_convergence",
            return_value=[],
        ),
        patch(
            "samplepath.plots.convergence.plot_sojourn_time_scatter", return_value=[]
        ),
        patch(
            "samplepath.plots.convergence.plot_residence_time_sojourn_time_coherence_charts",
            return_value=[],
        ),
        patch(
            "samplepath.plots.convergence.plot_residence_vs_sojourn_stack",
            return_value=[],
        ),
        patch(
            "samplepath.plots.convergence.plot_sample_path_convergence", return_value=[]
        ),
        patch(
            "samplepath.plots.convergence.ProcessTimeConvergencePanel.plot",
            return_value="process.png",
        ) as mock_plot,
    ):
        written = plot_convergence_charts(
            df, args, filter_result, metrics, empirical_metrics, "/tmp/out"
        )
    assert "process.png" in written
    assert isinstance(mock_plot.call_args.args[3], ChartConfig)
