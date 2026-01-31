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
    ArrivalDepartureRateConvergencePanel,
    ProcessTimeConvergencePanel,
    SamplePathConvergencePanel,
    SojournTimeScatterPanel,
    plot_arrival_departure_equilibrium_stack,
    plot_convergence_charts,
    plot_process_time_convergence_stack,
)
from samplepath.utils.duration_scale import MINUTES


def _t(s: str) -> pd.Timestamp:
    return pd.Timestamp(s)


def test_arrival_departure_rate_panel_clips_lambda():
    ax = MagicMock()
    times = [_t("2024-01-01"), _t("2024-01-02")]
    departures = np.array([0.0, 1.0])
    lambda_rate = np.array([0.5, 0.75])
    with (
        patch("samplepath.plots.convergence.render_line_chart"),
        patch("samplepath.plots.convergence._clip_axis_to_percentile") as mock_clip,
    ):
        ArrivalDepartureRateConvergencePanel().render(
            ax,
            times,
            departures,
            lambda_rate,
            lambda_pctl_upper=99.0,
            lambda_pctl_lower=1.0,
            lambda_warmup_seconds=1800.0,
        )
    mock_clip.assert_called_once()


def _call_plot_convergence_charts_with_rate_panels():
    metrics = SimpleNamespace(
        times=[_t("2024-01-01")],
        Arrivals=np.array([1.0]),
        Departures=np.array([0.0]),
        Lambda=np.array([0.5]),
        freq="D",
    )
    empirical_metrics = ElementWiseEmpiricalMetrics(
        times=metrics.times,
        W_star=np.array([1.0]),
        lam_star=np.array([0.5]),
        sojourn_vals=np.array([1.0]),
        residence_time_vals=np.array([1.0]),
        residence_completed=np.array([True]),
    )
    filter_result = SimpleNamespace(display="Filters: test", label="test")
    chart_config = ChartConfig()

    with (
        patch(
            "samplepath.plots.convergence.CumulativeArrivalRateConvergencePanel.plot",
            return_value="lambda.png",
        ) as mock_lambda,
        patch(
            "samplepath.plots.convergence.ArrivalDepartureRateConvergencePanel.plot",
            return_value="rate.png",
        ) as mock_rate,
        patch(
            "samplepath.plots.convergence.ProcessTimeConvergencePanel.plot",
            return_value="process.png",
        ),
        patch(
            "samplepath.plots.convergence.SojournTimeScatterPanel.plot",
            return_value="scatter.png",
        ),
        patch(
            "samplepath.plots.convergence.plot_arrival_departure_equilibrium_stack",
            return_value="arrival_stack.png",
        ),
        patch(
            "samplepath.plots.convergence.plot_process_time_convergence_stack",
            return_value="process_stack.png",
        ),
        patch(
            "samplepath.plots.convergence.SamplePathConvergencePanel.plot",
            return_value="sample.png",
        ),
    ):
        written = plot_convergence_charts(
            metrics, empirical_metrics, filter_result, chart_config, "/tmp/out"
        )
    return written, mock_rate, mock_lambda


def test_plot_convergence_charts_calls_rate_panel():
    _, mock_rate, _ = _call_plot_convergence_charts_with_rate_panels()
    assert mock_rate.call_count == 1


def test_plot_convergence_charts_includes_lambda_panel():
    written, _, _ = _call_plot_convergence_charts_with_rate_panels()
    assert "lambda.png" in written


def _call_plot_arrival_departure_equilibrium_stack_with_mocks():
    fig = MagicMock()
    axes = np.array([object() for _ in range(2)], dtype=object)
    metrics = SimpleNamespace(
        times=[_t("2024-01-01")],
        Arrivals=np.array([1.0]),
        Departures=np.array([0.0]),
        Lambda=np.array([0.5]),
        arrival_times=[_t("2024-01-01")],
        departure_times=[_t("2024-01-02")],
        freq="D",
    )
    filter_result = SimpleNamespace(display="Filters: test", label="test")
    chart_config = ChartConfig(with_event_marks=True, show_derivations=True)

    @contextmanager
    def fake_context(*args, **kwargs):
        yield fig, axes, "out.png"

    with (
        patch("samplepath.plots.convergence.layout_context", side_effect=fake_context),
        patch("samplepath.plots.convergence.CFDPanel.render") as mock_cfd,
        patch(
            "samplepath.plots.convergence.ArrivalDepartureRateConvergencePanel.render"
        ) as mock_rate,
    ):
        written = plot_arrival_departure_equilibrium_stack(
            filter_result, metrics, chart_config, "/tmp/out"
        )
    return written, mock_cfd, mock_rate


def test_plot_arrival_departure_equilibrium_stack_returns_paths():
    written, _, _ = _call_plot_arrival_departure_equilibrium_stack_with_mocks()
    assert written == "out.png"


def test_plot_arrival_departure_equilibrium_stack_calls_cfd_panel():
    _, mock_cfd, _ = _call_plot_arrival_departure_equilibrium_stack_with_mocks()
    assert mock_cfd.call_count == 1


def test_plot_arrival_departure_equilibrium_stack_calls_rate_panel():
    _, _, mock_rate = _call_plot_arrival_departure_equilibrium_stack_with_mocks()
    assert mock_rate.call_count == 1


def test_sample_path_convergence_panel_render_scores_points():
    ax = MagicMock()
    times = [_t("2024-01-01"), _t("2024-01-02")]
    L_vals = np.array([1.0, 2.0])
    lam_star = np.array([1.0, 1.0])
    W_star = np.array([1.0, 1.0])
    result = SamplePathConvergencePanel().render(
        ax,
        L_vals,
        lam_star,
        W_star,
        times,
        epsilon=0.5,
        horizon_seconds=0.0,
    )
    assert result == (1.0, 2, 2)


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


def test_process_time_convergence_panel_scales_values():
    ax = MagicMock()
    times = [_t("2024-01-01")]
    w_vals = np.array([120.0])
    w_prime_vals = np.array([60.0])
    w_star_vals = np.array([180.0])
    with patch("samplepath.plots.convergence.render_line_chart") as mock_render:
        ProcessTimeConvergencePanel().render(
            ax,
            times,
            w_vals,
            w_prime_vals,
            w_star_vals,
            scale=MINUTES,
        )
    first_call = mock_render.call_args_list[0]
    assert first_call.kwargs["label"] == "w(T) [min]"
    assert np.allclose(first_call.args[2], w_vals / 60.0)


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
        times=metrics.times,
        W_star=np.array([3.0]),
        lam_star=np.array([0.5]),
        sojourn_vals=np.array([2.0]),
        residence_time_vals=np.array([1.0]),
        residence_completed=np.array([True]),
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


def test_sojourn_time_scatter_panel_overlays_drop_lines():
    ax = MagicMock()
    times = [_t("2024-01-01")]
    w_vals = np.array([1.0])
    w_prime_vals = np.array([2.0])
    sojourn_vals = np.array([3.0])
    departures = [times[0]]
    with (
        patch(
            "samplepath.plots.convergence.build_event_overlays",
            return_value=["overlay"],
        ) as mock_overlays,
        patch("samplepath.plots.convergence.render_line_chart"),
        patch("samplepath.plots.convergence.render_scatter_chart"),
    ):
        SojournTimeScatterPanel(with_event_marks=True).render(
            ax,
            times,
            w_vals,
            w_prime_vals,
            departures,
            sojourn_vals,
        )
    assert mock_overlays.call_count == 2
    first_args, first_kwargs = (
        mock_overlays.call_args_list[0].args,
        mock_overlays.call_args_list[0].kwargs,
    )
    assert first_args[2] == []
    assert first_args[3] == departures
    assert first_kwargs["drop_lines_for_arrivals"] is False
    assert first_kwargs["drop_lines_for_departures"] is True
    second_args, second_kwargs = (
        mock_overlays.call_args_list[1].args,
        mock_overlays.call_args_list[1].kwargs,
    )
    assert second_args[2] == []
    assert second_args[3] == departures
    assert second_kwargs["drop_lines_for_arrivals"] is False
    assert second_kwargs["drop_lines_for_departures"] is True


def test_arrival_departure_rate_panel_scales_rates():
    ax = MagicMock()
    times = [_t("2024-01-01"), _t("2024-01-02")]
    departures = np.array([0.0, 1.0])
    lambda_rate = np.array([0.5, 0.75])
    with (
        patch("samplepath.plots.convergence.render_line_chart") as mock_render,
        patch("samplepath.plots.convergence._clip_axis_to_percentile"),
    ):
        ArrivalDepartureRateConvergencePanel().render(
            ax,
            times,
            departures,
            lambda_rate,
            scale=MINUTES,
        )
    first_call = mock_render.call_args_list[0]
    assert "1/min" in first_call.kwargs["label"]
    assert np.allclose(first_call.args[2], lambda_rate * 60.0)


def test_sojourn_time_scatter_panel_plot_calls_renderer():
    fig = MagicMock()
    ax = MagicMock()
    metrics = SimpleNamespace(
        times=[_t("2024-01-01")],
        w=np.array([1.0]),
        w_prime=np.array([2.0]),
        departure_times=[_t("2024-01-02")],
        freq="D",
    )
    filter_result = SimpleNamespace(display="Filters: test", label="test")
    chart_config = ChartConfig()
    empirical_metrics = ElementWiseEmpiricalMetrics(
        times=metrics.times,
        W_star=np.array([3.0]),
        lam_star=np.array([0.5]),
        sojourn_vals=np.array([4.0]),
        residence_time_vals=np.array([1.0]),
        residence_completed=np.array([True]),
    )

    @contextmanager
    def fake_context(out_path=None, **kwargs):
        assert out_path is None
        assert kwargs["out_dir"] == "/tmp/out"
        assert kwargs["subdir"] == "convergence/panels"
        assert kwargs["base_name"] == "residence_time_sojourn_time_scatter_plot"
        yield fig, ax, "out.png"

    with (
        patch("samplepath.plots.convergence.figure_context", side_effect=fake_context),
        patch(
            "samplepath.plots.convergence.SojournTimeScatterPanel.render"
        ) as mock_render,
    ):
        written = SojournTimeScatterPanel().plot(
            metrics, empirical_metrics, filter_result, chart_config, "/tmp/out"
        )
    assert written == "out.png"
    mock_render.assert_called_once()


def test_process_time_convergence_stack_calls_panel_renderers():
    fig = MagicMock()
    axes = np.array([object() for _ in range(2)], dtype=object)
    metrics = SimpleNamespace(
        times=[_t("2024-01-01")],
        w=np.array([1.0]),
        w_prime=np.array([2.0]),
        arrival_times=[_t("2024-01-01")],
        departure_times=[_t("2024-01-02")],
        freq="D",
    )
    empirical_metrics = ElementWiseEmpiricalMetrics(
        times=metrics.times,
        W_star=np.array([3.0]),
        lam_star=np.array([0.5]),
        sojourn_vals=np.array([4.0]),
        residence_time_vals=np.array([1.0]),
        residence_completed=np.array([True]),
    )
    filter_result = SimpleNamespace(display="Filters: test", label="test")
    chart_config = ChartConfig(with_event_marks=True)

    @contextmanager
    def fake_context(*args, **kwargs):
        yield fig, axes, "out.png"

    with (
        patch("samplepath.plots.convergence.layout_context", side_effect=fake_context),
        patch(
            "samplepath.plots.convergence.ProcessTimeConvergencePanel.render"
        ) as mock_top,
        patch(
            "samplepath.plots.convergence.SojournTimeScatterPanel.render"
        ) as mock_bottom,
    ):
        written = plot_process_time_convergence_stack(
            filter_result, metrics, empirical_metrics, chart_config, "/tmp/out"
        )
    assert written == "out.png"
    mock_top.assert_called_once()
    mock_bottom.assert_called_once()


def _call_plot_convergence_charts_with_process_panel():
    chart_config = ChartConfig()
    metrics = SimpleNamespace(
        times=[_t("2024-01-01")],
        w=np.array([1.0]),
        w_prime=np.array([2.0]),
        arrival_times=[],
        departure_times=[],
        freq=None,
    )
    empirical_metrics = ElementWiseEmpiricalMetrics(
        times=metrics.times,
        W_star=np.array([3.0]),
        lam_star=np.array([0.5]),
        sojourn_vals=np.array([2.0]),
        residence_time_vals=np.array([1.0]),
        residence_completed=np.array([True]),
    )
    filter_result = SimpleNamespace(display="Filters: test", label="test")
    with (
        patch(
            "samplepath.plots.convergence.CumulativeArrivalRateConvergencePanel.plot",
            return_value="lambda.png",
        ),
        patch(
            "samplepath.plots.convergence.ArrivalDepartureRateConvergencePanel.plot",
            return_value="rate.png",
        ),
        patch(
            "samplepath.plots.convergence.plot_arrival_departure_equilibrium_stack",
            return_value="arrival_stack.png",
        ),
        patch(
            "samplepath.plots.convergence.SojournTimeScatterPanel.plot",
            return_value="scatter.png",
        ),
        patch(
            "samplepath.plots.convergence.plot_process_time_convergence_stack",
            return_value="process_stack.png",
        ),
        patch(
            "samplepath.plots.convergence.SamplePathConvergencePanel.plot",
            return_value="sample.png",
        ),
        patch(
            "samplepath.plots.convergence.ProcessTimeConvergencePanel.plot",
            return_value="process.png",
        ) as mock_plot,
    ):
        written = plot_convergence_charts(
            metrics, empirical_metrics, filter_result, chart_config, "/tmp/out"
        )
    return written, mock_plot


def test_plot_convergence_charts_includes_process_time_panel():
    written, _ = _call_plot_convergence_charts_with_process_panel()
    assert "process.png" in written


def test_plot_convergence_charts_passes_chart_config_to_process_panel():
    _, mock_plot = _call_plot_convergence_charts_with_process_panel()
    assert isinstance(mock_plot.call_args.args[3], ChartConfig)
