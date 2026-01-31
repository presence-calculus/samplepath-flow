# -*- coding: utf-8 -*-
# Copyright (c) 2025 Krishna Kumar
# SPDX-License-Identifier: MIT
from contextlib import contextmanager
import os
from types import SimpleNamespace
from typing import Optional
from unittest.mock import MagicMock, patch

from matplotlib import colors as mcolors
import numpy as np
import pandas as pd

from samplepath.metrics import ElementWiseEmpiricalMetrics
from samplepath.plots import core
from samplepath.plots.chart_config import ChartConfig
from samplepath.plots.core import ClipOptions
from samplepath.plots.figure_context import resolve_chart_path
from samplepath.utils.duration_scale import MINUTES


def _t(s: str) -> pd.Timestamp:
    return pd.Timestamp(s)


def test_render_N_passes_overlays_with_default_color():
    ax = MagicMock()
    times = [_t("2024-01-01")]
    values = np.array([1.0])
    overlays = ["overlay"]
    with (
        patch("samplepath.plots.core.build_event_overlays", return_value=overlays),
        patch("samplepath.plots.core.render_step_chart") as mock_render,
    ):
        core.NPanel(with_event_marks=True).render(
            ax,
            times,
            values,
            arrival_times=[times[0]],
            departure_times=[times[0]],
        )
    mock_render.assert_called_once()
    _, kwargs = mock_render.call_args
    assert kwargs["fill"] is True
    assert kwargs["overlays"] == overlays
    ax.set_title.assert_called_once()
    ax.set_ylabel.assert_called_once_with("N(t)")


def test_render_N_no_title_when_suppressed():
    ax = MagicMock()
    times = [_t("2024-01-01")]
    values = np.array([1.0])
    with (
        patch("samplepath.plots.core.build_event_overlays", return_value=None),
        patch("samplepath.plots.core.render_step_chart"),
    ):
        core.NPanel(show_title=False).render(ax, times, values)
    ax.set_title.assert_not_called()


def test_render_N_title_appends_derivation_when_enabled():
    ax = MagicMock()
    times = [_t("2024-01-01")]
    values = np.array([1.0])
    with (
        patch("samplepath.plots.core.render_step_chart"),
        patch("samplepath.plots.core.MetricDerivations.get", return_value="DERIVATION"),
    ):
        core.NPanel(title="Base Title", show_derivations=True).render(ax, times, values)
    assert ax.set_title.call_args[0][0] == "Base Title: DERIVATION"


def test_render_N_title_uses_metric_derivations_lookup():
    ax = MagicMock()
    times = [_t("2024-01-01")]
    values = np.array([1.0])
    with (
        patch("samplepath.plots.core.render_step_chart"),
        patch("samplepath.plots.core.MetricDerivations.get") as mock_get,
    ):
        core.NPanel(title="Base Title", show_derivations=True).render(ax, times, values)
    assert mock_get.call_args[0][0] == "N"


def test_n_panel_render_uses_panel_state_for_title():
    ax = MagicMock()
    times = [_t("2024-01-01")]
    values = np.array([1.0])
    panel = core.NPanel(title="Base Title", show_derivations=True)
    with (
        patch("samplepath.plots.core.render_step_chart"),
        patch("samplepath.plots.core.MetricDerivations.get", return_value="DERIVATION"),
    ):
        panel.render(ax, times, values)
    assert ax.set_title.call_args[0][0] == "Base Title: DERIVATION"


def test_construct_title_returns_base_without_derivation():
    assert core.construct_title("Base", False, derivation_key="N") == "Base"


def test_construct_title_appends_derivation_when_available():
    with patch(
        "samplepath.plots.core.MetricDerivations.get", return_value="DERIVATION"
    ):
        title = core.construct_title("Base", True, derivation_key="N")
    assert title == "Base: DERIVATION"


def test_render_L_title_unchanged_when_derivations_disabled():
    ax = MagicMock()
    times = [_t("2024-01-01")]
    values = np.array([2.0])
    with patch("samplepath.plots.core.render_line_chart"):
        core.LPanel(title="Base Title", show_derivations=False).render(
            ax, times, values
        )
    assert ax.set_title.call_args[0][0] == "Base Title"


def test_render_Lambda_title_appends_derivation_when_enabled():
    ax = MagicMock()
    times = [_t("2024-01-01")]
    values = np.array([1.0])
    with (
        patch("samplepath.plots.core.render_line_chart"),
        patch("samplepath.plots.core.MetricDerivations.get", return_value="DERIVATION"),
    ):
        core.LambdaPanel(title="Base Title", show_derivations=True).render(
            ax, times, values
        )
    assert ax.set_title.call_args[0][0] == "Base Title: DERIVATION"


def test_render_w_title_appends_derivation_when_enabled():
    ax = MagicMock()
    times = [_t("2024-01-01")]
    values = np.array([1.0])
    with (
        patch("samplepath.plots.core.render_line_chart"),
        patch("samplepath.plots.core.MetricDerivations.get", return_value="DERIVATION"),
    ):
        core.WPanel(title="Base Title", show_derivations=True).render(ax, times, values)
    assert ax.set_title.call_args[0][0] == "Base Title: DERIVATION"


def test_render_H_title_appends_derivation_when_enabled():
    ax = MagicMock()
    times = [_t("2024-01-01")]
    values = np.array([1.0])
    with (
        patch("samplepath.plots.core.render_line_chart"),
        patch("samplepath.plots.core.MetricDerivations.get", return_value="DERIVATION"),
    ):
        core.HPanel(title="Base Title", show_derivations=True).render(ax, times, values)
    assert ax.set_title.call_args[0][0] == "Base Title: DERIVATION"


def test_render_H_passes_overlays_with_default_color():
    ax = MagicMock()
    times = [_t("2024-01-01")]
    values = np.array([2.0])
    overlays = ["overlay"]
    with (
        patch("samplepath.plots.core.build_event_overlays", return_value=overlays),
        patch("samplepath.plots.core.render_line_chart") as mock_render,
    ):
        core.HPanel(with_event_marks=True).render(
            ax,
            times,
            values,
            arrival_times=[times[0]],
            departure_times=[times[0]],
        )
    _, kwargs = mock_render.call_args
    assert kwargs["overlays"] == overlays
    assert "color" not in kwargs or kwargs["color"] == "tab:blue"


def test_render_H_passes_overlays():
    ax = MagicMock()
    times = [_t("2024-01-01")]
    values = np.array([2.0])
    overlays = ["overlay"]
    with (
        patch("samplepath.plots.core.build_event_overlays", return_value=overlays),
        patch("samplepath.plots.core.render_line_chart") as mock_render,
    ):
        core.HPanel(with_event_marks=True).render(
            ax,
            times,
            values,
            arrival_times=[times[0]],
            departure_times=[times[0]],
        )
    _, kwargs = mock_render.call_args
    assert kwargs["overlays"] == overlays


def test_render_Theta_label():
    ax = MagicMock()
    times = [_t("2024-01-01")]
    values = np.array([1.0])
    with patch("samplepath.plots.core.render_line_chart") as mock_render:
        core.ThetaPanel().render(ax, times, values)
    assert mock_render.call_args.kwargs["label"] == "Θ(T) [1/hr]"


def test_render_Theta_sets_ylabel():
    ax = MagicMock()
    times = [_t("2024-01-01")]
    values = np.array([1.0])
    with patch("samplepath.plots.core.render_line_chart"):
        core.ThetaPanel().render(ax, times, values)
    assert ax.set_ylabel.call_args[0][0] == "Θ(T) [1/hr]"


def test_render_Theta_passes_overlays():
    ax = MagicMock()
    times = [_t("2024-01-01")]
    values = np.array([1.0])
    overlays = ["overlay"]
    with (
        patch("samplepath.plots.core.build_event_overlays", return_value=overlays),
        patch("samplepath.plots.core.render_line_chart") as mock_render,
    ):
        core.ThetaPanel(with_event_marks=True).render(
            ax,
            times,
            values,
            departure_times=[times[0]],
        )
    assert mock_render.call_args.kwargs["overlays"] == overlays


def test_render_w_prime_label():
    ax = MagicMock()
    times = [_t("2024-01-01")]
    values = np.array([1.0])
    with patch("samplepath.plots.core.render_line_chart") as mock_render:
        core.WPrimePanel().render(ax, times, values)
    assert mock_render.call_args.kwargs["label"] == "w'(T) [hrs]"


def test_render_w_prime_sets_ylabel():
    ax = MagicMock()
    times = [_t("2024-01-01")]
    values = np.array([1.0])
    with patch("samplepath.plots.core.render_line_chart"):
        core.WPrimePanel().render(ax, times, values)
    assert ax.set_ylabel.call_args[0][0] == "w'(T) [hrs]"


def test_render_w_prime_passes_overlays():
    ax = MagicMock()
    times = [_t("2024-01-01")]
    values = np.array([1.0])
    overlays = ["overlay"]
    with (
        patch(
            "samplepath.plots.core.build_event_overlays", return_value=overlays
        ) as mock_overlays,
        patch("samplepath.plots.core.render_line_chart") as mock_render,
    ):
        core.WPrimePanel(with_event_marks=True).render(
            ax,
            times,
            values,
            arrival_times=[times[0]],
            departure_times=[times[0]],
        )
    assert mock_render.call_args.kwargs["overlays"] == overlays
    assert mock_overlays.call_args.kwargs["drop_lines_for_arrivals"] is False
    assert mock_overlays.call_args.kwargs["drop_lines_for_departures"] is True


def test_render_indicator_sets_ylabel():
    ax = MagicMock()
    with patch("samplepath.plots.core.render_scatter_chart"):
        core.EventIndicatorPanel().render(ax, [])
    assert ax.set_ylabel.call_args[0][0] == "Indicator"


def test_render_indicator_hides_y_ticks():
    ax = MagicMock()
    with patch("samplepath.plots.core.render_scatter_chart"):
        core.EventIndicatorPanel().render(ax, [])
    ax.set_yticks.assert_called_once_with([])


def test_render_indicator_passes_overlays():
    ax = MagicMock()
    arrival = _t("2024-01-01")
    departure = _t("2024-01-02")
    with patch("samplepath.plots.core.render_scatter_chart") as mock_render:
        core.EventIndicatorPanel(with_event_marks=True).render(
            ax,
            [arrival, departure],
            arrival_times=[arrival],
            departure_times=[departure],
        )
    overlays = mock_render.call_args.kwargs["overlays"]
    assert overlays[0].color == "purple"


def test_plot_single_panel_indicator_calls_renderer():
    fig = MagicMock()
    ax = MagicMock()
    metrics = _metrics_fixture()
    chart_config = ChartConfig()
    filter_result = SimpleNamespace(display="Filters: test", label="test")

    @contextmanager
    def fake_context(*args, **kwargs):
        yield fig, ax, "out.png"

    with (
        patch("samplepath.plots.core.figure_context", side_effect=fake_context),
        patch("samplepath.plots.core.EventIndicatorPanel.render") as mock_render,
    ):
        core.EventIndicatorPanel().plot(
            metrics, filter_result, chart_config, "/tmp/out"
        )
    mock_render.assert_called_once()


def test_render_arrivals_sets_ylabel():
    ax = MagicMock()
    with patch("samplepath.plots.core.render_step_chart"):
        core.ArrivalsPanel().render(ax, [], [])
    assert ax.set_ylabel.call_args[0][0] == "count"


def test_render_arrivals_passes_overlays():
    ax = MagicMock()
    times = [_t("2024-01-01")]
    values = np.array([1.0])
    overlays = ["overlay"]
    with (
        patch("samplepath.plots.core.build_event_overlays", return_value=overlays),
        patch("samplepath.plots.core.render_step_chart") as mock_render,
    ):
        core.ArrivalsPanel(with_event_marks=True).render(
            ax,
            times,
            values,
            arrival_times=times,
        )
    assert mock_render.call_args.kwargs["overlays"] == overlays


def test_plot_single_panel_arrivals_calls_renderer():
    fig = MagicMock()
    ax = MagicMock()
    metrics = _metrics_fixture()
    chart_config = ChartConfig()
    filter_result = SimpleNamespace(display="Filters: test", label="test")

    @contextmanager
    def fake_context(*args, **kwargs):
        yield fig, ax, "out.png"

    with (
        patch("samplepath.plots.core.figure_context", side_effect=fake_context),
        patch("samplepath.plots.core.ArrivalsPanel.render") as mock_render,
    ):
        core.ArrivalsPanel().plot(metrics, filter_result, chart_config, "/tmp/out")
    mock_render.assert_called_once()


def test_render_departures_sets_ylabel():
    ax = MagicMock()
    with patch("samplepath.plots.core.render_step_chart"):
        core.DeparturesPanel().render(ax, [], [])
    assert ax.set_ylabel.call_args[0][0] == "count"


def test_render_departures_passes_overlays():
    ax = MagicMock()
    times = [_t("2024-01-01")]
    values = np.array([1.0])
    overlays = ["overlay"]
    with (
        patch("samplepath.plots.core.build_event_overlays", return_value=overlays),
        patch("samplepath.plots.core.render_step_chart") as mock_render,
    ):
        core.DeparturesPanel(with_event_marks=True).render(
            ax,
            times,
            values,
            departure_times=times,
        )
    assert mock_render.call_args.kwargs["overlays"] == overlays


def test_plot_single_panel_departures_calls_renderer():
    fig = MagicMock()
    ax = MagicMock()
    metrics = _metrics_fixture()
    chart_config = ChartConfig()
    filter_result = SimpleNamespace(display="Filters: test", label="test")

    @contextmanager
    def fake_context(*args, **kwargs):
        yield fig, ax, "out.png"

    with (
        patch("samplepath.plots.core.figure_context", side_effect=fake_context),
        patch("samplepath.plots.core.DeparturesPanel.render") as mock_render,
    ):
        core.DeparturesPanel().plot(metrics, filter_result, chart_config, "/tmp/out")
    mock_render.assert_called_once()


def test_render_L_passes_overlays_with_default_color():
    ax = MagicMock()
    times = [_t("2024-01-01")]
    values = np.array([2.0])
    overlays = ["overlay"]
    with (
        patch("samplepath.plots.core.build_event_overlays", return_value=overlays),
        patch("samplepath.plots.core.render_line_chart") as mock_render,
    ):
        core.LPanel(with_event_marks=True).render(
            ax,
            times,
            values,
            arrival_times=[times[0]],
            departure_times=[times[0]],
        )
    _, kwargs = mock_render.call_args
    assert kwargs["overlays"] == overlays
    assert "color" not in kwargs or kwargs["color"] == "tab:blue"
    ax.set_title.assert_called_once()
    ax.set_ylabel.assert_called_once_with("L(T)")


def test_render_Lambda_passes_clip_options():
    ax = MagicMock()
    times = [_t("2024-01-01"), _t("2024-01-02")]
    values = np.array([1.0, 2.0])
    clip = ClipOptions(pctl_upper=99.0, pctl_lower=1.0, warmup_seconds=1800.0)
    with (
        patch("samplepath.plots.core.render_line_chart") as mock_line,
        patch("samplepath.plots.core._clip_axis_to_percentile") as mock_clip,
    ):
        core.LambdaPanel(clip_opts=clip).render(ax, times, values)
    mock_line.assert_called_once()
    args = mock_clip.call_args.args
    assert args[0] == ax
    assert args[1] == list(times)
    assert np.allclose(args[2], values * 3600.0)
    assert args[3:] == (99.0, 1.0, 1800.0)
    ax.set_ylabel.assert_called_once_with("Λ(T) [1/hr]")


def test_render_w_sets_defaults():
    ax = MagicMock()
    times = [_t("2024-01-01")]
    values = np.array([0.5])
    with patch("samplepath.plots.core.render_line_chart") as mock_line:
        core.WPanel().render(ax, times, values)
    mock_line.assert_called_once()
    ax.set_ylabel.assert_called_once_with("w(T) [hrs]")


def test_render_Lambda_scales_rate_values():
    ax = MagicMock()
    times = [_t("2024-01-01")]
    values = np.array([2.0])
    with patch("samplepath.plots.core.render_line_chart") as mock_line:
        core.LambdaPanel().render(ax, times, values, scale=MINUTES)
    assert mock_line.call_args.kwargs["label"] == "Λ(T) [1/min]"
    assert np.allclose(mock_line.call_args.args[2], values * 60.0)


def test_render_w_scales_duration_values():
    ax = MagicMock()
    times = [_t("2024-01-01")]
    values = np.array([120.0])
    with patch("samplepath.plots.core.render_line_chart") as mock_line:
        core.WPanel().render(ax, times, values, scale=MINUTES)
    assert mock_line.call_args.kwargs["label"] == "w(T) [min]"
    assert np.allclose(mock_line.call_args.args[2], values / 60.0)


def test_render_sojourn_sets_defaults():
    ax = MagicMock()
    times = [_t("2024-01-01")]
    values = np.array([0.5])
    with patch("samplepath.plots.core.render_line_chart") as mock_line:
        core.SojournTimePanel().render(ax, times, values)
    mock_line.assert_called_once()
    ax.set_ylabel.assert_called_once_with("W*(T) [hrs]")


def test_render_sojourn_scatter_uses_departure_times():
    ax = MagicMock()
    departures = [_t("2024-01-01"), _t("2024-01-02")]
    sojourn_vals = np.array([1.0, 2.0])
    with patch("samplepath.plots.core.render_scatter_chart") as mock_scatter:
        core.SojournTimeScatterPanel(with_event_marks=True).render(
            ax, departures, sojourn_vals
        )
    assert mock_scatter.call_args.args[1] == departures


def test_render_sojourn_scatter_uses_departure_color():
    ax = MagicMock()
    departures = [_t("2024-01-01")]
    sojourn_vals = np.array([1.0])
    with patch("samplepath.plots.core.render_scatter_chart") as mock_scatter:
        core.SojournTimeScatterPanel(with_event_marks=True).render(
            ax, departures, sojourn_vals
        )
    assert mock_scatter.call_args.kwargs["color"] == "green"


def test_render_sojourn_scatter_drop_lines_when_event_marks():
    ax = MagicMock()
    departures = [_t("2024-01-01")]
    sojourn_vals = np.array([1.0])
    with patch("samplepath.plots.core.render_scatter_chart") as mock_scatter:
        core.SojournTimeScatterPanel(with_event_marks=True).render(
            ax, departures, sojourn_vals
        )
    assert mock_scatter.call_args.kwargs["drop_lines"] == "vertical"


def test_render_residence_scatter_open_uses_arrival_color():
    ax = MagicMock()
    arrivals = [_t("2024-01-01")]
    vals = np.array([1.0])
    completed = np.array([False])
    with patch("samplepath.plots.core.render_scatter_chart") as mock_scatter:
        core.ResidenceTimeScatterPanel(with_event_marks=True).render(
            ax, arrivals, vals, completed
        )
    assert mock_scatter.call_args.kwargs["color"] == "purple"


def test_render_residence_scatter_completed_uses_departure_color():
    ax = MagicMock()
    arrivals = [_t("2024-01-01")]
    vals = np.array([1.0])
    completed = np.array([True])
    with patch("samplepath.plots.core.render_scatter_chart") as mock_scatter:
        core.ResidenceTimeScatterPanel(with_event_marks=True).render(
            ax, arrivals, vals, completed
        )
    assert mock_scatter.call_args.kwargs["color"] == "green"


def test_render_residence_scatter_drop_line_color_arrival():
    ax = MagicMock()
    arrivals = [_t("2024-01-01")]
    vals = np.array([1.0])
    completed = np.array([True])
    with patch("samplepath.plots.core.render_scatter_chart") as mock_scatter:
        core.ResidenceTimeScatterPanel(with_event_marks=True).render(
            ax, arrivals, vals, completed
        )
    assert mock_scatter.call_args.kwargs["drop_line_color"] == "purple"


def test_render_Lambda_arrival_overlays_when_enabled():
    ax = MagicMock()
    times = [_t("2024-01-01")]
    values = np.array([1.0])
    arrivals = [times[0]]
    with patch("samplepath.plots.core.render_line_chart") as mock_line:
        core.LambdaPanel(with_event_marks=True).render(
            ax,
            times,
            values,
            arrival_times=arrivals,
        )
    overlays = mock_line.call_args.kwargs["overlays"]
    assert overlays[0].color == "purple"


def test_render_w_overlays_when_enabled():
    ax = MagicMock()
    times = [_t("2024-01-01")]
    values = np.array([1.0])
    arrivals = [times[0]]
    departures = [times[0]]
    with patch("samplepath.plots.core.render_line_chart") as mock_line:
        core.WPanel(with_event_marks=True).render(
            ax,
            times,
            values,
            arrival_times=arrivals,
            departure_times=departures,
        )
    overlays = mock_line.call_args.kwargs["overlays"]
    assert overlays[0].color == "purple"


def test_render_w_overlays_include_departures():
    ax = MagicMock()
    times = [_t("2024-01-01")]
    values = np.array([1.0])
    arrivals = [times[0]]
    departures = [times[0]]
    with patch("samplepath.plots.core.render_line_chart") as mock_line:
        core.WPanel(with_event_marks=True).render(
            ax,
            times,
            values,
            arrival_times=arrivals,
            departure_times=departures,
        )
    overlays = mock_line.call_args.kwargs["overlays"]
    assert overlays[1].color == "green"


def test_render_w_overlays_departures_no_drop_lines():
    ax = MagicMock()
    times = [_t("2024-01-01")]
    values = np.array([1.0])
    arrivals = [times[0]]
    departures = [times[0]]
    with patch("samplepath.plots.core.render_line_chart") as mock_line:
        core.WPanel(with_event_marks=True).render(
            ax,
            times,
            values,
            arrival_times=arrivals,
            departure_times=departures,
        )
    overlays = mock_line.call_args.kwargs["overlays"]
    assert overlays[1].drop_lines is False


def test_render_H_sets_defaults():
    ax = MagicMock()
    times = [_t("2024-01-01")]
    values = np.array([1.0])
    with patch("samplepath.plots.core.render_line_chart") as mock_line:
        core.HPanel().render(ax, times, values)
    mock_line.assert_called_once()
    ax.set_ylabel.assert_called_once_with("H(T) [hrs·items]")


def _render_cfd_with_mocks(with_event_marks: bool = False):
    ax = MagicMock()
    times = [_t("2024-01-01")]
    arrivals = np.array([1.0])
    departures = np.array([0.0])
    with patch("samplepath.plots.core.render_step_chart") as mock_step:
        core.CFDPanel(with_event_marks=with_event_marks).render(
            ax,
            times,
            arrivals,
            departures,
            arrival_times=times,
            departure_times=times,
        )
    return ax, mock_step


def test_render_CFD_calls_step_twice():
    _, mock_step = _render_cfd_with_mocks()
    assert mock_step.call_count == 2


def test_render_CFD_arrivals_label():
    _, mock_step = _render_cfd_with_mocks()
    assert mock_step.call_args_list[0].kwargs["label"] == "A(T) - Cumulative arrivals"


def test_render_CFD_arrivals_color():
    _, mock_step = _render_cfd_with_mocks()
    assert mock_step.call_args_list[0].kwargs["color"] == "purple"


def test_render_CFD_arrivals_fill_false():
    _, mock_step = _render_cfd_with_mocks()
    assert mock_step.call_args_list[0].kwargs["fill"] is False


def test_render_CFD_departures_label():
    _, mock_step = _render_cfd_with_mocks()
    assert mock_step.call_args_list[1].kwargs["label"] == "D(T) - Cumulative departures"


def test_render_CFD_appends_derivations_to_labels_when_enabled():
    ax = MagicMock()
    times = [_t("2024-01-01")]
    arrivals = np.array([1.0])
    departures = np.array([0.0])

    def _derivation_for_key(key: str):
        return {
            "A": "A(T) = ∑ arrivals in [0, T]",
            "D": "D(T) = ∑ departures in [0, T]",
        }[key]

    with (
        patch("samplepath.plots.core.render_step_chart") as mock_step,
        patch(
            "samplepath.plots.core.MetricDerivations.get",
            side_effect=_derivation_for_key,
        ),
    ):
        core.CFDPanel(show_derivations=True).render(ax, times, arrivals, departures)
    assert (
        mock_step.call_args_list[0].kwargs["label"]
        == "A(T) - Cumulative arrivals — A(T) = ∑ arrivals in [0, T]"
    )
    assert (
        mock_step.call_args_list[1].kwargs["label"]
        == "D(T) - Cumulative departures — D(T) = ∑ departures in [0, T]"
    )


def test_render_CFD_departures_color():
    _, mock_step = _render_cfd_with_mocks()
    assert mock_step.call_args_list[1].kwargs["color"] == "green"


def test_render_CFD_departures_fill_false():
    _, mock_step = _render_cfd_with_mocks()
    assert mock_step.call_args_list[1].kwargs["fill"] is False


def test_render_CFD_arrivals_overlay_none_by_default():
    _, mock_step = _render_cfd_with_mocks()
    assert mock_step.call_args_list[0].kwargs["overlays"] is None


def test_render_CFD_departures_overlay_none_by_default():
    _, mock_step = _render_cfd_with_mocks()
    assert mock_step.call_args_list[1].kwargs["overlays"] is None


def test_render_CFD_arrivals_overlay_drop_lines_true():
    _, mock_step = _render_cfd_with_mocks(with_event_marks=True)
    overlay = mock_step.call_args_list[0].kwargs["overlays"][0]
    assert overlay.drop_lines is True


def test_render_CFD_departures_overlay_drop_lines_true():
    _, mock_step = _render_cfd_with_mocks(with_event_marks=True)
    overlay = mock_step.call_args_list[1].kwargs["overlays"][0]
    assert overlay.drop_lines is True


def test_render_CFD_arrivals_overlay_uses_arrival_times():
    ax = MagicMock()
    times = [_t("2024-01-01"), _t("2024-01-02")]
    arrivals = np.array([1.0, 2.0])
    departures = np.array([0.0, 1.0])
    arrival_times = [times[0]]
    departure_times = [times[1]]
    with patch("samplepath.plots.core.render_step_chart") as mock_step:
        core.CFDPanel(with_event_marks=True).render(
            ax,
            times,
            arrivals,
            departures,
            arrival_times=arrival_times,
            departure_times=departure_times,
        )
    overlay = mock_step.call_args_list[0].kwargs["overlays"][0]
    assert overlay.x == arrival_times


def test_render_CFD_departures_overlay_uses_departure_times():
    ax = MagicMock()
    times = [_t("2024-01-01"), _t("2024-01-02")]
    arrivals = np.array([1.0, 2.0])
    departures = np.array([0.0, 1.0])
    arrival_times = [times[0]]
    departure_times = [times[1]]
    with patch("samplepath.plots.core.render_step_chart") as mock_step:
        core.CFDPanel(with_event_marks=True).render(
            ax,
            times,
            arrivals,
            departures,
            arrival_times=arrival_times,
            departure_times=departure_times,
        )
    overlay = mock_step.call_args_list[1].kwargs["overlays"][0]
    assert overlay.x == departure_times


def test_render_CFD_sets_title():
    ax, _ = _render_cfd_with_mocks()
    ax.set_title.assert_called_once_with("Cumulative Flow Diagram")


def test_render_CFD_sets_ylabel():
    ax, _ = _render_cfd_with_mocks()
    ax.set_ylabel.assert_called_once_with("count")


def test_render_CFD_sets_legend():
    ax, _ = _render_cfd_with_mocks()
    ax.legend.assert_called_once()


def test_render_CFD_fills_between_when_arrivals_above_departures():
    ax = MagicMock()
    times = [_t("2024-01-01"), _t("2024-01-02")]
    arrivals = np.array([1.0, 2.0])
    departures = np.array([0.0, 3.0])
    with patch("samplepath.plots.core.render_step_chart"):
        core.CFDPanel().render(ax, times, arrivals, departures)
    ax.fill_between.assert_called_once()
    _, kwargs = ax.fill_between.call_args
    assert kwargs["color"] == "grey"
    assert kwargs["alpha"] == 0.3
    assert kwargs["step"] == "post"
    assert kwargs["interpolate"] is True
    assert kwargs["zorder"] == 1
    assert kwargs["where"].tolist() == [True, False]


def test_plot_single_panel_calls_renderer():
    fig = MagicMock()
    ax = MagicMock()

    @contextmanager
    def fake_context(*args, **kwargs):
        yield fig, ax, "out.png"

    with (
        patch("samplepath.plots.core.figure_context", side_effect=fake_context),
        patch("samplepath.plots.core.NPanel.render") as mock_render,
    ):
        core.NPanel(with_event_marks=True).plot(
            _metrics_fixture(),
            SimpleNamespace(display="Filters: test", label="test"),
            ChartConfig(with_event_marks=True),
            "/tmp/out",
        )
    mock_render.assert_called_once()


def test_NPanel_plot_uses_svg_format():
    fig = MagicMock()
    ax = MagicMock()
    metrics = _metrics_fixture()
    chart_config = ChartConfig(chart_format="svg")

    @contextmanager
    def fake_context(out_path=None, **kwargs):
        assert out_path is None
        assert kwargs["out_dir"] == "/tmp/out"
        assert kwargs["subdir"] == "core/panels"
        assert kwargs["base_name"] == "sample_path_N"
        assert kwargs["chart_config"] == chart_config
        yield fig, ax, "out.png"

    with patch("samplepath.plots.core.figure_context", side_effect=fake_context):
        core.NPanel().plot(
            metrics,
            SimpleNamespace(display="Filters: test", label="test"),
            chart_config,
            "/tmp/out",
        )


def test_NPanel_plot_uses_png_dpi():
    fig = MagicMock()
    ax = MagicMock()
    metrics = _metrics_fixture()
    chart_config = ChartConfig(chart_format="png", chart_dpi=200)

    @contextmanager
    def fake_context(out_path=None, **kwargs):
        assert out_path is None
        assert kwargs["out_dir"] == "/tmp/out"
        assert kwargs["subdir"] == "core/panels"
        assert kwargs["base_name"] == "sample_path_N"
        assert kwargs["chart_config"] == chart_config
        yield fig, ax, "out.png"

    with patch("samplepath.plots.core.figure_context", side_effect=fake_context):
        core.NPanel().plot(
            metrics,
            SimpleNamespace(display="Filters: test", label="test"),
            chart_config,
            "/tmp/out",
        )


def test_plot_single_panel_L_calls_renderer():
    fig = MagicMock()
    ax = MagicMock()
    metrics = _metrics_fixture()
    chart_config = ChartConfig()
    filter_result = SimpleNamespace(display="Filters: test", label="test")

    @contextmanager
    def fake_context(*args, **kwargs):
        yield fig, ax, "out.png"

    with (
        patch("samplepath.plots.core.figure_context", side_effect=fake_context),
        patch("samplepath.plots.core.LPanel.render") as mock_render,
    ):
        core.LPanel().plot(metrics, filter_result, chart_config, "/tmp/out")
    mock_render.assert_called_once()


def test_plot_single_panel_Lambda_calls_renderer():
    fig = MagicMock()
    ax = MagicMock()
    metrics = _metrics_fixture()
    chart_config = ChartConfig()
    filter_result = SimpleNamespace(display="Filters: test", label="test")

    @contextmanager
    def fake_context(*args, **kwargs):
        yield fig, ax, "out.png"

    with (
        patch("samplepath.plots.core.figure_context", side_effect=fake_context),
        patch("samplepath.plots.core.LambdaPanel.render") as mock_render,
    ):
        core.LambdaPanel().plot(metrics, filter_result, chart_config, "/tmp/out")
    mock_render.assert_called_once()


def test_plot_single_panel_w_calls_renderer():
    fig = MagicMock()
    ax = MagicMock()
    metrics = _metrics_fixture()
    chart_config = ChartConfig()
    filter_result = SimpleNamespace(display="Filters: test", label="test")

    @contextmanager
    def fake_context(*args, **kwargs):
        yield fig, ax, "out.png"

    with (
        patch("samplepath.plots.core.figure_context", side_effect=fake_context),
        patch("samplepath.plots.core.WPanel.render") as mock_render,
    ):
        core.WPanel().plot(metrics, filter_result, chart_config, "/tmp/out")
    mock_render.assert_called_once()


def test_plot_single_panel_H_calls_renderer():
    fig = MagicMock()
    ax = MagicMock()
    metrics = _metrics_fixture()
    chart_config = ChartConfig()
    filter_result = SimpleNamespace(display="Filters: test", label="test")

    @contextmanager
    def fake_context(*args, **kwargs):
        yield fig, ax, "out.png"

    with (
        patch("samplepath.plots.core.figure_context", side_effect=fake_context),
        patch("samplepath.plots.core.HPanel.render") as mock_render,
    ):
        core.HPanel().plot(metrics, filter_result, chart_config, "/tmp/out")
    mock_render.assert_called_once()


def test_plot_single_panel_Theta_calls_renderer():
    fig = MagicMock()
    ax = MagicMock()
    metrics = _metrics_fixture()
    chart_config = ChartConfig()
    filter_result = SimpleNamespace(display="Filters: test", label="test")

    @contextmanager
    def fake_context(*args, **kwargs):
        yield fig, ax, "out.png"

    with (
        patch("samplepath.plots.core.figure_context", side_effect=fake_context),
        patch("samplepath.plots.core.ThetaPanel.render") as mock_render,
    ):
        core.ThetaPanel().plot(metrics, filter_result, chart_config, "/tmp/out")
    mock_render.assert_called_once()


def test_plot_single_panel_w_prime_calls_renderer():
    fig = MagicMock()
    ax = MagicMock()
    metrics = _metrics_fixture()
    chart_config = ChartConfig()
    filter_result = SimpleNamespace(display="Filters: test", label="test")

    @contextmanager
    def fake_context(*args, **kwargs):
        yield fig, ax, "out.png"

    with (
        patch("samplepath.plots.core.figure_context", side_effect=fake_context),
        patch("samplepath.plots.core.WPrimePanel.render") as mock_render,
    ):
        core.WPrimePanel().plot(metrics, filter_result, chart_config, "/tmp/out")
    mock_render.assert_called_once()


def test_plot_single_panel_sojourn_calls_renderer():
    fig = MagicMock()
    ax = MagicMock()
    metrics = _metrics_fixture()
    empirical_metrics = _empirical_metrics_fixture(times=metrics.times)
    chart_config = ChartConfig()
    filter_result = SimpleNamespace(display="Filters: test", label="test")

    @contextmanager
    def fake_context(*args, **kwargs):
        yield fig, ax, "out.png"

    with (
        patch("samplepath.plots.core.figure_context", side_effect=fake_context),
        patch("samplepath.plots.core.SojournTimePanel.render") as mock_render,
    ):
        core.SojournTimePanel().plot(
            metrics, empirical_metrics, filter_result, chart_config, "/tmp/out"
        )
    mock_render.assert_called_once()


def _plot_H_panel_capture_render_kwargs(with_event_marks: bool = False):
    fig = MagicMock()
    ax = MagicMock()
    metrics = _metrics_fixture()
    chart_config = ChartConfig(with_event_marks=with_event_marks)
    filter_result = SimpleNamespace(display="Filters: test", label="test")

    @contextmanager
    def fake_context(*args, **kwargs):
        yield fig, ax, "out.png"

    with (
        patch("samplepath.plots.core.figure_context", side_effect=fake_context),
        patch("samplepath.plots.core.HPanel.render") as mock_render,
    ):
        core.HPanel(with_event_marks=with_event_marks).plot(
            metrics, filter_result, chart_config, "/tmp/out"
        )
    return mock_render.call_args.kwargs, metrics


def test_plot_H_passes_arrival_times():
    kwargs, metrics = _plot_H_panel_capture_render_kwargs(with_event_marks=True)
    assert kwargs["arrival_times"] == metrics.arrival_times


def test_plot_H_passes_departure_times():
    kwargs, metrics = _plot_H_panel_capture_render_kwargs(with_event_marks=True)
    assert kwargs["departure_times"] == metrics.departure_times


def test_plot_single_panel_CFD_calls_renderer():
    fig = MagicMock()
    ax = MagicMock()
    metrics = _metrics_fixture()
    chart_config = ChartConfig()
    filter_result = SimpleNamespace(display="Filters: test", label="test")

    @contextmanager
    def fake_context(*args, **kwargs):
        yield fig, ax, "out.png"

    with (
        patch("samplepath.plots.core.figure_context", side_effect=fake_context),
        patch("samplepath.plots.core.CFDPanel.render") as mock_render,
    ):
        core.CFDPanel().plot(metrics, filter_result, chart_config, "/tmp/out")
    mock_render.assert_called_once()


def test_CFDPanel_plot_uses_metrics_freq_for_unit():
    fig = MagicMock()
    ax = MagicMock()
    metrics = _metrics_fixture(freq="W")
    chart_config = ChartConfig()
    filter_result = SimpleNamespace(display="Filters: test", label="test")

    @contextmanager
    def fake_context(*args, **kwargs):
        yield fig, ax, "out.png"

    with patch("samplepath.plots.core.figure_context", side_effect=fake_context) as ctx:
        core.CFDPanel().plot(metrics, filter_result, chart_config, "/tmp/out")
    assert ctx.call_args.kwargs["unit"] == "W"


def test_CFDPanel_plot_uses_filter_caption():
    fig = MagicMock()
    ax = MagicMock()
    metrics = _metrics_fixture()
    chart_config = ChartConfig()
    filter_result = SimpleNamespace(display="Filters: test", label="test")

    @contextmanager
    def fake_context(*args, **kwargs):
        yield fig, ax, "out.png"

    with patch("samplepath.plots.core.figure_context", side_effect=fake_context) as ctx:
        core.CFDPanel().plot(metrics, filter_result, chart_config, "/tmp/out")
    assert ctx.call_args.kwargs["caption"] == "Filters: test"


def test_plot_CFD_uses_single_panel_layout():
    fig = MagicMock()
    ax = MagicMock()
    metrics = _metrics_fixture()
    chart_config = ChartConfig()
    filter_result = SimpleNamespace(display="Filters: test", label="test")

    @contextmanager
    def fake_context(*args, **kwargs):
        yield fig, ax, "out.png"

    with patch("samplepath.plots.core.figure_context", side_effect=fake_context) as ctx:
        core.CFDPanel().plot(metrics, filter_result, chart_config, "/tmp/out")
    assert ctx.call_args.kwargs["nrows"] == 1


def test_plot_core_stack_calls_all_renderers():
    fig = MagicMock()
    axes = np.array([object() for _ in range(4)], dtype=object)
    metrics = _metrics_fixture()
    chart_config = ChartConfig(
        with_event_marks=True,
        lambda_pctl_upper=99.0,
        lambda_pctl_lower=1.0,
        lambda_warmup_seconds=1800.0,
    )
    filter_result = SimpleNamespace(display="Filters: test", label="test")

    @contextmanager
    def fake_context(*args, **kwargs):
        yield fig, axes, "out.png"

    with (
        patch("samplepath.plots.core.layout_context", side_effect=fake_context),
        patch("samplepath.plots.core.NPanel.render") as mock_N,
        patch("samplepath.plots.core.LPanel.render") as mock_L,
        patch("samplepath.plots.core.LambdaPanel.render") as mock_Lam,
        patch("samplepath.plots.core.WPanel.render") as mock_w,
    ):
        core.plot_core_stack(
            metrics,
            filter_result,
            chart_config,
            "/tmp/out",
        )

    mock_N.assert_called_once()
    mock_L.assert_called_once()
    mock_Lam.assert_called_once()
    mock_w.assert_called_once()


def test_plot_core_stack_applies_layout_and_caption():
    fig = MagicMock()
    axes = np.array([object() for _ in range(4)], dtype=object)
    metrics = _metrics_fixture()
    chart_config = ChartConfig()
    filter_result = SimpleNamespace(display="Filters: test", label="test")
    outer_chart_config = chart_config

    @contextmanager
    def fake_context(
        out_path=None,
        *,
        layout,
        decor,
        unit,
        format_axis_fn,
        format_targets,
        chart_config=None,
        out_dir=None,
        subdir=None,
        base_name=None,
    ):
        assert out_path is None
        assert chart_config == outer_chart_config
        assert out_dir == "/tmp/out"
        assert subdir is None
        assert base_name == "sample_path_flow_metrics"
        assert layout.nrows == 4
        assert layout.ncols == 1
        assert layout.figsize == (12.0, 11.0)
        assert layout.sharex is True
        assert decor.suptitle == "Sample Path Flow Metrics"
        assert decor.caption == "Filters: test"
        assert decor.caption_position == "top"
        assert decor.caption_y == 0.945
        assert decor.tight_layout is True
        assert decor.tight_layout_rect == (0, 0, 1, 0.96)
        assert format_axis_fn is not None
        assert format_targets == "bottom_row"
        yield fig, axes, "out.png"

    with (
        patch("samplepath.plots.core.layout_context", side_effect=fake_context),
        patch("samplepath.plots.core.NPanel.render"),
        patch("samplepath.plots.core.LPanel.render"),
        patch("samplepath.plots.core.LambdaPanel.render"),
        patch("samplepath.plots.core.WPanel.render"),
    ):
        core.plot_core_stack(
            metrics,
            filter_result,
            chart_config,
            "/tmp/out",
        )


def test_plot_core_stack_uses_tighter_layout_without_caption():
    fig = MagicMock()
    axes = np.array([object() for _ in range(4)], dtype=object)
    metrics = _metrics_fixture()
    chart_config = ChartConfig()
    filter_result = SimpleNamespace(display="Filters: ", label="")
    outer_chart_config = chart_config

    @contextmanager
    def fake_context(
        out_path=None,
        *,
        layout,
        decor,
        unit,
        format_axis_fn,
        format_targets,
        chart_config=None,
        out_dir=None,
        subdir=None,
        base_name=None,
    ):
        assert out_path is None
        assert chart_config == outer_chart_config
        assert out_dir == "/tmp/out"
        assert subdir is None
        assert base_name == "sample_path_flow_metrics"
        assert decor.caption == "Filters: None"
        assert decor.tight_layout_rect == (0, 0, 1, 0.96)
        yield fig, axes, "out.png"

    with (
        patch("samplepath.plots.core.layout_context", side_effect=fake_context),
        patch("samplepath.plots.core.NPanel.render"),
        patch("samplepath.plots.core.LPanel.render"),
        patch("samplepath.plots.core.LambdaPanel.render"),
        patch("samplepath.plots.core.WPanel.render"),
    ):
        core.plot_core_stack(
            metrics,
            filter_result,
            chart_config,
            "/tmp/out",
        )


def _metrics_fixture(freq: str | None = "D"):
    return SimpleNamespace(
        times=[_t("2024-01-01")],
        N=np.array([1.0]),
        L=np.array([2.0]),
        Lambda=np.array([3.0]),
        w=np.array([4.0]),
        w_prime=np.array([4.5]),
        H=np.array([5.0]),
        Arrivals=np.array([1.0]),
        Departures=np.array([0.0]),
        Theta=np.array([0.0]),
        arrival_times=[_t("2024-01-01")],
        departure_times=[_t("2024-01-02")],
        freq=freq,
    )


def _empirical_metrics_fixture(
    *, times: Optional[list[pd.Timestamp]] = None, W_star: Optional[np.ndarray] = None
) -> ElementWiseEmpiricalMetrics:
    if times is None:
        times = [_t("2024-01-01")]
    if W_star is None:
        W_star = np.array([1.0])
    return ElementWiseEmpiricalMetrics(
        times=times,
        W_star=W_star,
        lam_star=np.array([0.25]),
        sojourn_vals=np.array([0.5]),
        residence_time_vals=np.array([1.0]),
        residence_completed=np.array([True]),
    )


def _llw_metrics(
    *,
    times: list[pd.Timestamp],
    L_vals: np.ndarray,
    Lam_vals: np.ndarray,
    w_vals: np.ndarray,
    arrival_times: Optional[list[pd.Timestamp]] = None,
    departure_times: Optional[list[pd.Timestamp]] = None,
):
    return SimpleNamespace(
        times=times,
        L=L_vals,
        Lambda=Lam_vals,
        w=w_vals,
        arrival_times=arrival_times or [],
        departure_times=departure_times or [],
        freq=None,
    )


def _ltheta_metrics(
    *,
    times: list[pd.Timestamp],
    L_vals: np.ndarray,
    Theta_vals: np.ndarray,
    w_prime_vals: np.ndarray,
    arrival_times: Optional[list[pd.Timestamp]] = None,
    departure_times: Optional[list[pd.Timestamp]] = None,
):
    return SimpleNamespace(
        times=times,
        L=L_vals,
        Theta=Theta_vals,
        w_prime=w_prime_vals,
        arrival_times=arrival_times or [],
        departure_times=departure_times or [],
        freq=None,
    )


def _fake_llw_context(fig, ax, *, caption: str, out_dir: str = "/tmp/out"):
    @contextmanager
    def _ctx(out_path=None, **kwargs):
        assert out_path is None
        assert kwargs["figsize"] == (6.0, 6.0)
        assert kwargs["caption"] == caption
        assert kwargs["unit"] is None
        assert kwargs["out_dir"] == out_dir
        assert kwargs["subdir"] == "core/panels"
        assert kwargs["base_name"] == "littles_law_invariant"
        chart_format = kwargs["chart_config"].chart_format
        yield fig, ax, resolve_chart_path(
            out_dir, "core/panels", "littles_law_invariant", chart_format
        )

    return _ctx


def _fake_ltheta_context(fig, ax, *, caption: str, out_dir: str = "/tmp/out"):
    @contextmanager
    def _ctx(out_path=None, **kwargs):
        assert out_path is None
        assert kwargs["figsize"] == (6.0, 6.0)
        assert kwargs["caption"] == caption
        assert kwargs["unit"] is None
        assert kwargs["out_dir"] == out_dir
        assert kwargs["subdir"] == "core/panels"
        assert kwargs["base_name"] == "departure_littles_law_invariant"
        chart_format = kwargs["chart_config"].chart_format
        yield fig, ax, resolve_chart_path(
            out_dir, "core/panels", "departure_littles_law_invariant", chart_format
        )

    return _ctx


def test_core_driver_returns_expected_paths():
    metrics = _metrics_fixture()
    empirical_metrics = _empirical_metrics_fixture(times=metrics.times)
    out_dir = "/tmp/out"
    args = SimpleNamespace(lambda_pctl=99.0, lambda_lower_pctl=1.0, lambda_warmup=0.5)
    chart_config = ChartConfig.init_from_args(args)
    filter_result = SimpleNamespace(display="Filters: test", label="test")
    expected = [
        resolve_chart_path(
            out_dir, "core/panels", "sample_path_N", chart_config.chart_format
        ),
        resolve_chart_path(
            out_dir, "core/panels", "time_average_N_L", chart_config.chart_format
        ),
        resolve_chart_path(
            out_dir,
            "core/panels",
            "cumulative_arrival_rate_Lambda",
            chart_config.chart_format,
        ),
        resolve_chart_path(
            out_dir,
            "core/panels",
            "cumulative_departure_rate_Theta",
            chart_config.chart_format,
        ),
        resolve_chart_path(
            out_dir,
            "core/panels",
            "arrival_departure_indicator_process",
            chart_config.chart_format,
        ),
        resolve_chart_path(
            out_dir,
            "core/panels",
            "cumulative_arrivals_A",
            chart_config.chart_format,
        ),
        resolve_chart_path(
            out_dir,
            "core/panels",
            "cumulative_departures_D",
            chart_config.chart_format,
        ),
        resolve_chart_path(
            out_dir,
            "core/panels",
            "average_residence_time_w",
            chart_config.chart_format,
        ),
        resolve_chart_path(
            out_dir,
            "core/panels",
            "sojourn_time_w_star",
            chart_config.chart_format,
        ),
        resolve_chart_path(
            out_dir,
            "core/panels",
            "sojourn_time_scatter",
            chart_config.chart_format,
        ),
        resolve_chart_path(
            out_dir,
            "core/panels",
            "residence_time_scatter",
            chart_config.chart_format,
        ),
        resolve_chart_path(
            out_dir,
            "core/panels",
            "average_residence_time_w_prime",
            chart_config.chart_format,
        ),
        resolve_chart_path(
            out_dir,
            "core/panels",
            "cumulative_presence_mass_H",
            chart_config.chart_format,
        ),
        resolve_chart_path(
            out_dir, "core/panels", "cumulative_flow_diagram", chart_config.chart_format
        ),
        resolve_chart_path(
            out_dir, "core/panels", "littles_law_invariant", chart_config.chart_format
        ),
        resolve_chart_path(
            out_dir,
            "core/panels",
            "departure_littles_law_invariant",
            chart_config.chart_format,
        ),
        resolve_chart_path(
            out_dir, None, "sample_path_flow_metrics", chart_config.chart_format
        ),
        resolve_chart_path(
            out_dir, "core", "lt_derivation_stack", chart_config.chart_format
        ),
        resolve_chart_path(
            out_dir, "core", "departure_flow_metrics", chart_config.chart_format
        ),
    ]
    with (
        patch("samplepath.plots.core.plot_core_stack") as mock_stack,
        patch("samplepath.plots.core.plot_LT_derivation_stack") as mock_lt_stack,
        patch(
            "samplepath.plots.core.plot_departure_flow_metrics_stack"
        ) as mock_departure_stack,
        patch("samplepath.plots.core.NPanel.plot") as mock_plot_N,
        patch("samplepath.plots.core.LPanel.plot") as mock_plot_L,
        patch("samplepath.plots.core.LambdaPanel.plot") as mock_plot_Lam,
        patch("samplepath.plots.core.ThetaPanel.plot") as mock_plot_Theta,
        patch("samplepath.plots.core.WPanel.plot") as mock_plot_w,
        patch("samplepath.plots.core.SojournTimePanel.plot") as mock_plot_w_star,
        patch(
            "samplepath.plots.core.SojournTimeScatterPanel.plot"
        ) as mock_plot_sojourn_scatter,
        patch(
            "samplepath.plots.core.ResidenceTimeScatterPanel.plot"
        ) as mock_plot_residence_scatter,
        patch("samplepath.plots.core.WPrimePanel.plot") as mock_plot_w_prime,
        patch("samplepath.plots.core.HPanel.plot") as mock_plot_H,
        patch("samplepath.plots.core.CFDPanel.plot") as mock_plot_CFD,
        patch("samplepath.plots.core.LLWPanel.plot") as mock_plot_llw,
        patch("samplepath.plots.core.LThetaWPrimePanel.plot") as mock_plot_ltheta,
        patch("samplepath.plots.core.EventIndicatorPanel.plot") as mock_plot_indicator,
        patch("samplepath.plots.core.ArrivalsPanel.plot") as mock_plot_A,
        patch("samplepath.plots.core.DeparturesPanel.plot") as mock_plot_D,
    ):
        mock_stack.return_value = expected[16]
        mock_lt_stack.return_value = expected[17]
        mock_departure_stack.return_value = expected[18]
        mock_plot_N.return_value = expected[0]
        mock_plot_L.return_value = expected[1]
        mock_plot_Lam.return_value = expected[2]
        mock_plot_Theta.return_value = expected[3]
        mock_plot_indicator.return_value = expected[4]
        mock_plot_A.return_value = expected[5]
        mock_plot_D.return_value = expected[6]
        mock_plot_w.return_value = expected[7]
        mock_plot_w_star.return_value = expected[8]
        mock_plot_sojourn_scatter.return_value = expected[9]
        mock_plot_residence_scatter.return_value = expected[10]
        mock_plot_w_prime.return_value = expected[11]
        mock_plot_H.return_value = expected[12]
        mock_plot_CFD.return_value = expected[13]
        mock_plot_llw.return_value = expected[14]
        mock_plot_ltheta.return_value = expected[15]
        written = core.plot_core_flow_metrics_charts(
            metrics, empirical_metrics, filter_result, chart_config, out_dir
        )
    assert written == expected
    mock_plot_N.assert_called_once()
    mock_plot_L.assert_called_once()
    mock_plot_Lam.assert_called_once()
    mock_plot_Theta.assert_called_once()
    mock_plot_w.assert_called_once()
    mock_plot_w_star.assert_called_once()
    mock_plot_sojourn_scatter.assert_called_once()
    mock_plot_residence_scatter.assert_called_once()
    mock_plot_w_prime.assert_called_once()
    mock_plot_H.assert_called_once()
    mock_plot_CFD.assert_called_once()
    mock_plot_llw.assert_called_once()
    mock_plot_ltheta.assert_called_once()
    mock_plot_indicator.assert_called_once()
    mock_plot_A.assert_called_once()
    mock_plot_D.assert_called_once()
    mock_lt_stack.assert_called_once()
    mock_departure_stack.assert_called_once()


def test_core_driver_calls_plot_core_stack_with_expected_args():
    metrics = _metrics_fixture()
    empirical_metrics = _empirical_metrics_fixture(times=metrics.times)
    out_dir = "/tmp/out"
    args = SimpleNamespace(
        lambda_pctl=98.0,
        lambda_lower_pctl=2.0,
        lambda_warmup=1.5,
        with_event_marks=True,
    )
    chart_config = ChartConfig.init_from_args(args)
    filter_result = SimpleNamespace(display="Filters: test", label="test")
    with (
        patch("samplepath.plots.core.plot_core_stack") as mock_stack,
        patch("samplepath.plots.core.plot_LT_derivation_stack"),
        patch("samplepath.plots.core.plot_departure_flow_metrics_stack"),
        patch("samplepath.plots.core.NPanel.plot") as mock_plot_N,
        patch("samplepath.plots.core.LPanel.plot") as mock_plot_L,
        patch("samplepath.plots.core.LambdaPanel.plot") as mock_plot_Lam,
        patch("samplepath.plots.core.ThetaPanel.plot") as mock_plot_Theta,
        patch("samplepath.plots.core.WPanel.plot") as mock_plot_w,
        patch("samplepath.plots.core.SojournTimePanel.plot") as mock_plot_w_star,
        patch(
            "samplepath.plots.core.SojournTimeScatterPanel.plot"
        ) as mock_plot_sojourn_scatter,
        patch(
            "samplepath.plots.core.ResidenceTimeScatterPanel.plot"
        ) as mock_plot_residence_scatter,
        patch("samplepath.plots.core.WPrimePanel.plot") as mock_plot_w_prime,
        patch("samplepath.plots.core.HPanel.plot") as mock_plot_H,
        patch("samplepath.plots.core.CFDPanel.plot") as mock_plot_CFD,
        patch("samplepath.plots.core.LLWPanel.plot"),
        patch("samplepath.plots.core.LThetaWPrimePanel.plot"),
        patch("samplepath.plots.core.EventIndicatorPanel.plot"),
        patch("samplepath.plots.core.ArrivalsPanel.plot"),
        patch("samplepath.plots.core.DeparturesPanel.plot"),
    ):
        core.plot_core_flow_metrics_charts(
            metrics, empirical_metrics, filter_result, chart_config, out_dir
        )
    mock_stack.assert_called_once_with(metrics, filter_result, chart_config, out_dir)
    mock_plot_N.assert_called_once()
    mock_plot_L.assert_called_once()
    mock_plot_Lam.assert_called_once()
    mock_plot_Theta.assert_called_once()
    mock_plot_w.assert_called_once()
    mock_plot_w_star.assert_called_once()
    mock_plot_sojourn_scatter.assert_called_once()
    mock_plot_residence_scatter.assert_called_once()
    mock_plot_w_prime.assert_called_once()
    mock_plot_H.assert_called_once()
    mock_plot_CFD.assert_called_once()


def test_core_driver_passes_event_marks_to_Lambda_and_w():
    metrics = _metrics_fixture()
    empirical_metrics = _empirical_metrics_fixture(times=metrics.times)
    out_dir = "/tmp/out"
    args = SimpleNamespace(
        lambda_pctl=99.0,
        lambda_lower_pctl=1.0,
        lambda_warmup=0.5,
        with_event_marks=True,
    )
    chart_config = ChartConfig.init_from_args(args)
    filter_result = SimpleNamespace(display="Filters: test", label="test")
    with (
        patch("samplepath.plots.core.plot_core_stack"),
        patch("samplepath.plots.core.plot_LT_derivation_stack"),
        patch("samplepath.plots.core.plot_departure_flow_metrics_stack"),
        patch("samplepath.plots.core.NPanel.plot") as mock_plot_N,
        patch("samplepath.plots.core.LPanel.plot") as mock_plot_L,
        patch("samplepath.plots.core.HPanel.plot"),
        patch("samplepath.plots.core.CFDPanel") as mock_cfd_cls,
        patch("samplepath.plots.core.LLWPanel"),
        patch("samplepath.plots.core.LThetaWPrimePanel"),
        patch("samplepath.plots.core.EventIndicatorPanel"),
        patch("samplepath.plots.core.ArrivalsPanel"),
        patch("samplepath.plots.core.DeparturesPanel"),
        patch("samplepath.plots.core.LambdaPanel") as mock_lam_cls,
        patch("samplepath.plots.core.ThetaPanel") as mock_theta_cls,
        patch("samplepath.plots.core.WPanel") as mock_w_cls,
        patch("samplepath.plots.core.SojournTimePanel") as mock_w_star_cls,
        patch(
            "samplepath.plots.core.SojournTimeScatterPanel"
        ) as mock_sojourn_scatter_cls,
        patch(
            "samplepath.plots.core.ResidenceTimeScatterPanel"
        ) as mock_residence_scatter_cls,
        patch("samplepath.plots.core.WPrimePanel") as mock_w_prime_cls,
    ):
        core.plot_core_flow_metrics_charts(
            metrics, empirical_metrics, filter_result, chart_config, out_dir
        )
    assert mock_lam_cls.call_args.kwargs["with_event_marks"] is True
    assert mock_theta_cls.call_args.kwargs["with_event_marks"] is True
    assert mock_w_cls.call_args.kwargs["with_event_marks"] is True
    assert mock_w_star_cls.call_args.kwargs["with_event_marks"] is True
    assert mock_sojourn_scatter_cls.call_args.kwargs["with_event_marks"] is True
    assert mock_residence_scatter_cls.call_args.kwargs["with_event_marks"] is True
    assert mock_w_prime_cls.call_args.kwargs["with_event_marks"] is True
    assert mock_cfd_cls.call_args.kwargs["with_event_marks"] is True
    mock_plot_N.assert_called_once()
    mock_plot_L.assert_called_once()
    mock_lam_cls.assert_called_once()
    mock_theta_cls.assert_called_once()
    mock_w_cls.assert_called_once()
    mock_w_prime_cls.assert_called_once()


def test_core_driver_passes_show_derivations_to_CFD():
    metrics = _metrics_fixture()
    empirical_metrics = _empirical_metrics_fixture(times=metrics.times)
    out_dir = "/tmp/out"
    args = SimpleNamespace(
        lambda_pctl=99.0,
        lambda_lower_pctl=1.0,
        lambda_warmup=0.5,
        show_derivations=True,
    )
    chart_config = ChartConfig.init_from_args(args)
    filter_result = SimpleNamespace(display="Filters: test", label="test")
    with (
        patch("samplepath.plots.core.plot_core_stack"),
        patch("samplepath.plots.core.plot_LT_derivation_stack"),
        patch("samplepath.plots.core.plot_departure_flow_metrics_stack"),
        patch("samplepath.plots.core.NPanel.plot") as mock_plot_N,
        patch("samplepath.plots.core.LPanel.plot") as mock_plot_L,
        patch("samplepath.plots.core.LambdaPanel.plot") as mock_plot_Lam,
        patch("samplepath.plots.core.ThetaPanel.plot") as mock_plot_Theta,
        patch("samplepath.plots.core.WPanel.plot") as mock_plot_w,
        patch("samplepath.plots.core.SojournTimePanel.plot") as mock_plot_w_star,
        patch("samplepath.plots.core.SojournTimeScatterPanel.plot"),
        patch("samplepath.plots.core.ResidenceTimeScatterPanel.plot"),
        patch("samplepath.plots.core.WPrimePanel.plot") as mock_plot_w_prime,
        patch("samplepath.plots.core.HPanel.plot") as mock_plot_H,
        patch("samplepath.plots.core.LLWPanel.plot"),
        patch("samplepath.plots.core.LThetaWPrimePanel.plot"),
        patch("samplepath.plots.core.CFDPanel") as mock_cfd_cls,
        patch("samplepath.plots.core.EventIndicatorPanel.plot"),
        patch("samplepath.plots.core.ArrivalsPanel.plot"),
        patch("samplepath.plots.core.DeparturesPanel.plot"),
    ):
        core.plot_core_flow_metrics_charts(
            metrics, empirical_metrics, filter_result, chart_config, out_dir
        )
    assert mock_cfd_cls.call_args.kwargs["show_derivations"] is True
    mock_plot_N.assert_called_once()
    mock_plot_L.assert_called_once()
    mock_plot_Lam.assert_called_once()
    mock_plot_Theta.assert_called_once()
    mock_plot_w.assert_called_once()
    mock_plot_w_star.assert_called_once()
    mock_plot_w_prime.assert_called_once()
    mock_plot_H.assert_called_once()


def test_core_driver_uses_metrics_freq_for_unit():
    metrics = _metrics_fixture(freq="W")
    empirical_metrics = _empirical_metrics_fixture(times=metrics.times)
    out_dir = "/tmp/out"
    args = SimpleNamespace(lambda_pctl=99.0, lambda_lower_pctl=1.0, lambda_warmup=0.5)
    chart_config = ChartConfig.init_from_args(args)
    filter_result = SimpleNamespace(display="Filters: test", label="test")
    with (
        patch("samplepath.plots.core.plot_core_stack"),
        patch("samplepath.plots.core.plot_LT_derivation_stack"),
        patch("samplepath.plots.core.plot_departure_flow_metrics_stack"),
        patch("samplepath.plots.core.NPanel.plot") as mock_plot,
    ):
        with (
            patch("samplepath.plots.core.LPanel.plot"),
            patch("samplepath.plots.core.LambdaPanel.plot"),
            patch("samplepath.plots.core.ThetaPanel.plot"),
            patch("samplepath.plots.core.WPanel.plot"),
            patch("samplepath.plots.core.SojournTimePanel.plot"),
            patch("samplepath.plots.core.SojournTimeScatterPanel.plot"),
            patch("samplepath.plots.core.ResidenceTimeScatterPanel.plot"),
            patch("samplepath.plots.core.WPrimePanel.plot"),
            patch("samplepath.plots.core.HPanel.plot"),
            patch("samplepath.plots.core.CFDPanel.plot"),
            patch("samplepath.plots.core.LLWPanel.plot"),
            patch("samplepath.plots.core.LThetaWPrimePanel.plot"),
            patch("samplepath.plots.core.EventIndicatorPanel.plot"),
            patch("samplepath.plots.core.ArrivalsPanel.plot"),
            patch("samplepath.plots.core.DeparturesPanel.plot"),
        ):
            core.plot_core_flow_metrics_charts(
                metrics, empirical_metrics, filter_result, chart_config, out_dir
            )
    assert mock_plot.call_args.args[2] == chart_config


def test_LambdaPanel_plot_falls_back_to_timestamp_unit():
    fig = MagicMock()
    ax = MagicMock()
    metrics = _metrics_fixture(freq=None)
    chart_config = ChartConfig()
    filter_result = SimpleNamespace(display="Filters: test", label="test")

    @contextmanager
    def fake_context(*args, **kwargs):
        yield fig, ax, "out.png"

    with patch("samplepath.plots.core.figure_context", side_effect=fake_context) as ctx:
        core.LambdaPanel().plot(metrics, filter_result, chart_config, "/tmp/out")
    assert ctx.call_args.kwargs["unit"] == "timestamp"


def test_NPanel_plot_uses_metrics_freq_for_unit():
    fig = MagicMock()
    ax = MagicMock()
    metrics = _metrics_fixture(freq="W")

    @contextmanager
    def fake_context(*args, **kwargs):
        yield fig, ax, "out.png"

    with patch("samplepath.plots.core.figure_context", side_effect=fake_context) as ctx:
        core.NPanel().plot(
            metrics,
            SimpleNamespace(display="Filters: test", label="test"),
            ChartConfig(),
            "/tmp/out",
        )
    assert ctx.call_args.kwargs["unit"] == "W"


def test_NPanel_plot_falls_back_to_timestamp_unit():
    fig = MagicMock()
    ax = MagicMock()
    metrics = _metrics_fixture(freq=None)

    @contextmanager
    def fake_context(*args, **kwargs):
        yield fig, ax, "out.png"

    with patch("samplepath.plots.core.figure_context", side_effect=fake_context) as ctx:
        core.NPanel().plot(
            metrics,
            SimpleNamespace(display="Filters: test", label="test"),
            ChartConfig(),
            "/tmp/out",
        )
    assert ctx.call_args.kwargs["unit"] == "timestamp"


def test_WPanel_plot_uses_filter_display_caption():
    fig = MagicMock()
    ax = MagicMock()
    metrics = _metrics_fixture()
    chart_config = ChartConfig()
    filter_result = SimpleNamespace(display="Filters: test", label="test")

    @contextmanager
    def fake_context(*args, **kwargs):
        yield fig, ax, "out.png"

    with patch("samplepath.plots.core.figure_context", side_effect=fake_context) as ctx:
        core.WPanel().plot(metrics, filter_result, chart_config, "/tmp/out")
    assert ctx.call_args.kwargs["caption"] == "Filters: test"


def test_core_driver_calls_plot_H_under_core_dir():
    metrics = _metrics_fixture()
    empirical_metrics = _empirical_metrics_fixture(times=metrics.times)
    out_dir = "/tmp/out"
    args = SimpleNamespace(lambda_pctl=99.0, lambda_lower_pctl=1.0, lambda_warmup=0.5)
    chart_config = ChartConfig.init_from_args(args)
    filter_result = SimpleNamespace(display="Filters: test", label="test")
    with (
        patch("samplepath.plots.core.plot_core_stack"),
        patch("samplepath.plots.core.plot_LT_derivation_stack"),
        patch("samplepath.plots.core.plot_departure_flow_metrics_stack"),
        patch("samplepath.plots.core.HPanel.plot") as mock_plot,
    ):
        with (
            patch("samplepath.plots.core.NPanel.plot"),
            patch("samplepath.plots.core.LPanel.plot"),
            patch("samplepath.plots.core.LambdaPanel.plot"),
            patch("samplepath.plots.core.ThetaPanel.plot"),
            patch("samplepath.plots.core.WPanel.plot"),
            patch("samplepath.plots.core.SojournTimePanel.plot"),
            patch("samplepath.plots.core.SojournTimeScatterPanel.plot"),
            patch("samplepath.plots.core.ResidenceTimeScatterPanel.plot"),
            patch("samplepath.plots.core.WPrimePanel.plot"),
            patch("samplepath.plots.core.CFDPanel.plot"),
            patch("samplepath.plots.core.LLWPanel.plot"),
            patch("samplepath.plots.core.LThetaWPrimePanel.plot"),
            patch("samplepath.plots.core.EventIndicatorPanel.plot"),
            patch("samplepath.plots.core.ArrivalsPanel.plot"),
            patch("samplepath.plots.core.DeparturesPanel.plot"),
        ):
            core.plot_core_flow_metrics_charts(
                metrics, empirical_metrics, filter_result, chart_config, out_dir
            )
    assert mock_plot.call_args.args[-1] == out_dir


def test_core_driver_calls_plot_CFD_under_core_dir():
    out_dir = "/tmp/out"
    args = SimpleNamespace(lambda_pctl=None, lambda_lower_pctl=None, lambda_warmup=0.0)
    chart_config = ChartConfig.init_from_args(args)
    filter_result = SimpleNamespace(display="Filters: test", label="test")
    metrics = SimpleNamespace(
        times=[_t("2024-01-01")],
        Arrivals=np.array([1.0]),
        Departures=np.array([0.0]),
        N=np.array([1.0]),
        L=np.array([1.0]),
        Lambda=np.array([1.0]),
        w=np.array([1.0]),
        H=np.array([1.0]),
        arrival_times=[],
        departure_times=[],
        freq=None,
    )
    empirical_metrics = _empirical_metrics_fixture(times=metrics.times)
    with (
        patch("samplepath.plots.core.plot_core_stack"),
        patch("samplepath.plots.core.plot_LT_derivation_stack"),
        patch("samplepath.plots.core.plot_departure_flow_metrics_stack"),
        patch("samplepath.plots.core.NPanel.plot"),
        patch("samplepath.plots.core.LPanel.plot"),
        patch("samplepath.plots.core.LambdaPanel.plot"),
        patch("samplepath.plots.core.ThetaPanel.plot"),
        patch("samplepath.plots.core.WPanel.plot"),
        patch("samplepath.plots.core.SojournTimePanel.plot"),
        patch("samplepath.plots.core.SojournTimeScatterPanel.plot"),
        patch("samplepath.plots.core.ResidenceTimeScatterPanel.plot"),
        patch("samplepath.plots.core.WPrimePanel.plot"),
        patch("samplepath.plots.core.HPanel.plot"),
        patch("samplepath.plots.core.LLWPanel.plot"),
        patch("samplepath.plots.core.LThetaWPrimePanel.plot"),
        patch("samplepath.plots.core.CFDPanel.plot") as mock_plot,
        patch("samplepath.plots.core.EventIndicatorPanel.plot"),
        patch("samplepath.plots.core.ArrivalsPanel.plot"),
        patch("samplepath.plots.core.DeparturesPanel.plot"),
    ):
        core.plot_core_flow_metrics_charts(
            metrics, empirical_metrics, filter_result, chart_config, out_dir
        )
    assert mock_plot.call_args.args[-1] == out_dir


def test_core_driver_passes_event_marks_to_CFD():
    metrics = _metrics_fixture()
    empirical_metrics = _empirical_metrics_fixture(times=metrics.times)
    out_dir = "/tmp/out"
    args = SimpleNamespace(
        lambda_pctl=None,
        lambda_lower_pctl=None,
        lambda_warmup=0.0,
        with_event_marks=True,
    )
    chart_config = ChartConfig.init_from_args(args)
    filter_result = SimpleNamespace(display="Filters: test", label="test")
    with (
        patch("samplepath.plots.core.plot_core_stack"),
        patch("samplepath.plots.core.plot_LT_derivation_stack"),
        patch("samplepath.plots.core.plot_departure_flow_metrics_stack"),
        patch("samplepath.plots.core.NPanel.plot"),
        patch("samplepath.plots.core.LPanel.plot"),
        patch("samplepath.plots.core.LambdaPanel.plot"),
        patch("samplepath.plots.core.ThetaPanel.plot"),
        patch("samplepath.plots.core.WPanel.plot"),
        patch("samplepath.plots.core.SojournTimePanel.plot"),
        patch("samplepath.plots.core.SojournTimeScatterPanel.plot"),
        patch("samplepath.plots.core.ResidenceTimeScatterPanel.plot"),
        patch("samplepath.plots.core.WPrimePanel.plot"),
        patch("samplepath.plots.core.HPanel.plot"),
        patch("samplepath.plots.core.LLWPanel"),
        patch("samplepath.plots.core.LThetaWPrimePanel"),
        patch("samplepath.plots.core.CFDPanel") as mock_cfd_cls,
        patch("samplepath.plots.core.EventIndicatorPanel"),
        patch("samplepath.plots.core.ArrivalsPanel"),
        patch("samplepath.plots.core.DeparturesPanel"),
    ):
        core.plot_core_flow_metrics_charts(
            metrics, empirical_metrics, filter_result, chart_config, out_dir
        )
    assert mock_cfd_cls.call_args.kwargs["with_event_marks"] is True


def test_core_driver_passes_event_marks_to_departure_invariant():
    metrics = _metrics_fixture()
    empirical_metrics = _empirical_metrics_fixture(times=metrics.times)
    out_dir = "/tmp/out"
    args = SimpleNamespace(
        lambda_pctl=None,
        lambda_lower_pctl=None,
        lambda_warmup=0.0,
        with_event_marks=True,
    )
    chart_config = ChartConfig.init_from_args(args)
    filter_result = SimpleNamespace(display="Filters: test", label="test")
    with (
        patch("samplepath.plots.core.plot_core_stack"),
        patch("samplepath.plots.core.plot_LT_derivation_stack"),
        patch("samplepath.plots.core.plot_departure_flow_metrics_stack"),
        patch("samplepath.plots.core.NPanel.plot"),
        patch("samplepath.plots.core.LPanel.plot"),
        patch("samplepath.plots.core.LambdaPanel.plot"),
        patch("samplepath.plots.core.ThetaPanel.plot"),
        patch("samplepath.plots.core.WPanel.plot"),
        patch("samplepath.plots.core.SojournTimePanel.plot"),
        patch("samplepath.plots.core.SojournTimeScatterPanel.plot"),
        patch("samplepath.plots.core.ResidenceTimeScatterPanel.plot"),
        patch("samplepath.plots.core.WPrimePanel.plot"),
        patch("samplepath.plots.core.HPanel.plot"),
        patch("samplepath.plots.core.CFDPanel.plot"),
        patch("samplepath.plots.core.LLWPanel.plot"),
        patch("samplepath.plots.core.LThetaWPrimePanel") as mock_panel,
        patch("samplepath.plots.core.EventIndicatorPanel.plot"),
        patch("samplepath.plots.core.ArrivalsPanel.plot"),
        patch("samplepath.plots.core.DeparturesPanel.plot"),
    ):
        core.plot_core_flow_metrics_charts(
            metrics, empirical_metrics, filter_result, chart_config, out_dir
        )
    assert mock_panel.call_args.kwargs["with_event_marks"] is True


def test_LLWPanel_renders_invariant_chart():
    fig = MagicMock()
    ax = MagicMock()
    times = [_t("2024-01-01"), _t("2024-01-02")]
    metrics = _llw_metrics(
        times=times,
        L_vals=np.array([1.0, 2.0]),
        Lam_vals=np.array([1.0, 1.5]),
        w_vals=np.array([2.0, 1.0]),
        arrival_times=[times[0]],
        departure_times=[times[1]],
    )
    filter_result = SimpleNamespace(display="Filters: test", label="test")
    expected_path = resolve_chart_path(
        "/tmp/out", "core/panels", "littles_law_invariant", "png"
    )
    with patch(
        "samplepath.plots.core.figure_context",
        side_effect=_fake_llw_context(fig, ax, caption="Filters: test"),
    ):
        written = core.LLWPanel(title="L(T) vs Λ(T).w(T)").plot(
            metrics, filter_result, ChartConfig(), "/tmp/out"
        )
    ax.plot.assert_called_once()
    ax.set_aspect.assert_called_once_with("equal", adjustable="box")
    ax.grid.assert_not_called()
    ax.set_xlabel.assert_called_once_with("L(T)")
    ax.set_ylabel.assert_called_once_with("Λ(T)·w(T)")
    ax.set_title.assert_called_once_with("L(T) vs Λ(T).w(T)")
    assert written == expected_path


def test_LThetaWPrimePanel_writes_expected_path():
    fig = MagicMock()
    ax = MagicMock()
    times = [_t("2024-01-01"), _t("2024-01-02")]
    metrics = _ltheta_metrics(
        times=times,
        L_vals=np.array([1.0, 2.0]),
        Theta_vals=np.array([1.0, 2.0]),
        w_prime_vals=np.array([2.0, 1.0]),
        arrival_times=[times[0]],
        departure_times=[times[1]],
    )
    filter_result = SimpleNamespace(display="Filters: test", label="test")
    expected_path = resolve_chart_path(
        "/tmp/out", "core/panels", "departure_littles_law_invariant", "png"
    )
    with patch(
        "samplepath.plots.core.figure_context",
        side_effect=_fake_ltheta_context(fig, ax, caption="Filters: test"),
    ):
        written = core.LThetaWPrimePanel(title="L(T) vs Θ(T).w'(T)").plot(
            metrics, filter_result, ChartConfig(), "/tmp/out"
        )
    assert written == expected_path


def test_LThetaWPrimePanel_sets_ylabel():
    fig = MagicMock()
    ax = MagicMock()
    times = [_t("2024-01-01")]
    metrics = _ltheta_metrics(
        times=times,
        L_vals=np.array([1.0]),
        Theta_vals=np.array([2.0]),
        w_prime_vals=np.array([3.0]),
    )
    filter_result = SimpleNamespace(display="Filters: test", label="test")
    with patch(
        "samplepath.plots.core.figure_context",
        side_effect=_fake_ltheta_context(fig, ax, caption="Filters: test"),
    ):
        core.LThetaWPrimePanel().plot(metrics, filter_result, ChartConfig(), "/tmp/out")
    assert ax.set_ylabel.call_args.args[0] == "Θ(T)·w'(T)"


def test_LThetaWPrimePanel_scatter_uses_theta_w_prime_product():
    fig = MagicMock()
    ax = MagicMock()
    times = [_t("2024-01-01"), _t("2024-01-02")]
    Theta_vals = np.array([2.0, 3.0])
    w_prime_vals = np.array([4.0, 5.0])
    metrics = _ltheta_metrics(
        times=times,
        L_vals=np.array([1.0, 2.0]),
        Theta_vals=Theta_vals,
        w_prime_vals=w_prime_vals,
    )
    filter_result = SimpleNamespace(display="Filters: test", label="test")
    with patch(
        "samplepath.plots.core.figure_context",
        side_effect=_fake_ltheta_context(fig, ax, caption="Filters: test"),
    ):
        core.LThetaWPrimePanel().plot(metrics, filter_result, ChartConfig(), "/tmp/out")
    plotted = ax.scatter.call_args.args[1]
    assert np.allclose(plotted, Theta_vals * w_prime_vals)


def test_LLWPanel_skips_reference_line_on_nonfinite():
    fig = MagicMock()
    ax = MagicMock()
    times = [_t("2024-01-01")]
    metrics = _llw_metrics(
        times=times,
        L_vals=np.array([np.nan]),
        Lam_vals=np.array([np.nan]),
        w_vals=np.array([np.nan]),
    )
    filter_result = SimpleNamespace(display="Filters: test", label="test")
    with patch(
        "samplepath.plots.core.figure_context",
        side_effect=_fake_llw_context(fig, ax, caption="Filters: test"),
    ):
        core.LLWPanel().plot(metrics, filter_result, ChartConfig(), "/tmp/out")
    ax.plot.assert_not_called()
    ax.scatter.assert_not_called()


def test_LLWPanel_event_marks_colors_arrivals_purple():
    fig = MagicMock()
    ax = MagicMock()
    times = [_t("2024-01-01"), _t("2024-01-02")]
    metrics = _llw_metrics(
        times=times,
        L_vals=np.array([1.0, 2.0]),
        Lam_vals=np.array([1.0, 1.0]),
        w_vals=np.array([1.0, 1.0]),
        arrival_times=[times[0]],
        departure_times=[times[1]],
    )
    filter_result = SimpleNamespace(display="Filters: test", label="test")
    with patch(
        "samplepath.plots.core.figure_context",
        side_effect=_fake_llw_context(fig, ax, caption="Filters: test"),
    ):
        core.LLWPanel(with_event_marks=True).plot(
            metrics, filter_result, ChartConfig(), "/tmp/out"
        )
    colors = ax.scatter.call_args.kwargs["color"]
    assert colors[0][:3] == mcolors.to_rgba("purple")[:3]


def test_LLWPanel_event_marks_colors_departures_green():
    fig = MagicMock()
    ax = MagicMock()
    times = [_t("2024-01-01"), _t("2024-01-02")]
    metrics = _llw_metrics(
        times=times,
        L_vals=np.array([1.0, 2.0]),
        Lam_vals=np.array([1.0, 1.0]),
        w_vals=np.array([1.0, 1.0]),
        arrival_times=[times[0]],
        departure_times=[times[1]],
    )
    filter_result = SimpleNamespace(display="Filters: test", label="test")
    with patch(
        "samplepath.plots.core.figure_context",
        side_effect=_fake_llw_context(fig, ax, caption="Filters: test"),
    ):
        core.LLWPanel(with_event_marks=True).plot(
            metrics, filter_result, ChartConfig(), "/tmp/out"
        )
    colors = ax.scatter.call_args.kwargs["color"]
    assert colors[1][:3] == mcolors.to_rgba("green")[:3]


def test_LLWPanel_event_marks_alpha_increases():
    fig = MagicMock()
    ax = MagicMock()
    times = [_t("2024-01-01"), _t("2024-01-02")]
    metrics = _llw_metrics(
        times=times,
        L_vals=np.array([1.0, 2.0]),
        Lam_vals=np.array([1.0, 1.0]),
        w_vals=np.array([1.0, 1.0]),
        arrival_times=[times[0]],
        departure_times=[times[1]],
    )
    filter_result = SimpleNamespace(display="Filters: test", label="test")
    with patch(
        "samplepath.plots.core.figure_context",
        side_effect=_fake_llw_context(fig, ax, caption="Filters: test"),
    ):
        core.LLWPanel(with_event_marks=True).plot(
            metrics, filter_result, ChartConfig(), "/tmp/out"
        )
    colors = ax.scatter.call_args.kwargs["color"]
    assert colors[0][3] < colors[1][3]


def test_LLWPanel_event_marks_drop_lines_arrival_color():
    fig = MagicMock()
    ax = MagicMock()
    times = [_t("2024-01-01"), _t("2024-01-02")]
    metrics = _llw_metrics(
        times=times,
        L_vals=np.array([1.0, 2.0]),
        Lam_vals=np.array([1.0, 1.0]),
        w_vals=np.array([1.0, 1.0]),
        arrival_times=[times[0]],
        departure_times=[times[1]],
    )
    filter_result = SimpleNamespace(display="Filters: test", label="test")
    with patch(
        "samplepath.plots.core.figure_context",
        side_effect=_fake_llw_context(fig, ax, caption="Filters: test"),
    ):
        core.LLWPanel(with_event_marks=True).plot(
            metrics, filter_result, ChartConfig(), "/tmp/out"
        )
    colors = ax.vlines.call_args.kwargs["colors"]
    assert np.allclose(colors[0], mcolors.to_rgba("purple", alpha=0.25))


def test_LLWPanel_event_marks_drop_lines_departure_color():
    fig = MagicMock()
    ax = MagicMock()
    times = [_t("2024-01-01"), _t("2024-01-02")]
    metrics = _llw_metrics(
        times=times,
        L_vals=np.array([1.0, 2.0]),
        Lam_vals=np.array([1.0, 1.0]),
        w_vals=np.array([1.0, 1.0]),
        arrival_times=[times[0]],
        departure_times=[times[1]],
    )
    filter_result = SimpleNamespace(display="Filters: test", label="test")
    with patch(
        "samplepath.plots.core.figure_context",
        side_effect=_fake_llw_context(fig, ax, caption="Filters: test"),
    ):
        core.LLWPanel(with_event_marks=True).plot(
            metrics, filter_result, ChartConfig(), "/tmp/out"
        )
    colors = ax.vlines.call_args.kwargs["colors"]
    assert np.allclose(colors, [mcolors.to_rgba("purple", alpha=0.25)])


def test_LLWPanel_event_marks_drop_lines_only_for_arrivals():
    fig = MagicMock()
    ax = MagicMock()
    times = [_t("2024-01-01"), _t("2024-01-02")]
    metrics = _llw_metrics(
        times=times,
        L_vals=np.array([1.0, 2.0]),
        Lam_vals=np.array([1.0, 1.0]),
        w_vals=np.array([1.0, 1.0]),
        arrival_times=[times[0]],
        departure_times=[times[1]],
    )
    filter_result = SimpleNamespace(display="Filters: test", label="test")
    with patch(
        "samplepath.plots.core.figure_context",
        side_effect=_fake_llw_context(fig, ax, caption="Filters: test"),
    ):
        core.LLWPanel(with_event_marks=True).plot(
            metrics, filter_result, ChartConfig(), "/tmp/out"
        )
    vlines_x = ax.vlines.call_args.args[0]
    assert np.allclose(vlines_x, np.array([1.0]))


def test_LLWPanel_event_marks_hlines_arrival_color():
    fig = MagicMock()
    ax = MagicMock()
    times = [_t("2024-01-01"), _t("2024-01-02")]
    metrics = _llw_metrics(
        times=times,
        L_vals=np.array([1.0, 2.0]),
        Lam_vals=np.array([1.0, 1.0]),
        w_vals=np.array([1.0, 1.0]),
        arrival_times=[times[0]],
        departure_times=[times[1]],
    )
    filter_result = SimpleNamespace(display="Filters: test", label="test")
    with patch(
        "samplepath.plots.core.figure_context",
        side_effect=_fake_llw_context(fig, ax, caption="Filters: test"),
    ):
        core.LLWPanel(with_event_marks=True).plot(
            metrics, filter_result, ChartConfig(), "/tmp/out"
        )
    colors = ax.hlines.call_args.kwargs["colors"]
    assert np.allclose(colors[0], mcolors.to_rgba("purple", alpha=0.25))


def test_LLWPanel_event_marks_hlines_departure_color():
    fig = MagicMock()
    ax = MagicMock()
    times = [_t("2024-01-01"), _t("2024-01-02")]
    metrics = _llw_metrics(
        times=times,
        L_vals=np.array([1.0, 2.0]),
        Lam_vals=np.array([1.0, 1.0]),
        w_vals=np.array([1.0, 1.0]),
        arrival_times=[times[0]],
        departure_times=[times[1]],
    )
    filter_result = SimpleNamespace(display="Filters: test", label="test")
    with patch(
        "samplepath.plots.core.figure_context",
        side_effect=_fake_llw_context(fig, ax, caption="Filters: test"),
    ):
        core.LLWPanel(with_event_marks=True).plot(
            metrics, filter_result, ChartConfig(), "/tmp/out"
        )
    colors = ax.hlines.call_args.kwargs["colors"]
    assert np.allclose(colors, [mcolors.to_rgba("purple", alpha=0.25)])


def test_LLWPanel_event_marks_hlines_only_for_arrivals():
    fig = MagicMock()
    ax = MagicMock()
    times = [_t("2024-01-01"), _t("2024-01-02")]
    metrics = _llw_metrics(
        times=times,
        L_vals=np.array([1.0, 2.0]),
        Lam_vals=np.array([1.0, 1.0]),
        w_vals=np.array([1.0, 1.0]),
        arrival_times=[times[0]],
        departure_times=[times[1]],
    )
    filter_result = SimpleNamespace(display="Filters: test", label="test")
    with patch(
        "samplepath.plots.core.figure_context",
        side_effect=_fake_llw_context(fig, ax, caption="Filters: test"),
    ):
        core.LLWPanel(with_event_marks=True).plot(
            metrics, filter_result, ChartConfig(), "/tmp/out"
        )
    hlines_y = ax.hlines.call_args.args[0]
    assert np.allclose(hlines_y, np.array([1.0]))


def test_LThetaWPrimePanel_event_marks_drop_lines_only_for_departures():
    fig = MagicMock()
    ax = MagicMock()
    times = [_t("2024-01-01"), _t("2024-01-02")]
    metrics = _ltheta_metrics(
        times=times,
        L_vals=np.array([1.0, 2.0]),
        Theta_vals=np.array([1.0, 1.0]),
        w_prime_vals=np.array([1.0, 1.0]),
        arrival_times=[times[0]],
        departure_times=[times[1]],
    )
    filter_result = SimpleNamespace(display="Filters: test", label="test")
    with patch(
        "samplepath.plots.core.figure_context",
        side_effect=_fake_ltheta_context(fig, ax, caption="Filters: test"),
    ):
        core.LThetaWPrimePanel(with_event_marks=True).plot(
            metrics, filter_result, ChartConfig(), "/tmp/out"
        )
    vlines_x = ax.vlines.call_args.args[0]
    assert np.allclose(vlines_x, np.array([2.0]))


def test_LThetaWPrimePanel_event_marks_hlines_only_for_departures():
    fig = MagicMock()
    ax = MagicMock()
    times = [_t("2024-01-01"), _t("2024-01-02")]
    metrics = _ltheta_metrics(
        times=times,
        L_vals=np.array([1.0, 2.0]),
        Theta_vals=np.array([1.0, 1.0]),
        w_prime_vals=np.array([1.0, 1.0]),
        arrival_times=[times[0]],
        departure_times=[times[1]],
    )
    filter_result = SimpleNamespace(display="Filters: test", label="test")
    with patch(
        "samplepath.plots.core.figure_context",
        side_effect=_fake_ltheta_context(fig, ax, caption="Filters: test"),
    ):
        core.LThetaWPrimePanel(with_event_marks=True).plot(
            metrics, filter_result, ChartConfig(), "/tmp/out"
        )
    hlines_y = ax.hlines.call_args.args[0]
    assert np.allclose(hlines_y, np.array([1.0]))


def test_LLWPanel_event_marks_adds_legend():
    fig = MagicMock()
    ax = MagicMock()
    times = [_t("2024-01-01"), _t("2024-01-02")]
    metrics = _llw_metrics(
        times=times,
        L_vals=np.array([1.0, 2.0]),
        Lam_vals=np.array([1.0, 1.0]),
        w_vals=np.array([1.0, 1.0]),
        arrival_times=[times[0]],
        departure_times=[times[1]],
    )
    filter_result = SimpleNamespace(display="Filters: test", label="test")
    with patch(
        "samplepath.plots.core.figure_context",
        side_effect=_fake_llw_context(fig, ax, caption="Filters: test"),
    ):
        core.LLWPanel(with_event_marks=True).plot(
            metrics, filter_result, ChartConfig(), "/tmp/out"
        )
    ax.legend.assert_called_once()


def test_LLWPanel_no_event_marks_alpha_increases():
    fig = MagicMock()
    ax = MagicMock()
    times = [_t("2024-01-01"), _t("2024-01-02")]
    metrics = _llw_metrics(
        times=times,
        L_vals=np.array([1.0, 2.0]),
        Lam_vals=np.array([1.0, 1.0]),
        w_vals=np.array([1.0, 1.0]),
    )
    filter_result = SimpleNamespace(display="Filters: test", label="test")
    with patch(
        "samplepath.plots.core.figure_context",
        side_effect=_fake_llw_context(fig, ax, caption="Filters: test"),
    ):
        core.LLWPanel(with_event_marks=False).plot(
            metrics, filter_result, ChartConfig(), "/tmp/out"
        )
    colors = ax.scatter.call_args.kwargs["color"]
    assert colors[0][3] < colors[1][3]


def test_LLWPanel_no_event_marks_has_no_legend():
    fig = MagicMock()
    ax = MagicMock()
    times = [_t("2024-01-01"), _t("2024-01-02")]
    metrics = _llw_metrics(
        times=times,
        L_vals=np.array([1.0, 2.0]),
        Lam_vals=np.array([1.0, 1.0]),
        w_vals=np.array([1.0, 1.0]),
    )
    filter_result = SimpleNamespace(display="Filters: test", label="test")
    with patch(
        "samplepath.plots.core.figure_context",
        side_effect=_fake_llw_context(fig, ax, caption="Filters: test"),
    ):
        core.LLWPanel(with_event_marks=False).plot(
            metrics, filter_result, ChartConfig(), "/tmp/out"
        )
    ax.legend.assert_not_called()


def test_LLWPanel_departure_overrides_arrival_color():
    fig = MagicMock()
    ax = MagicMock()
    times = [_t("2024-01-01")]
    metrics = _llw_metrics(
        times=times,
        L_vals=np.array([1.0]),
        Lam_vals=np.array([1.0]),
        w_vals=np.array([1.0]),
        arrival_times=times,
        departure_times=times,
    )
    filter_result = SimpleNamespace(display="Filters: test", label="test")
    with patch(
        "samplepath.plots.core.figure_context",
        side_effect=_fake_llw_context(fig, ax, caption="Filters: test"),
    ):
        core.LLWPanel(with_event_marks=True).plot(
            metrics, filter_result, ChartConfig(), "/tmp/out"
        )
    colors = ax.scatter.call_args.kwargs["color"]
    assert colors[0][:3] == mcolors.to_rgba("green")[:3]


def test_core_driver_calls_invariant_plot_under_core_dir():
    metrics = _metrics_fixture()
    empirical_metrics = _empirical_metrics_fixture(times=metrics.times)
    out_dir = "/tmp/out"
    args = SimpleNamespace(lambda_pctl=99.0, lambda_lower_pctl=1.0, lambda_warmup=0.5)
    chart_config = ChartConfig.init_from_args(args)
    filter_result = SimpleNamespace(display="Filters: test", label="test")
    with (
        patch("samplepath.plots.core.plot_core_stack"),
        patch("samplepath.plots.core.plot_LT_derivation_stack"),
        patch("samplepath.plots.core.plot_departure_flow_metrics_stack"),
        patch("samplepath.plots.core.LLWPanel") as mock_panel,
        patch("samplepath.plots.core.LThetaWPrimePanel"),
    ):
        with (
            patch("samplepath.plots.core.NPanel.plot"),
            patch("samplepath.plots.core.LPanel.plot"),
            patch("samplepath.plots.core.LambdaPanel.plot"),
            patch("samplepath.plots.core.ThetaPanel.plot"),
            patch("samplepath.plots.core.WPanel.plot"),
            patch("samplepath.plots.core.SojournTimePanel.plot"),
            patch("samplepath.plots.core.SojournTimeScatterPanel.plot"),
            patch("samplepath.plots.core.ResidenceTimeScatterPanel.plot"),
            patch("samplepath.plots.core.WPrimePanel.plot"),
            patch("samplepath.plots.core.HPanel.plot"),
            patch("samplepath.plots.core.CFDPanel.plot"),
            patch("samplepath.plots.core.EventIndicatorPanel.plot"),
            patch("samplepath.plots.core.ArrivalsPanel.plot"),
            patch("samplepath.plots.core.DeparturesPanel.plot"),
        ):
            core.plot_core_flow_metrics_charts(
                metrics, empirical_metrics, filter_result, chart_config, out_dir
            )
    mock_panel.assert_called_once_with(
        with_event_marks=False,
    )
    mock_panel.return_value.plot.assert_called_once_with(
        metrics,
        filter_result,
        chart_config,
        out_dir,
    )


def test_core_driver_calls_departure_invariant_plot_under_core_dir():
    metrics = _metrics_fixture()
    empirical_metrics = _empirical_metrics_fixture(times=metrics.times)
    out_dir = "/tmp/out"
    args = SimpleNamespace(lambda_pctl=99.0, lambda_lower_pctl=1.0, lambda_warmup=0.5)
    chart_config = ChartConfig.init_from_args(args)
    filter_result = SimpleNamespace(display="Filters: test", label="test")
    with (
        patch("samplepath.plots.core.plot_core_stack"),
        patch("samplepath.plots.core.plot_LT_derivation_stack"),
        patch("samplepath.plots.core.plot_departure_flow_metrics_stack"),
        patch("samplepath.plots.core.LLWPanel"),
        patch("samplepath.plots.core.LThetaWPrimePanel") as mock_panel,
    ):
        with (
            patch("samplepath.plots.core.NPanel.plot"),
            patch("samplepath.plots.core.LPanel.plot"),
            patch("samplepath.plots.core.LambdaPanel.plot"),
            patch("samplepath.plots.core.ThetaPanel.plot"),
            patch("samplepath.plots.core.WPanel.plot"),
            patch("samplepath.plots.core.SojournTimePanel.plot"),
            patch("samplepath.plots.core.SojournTimeScatterPanel.plot"),
            patch("samplepath.plots.core.ResidenceTimeScatterPanel.plot"),
            patch("samplepath.plots.core.WPrimePanel.plot"),
            patch("samplepath.plots.core.HPanel.plot"),
            patch("samplepath.plots.core.CFDPanel.plot"),
            patch("samplepath.plots.core.EventIndicatorPanel.plot"),
            patch("samplepath.plots.core.ArrivalsPanel.plot"),
            patch("samplepath.plots.core.DeparturesPanel.plot"),
        ):
            core.plot_core_flow_metrics_charts(
                metrics, empirical_metrics, filter_result, chart_config, out_dir
            )
    assert mock_panel.call_args.kwargs["with_event_marks"] is False


def test_core_driver_omits_caption_when_label_empty():
    metrics = _metrics_fixture()
    empirical_metrics = _empirical_metrics_fixture(times=metrics.times)
    out_dir = "/tmp/out"
    args = SimpleNamespace(lambda_pctl=99.0, lambda_lower_pctl=1.0, lambda_warmup=0.5)
    chart_config = ChartConfig.init_from_args(args)
    filter_result = SimpleNamespace(display="Filters: ", label="")
    with (
        patch("samplepath.plots.core.plot_core_stack") as mock_stack,
        patch("samplepath.plots.core.plot_LT_derivation_stack"),
        patch("samplepath.plots.core.plot_departure_flow_metrics_stack"),
        patch("samplepath.plots.core.NPanel.plot"),
        patch("samplepath.plots.core.LPanel.plot"),
        patch("samplepath.plots.core.LambdaPanel.plot"),
        patch("samplepath.plots.core.ThetaPanel.plot"),
        patch("samplepath.plots.core.WPanel.plot"),
        patch("samplepath.plots.core.SojournTimePanel.plot"),
        patch("samplepath.plots.core.SojournTimeScatterPanel.plot"),
        patch("samplepath.plots.core.ResidenceTimeScatterPanel.plot"),
        patch("samplepath.plots.core.WPrimePanel.plot"),
        patch("samplepath.plots.core.HPanel.plot"),
        patch("samplepath.plots.core.CFDPanel.plot"),
        patch("samplepath.plots.core.LLWPanel.plot"),
        patch("samplepath.plots.core.LThetaWPrimePanel.plot"),
        patch("samplepath.plots.core.EventIndicatorPanel.plot"),
        patch("samplepath.plots.core.ArrivalsPanel.plot"),
        patch("samplepath.plots.core.DeparturesPanel.plot"),
    ):
        core.plot_core_flow_metrics_charts(
            metrics, empirical_metrics, filter_result, chart_config, out_dir
        )
    mock_stack.assert_called_once_with(metrics, filter_result, chart_config, out_dir)


def _call_LT_derivation_stack_with_mocks():
    """Helper to call plot_LT_derivation_stack with standard mocks."""
    fig = MagicMock()
    axes = np.array([object() for _ in range(4)], dtype=object)
    metrics = _metrics_fixture()
    chart_config = ChartConfig(with_event_marks=True, show_derivations=True)
    filter_result = SimpleNamespace(display="Filters: test", label="test")

    @contextmanager
    def fake_context(*args, **kwargs):
        yield fig, axes, "out.png"

    mocks = {}
    with (
        patch("samplepath.plots.core.layout_context", side_effect=fake_context),
        patch("samplepath.plots.core.CFDPanel.render") as mock_CFD,
        patch("samplepath.plots.core.NPanel.render") as mock_N,
        patch("samplepath.plots.core.HPanel.render") as mock_H,
        patch("samplepath.plots.core.LPanel.render") as mock_L,
    ):
        core.plot_LT_derivation_stack(metrics, filter_result, chart_config, "/tmp/out")
        mocks["CFD"] = mock_CFD
        mocks["N"] = mock_N
        mocks["H"] = mock_H
        mocks["L"] = mock_L
    return mocks


def test_plot_LT_derivation_stack_calls_CFD_render():
    mocks = _call_LT_derivation_stack_with_mocks()
    mocks["CFD"].assert_called_once()


def test_plot_LT_derivation_stack_calls_N_render():
    mocks = _call_LT_derivation_stack_with_mocks()
    mocks["N"].assert_called_once()


def test_plot_LT_derivation_stack_calls_H_render():
    mocks = _call_LT_derivation_stack_with_mocks()
    mocks["H"].assert_called_once()


def test_plot_LT_derivation_stack_calls_L_render():
    mocks = _call_LT_derivation_stack_with_mocks()
    mocks["L"].assert_called_once()


def _capture_LT_derivation_stack_layout_context():
    """Helper to capture layout_context kwargs from plot_LT_derivation_stack."""
    fig = MagicMock()
    axes = np.array([object() for _ in range(4)], dtype=object)
    metrics = _metrics_fixture()
    chart_config = ChartConfig()
    filter_result = SimpleNamespace(display="Filters: test", label="test")
    captured = {}

    @contextmanager
    def fake_context(out_path=None, **kwargs):
        captured["out_path"] = out_path
        captured.update(kwargs)
        yield fig, axes, "out.png"

    with (
        patch("samplepath.plots.core.layout_context", side_effect=fake_context),
        patch("samplepath.plots.core.CFDPanel.render"),
        patch("samplepath.plots.core.NPanel.render"),
        patch("samplepath.plots.core.HPanel.render"),
        patch("samplepath.plots.core.LPanel.render"),
    ):
        core.plot_LT_derivation_stack(metrics, filter_result, chart_config, "/tmp/out")
    captured["outer_chart_config"] = chart_config
    return captured


def _call_departure_flow_metrics_stack_with_mocks():
    fig = MagicMock()
    axes = np.array([object() for _ in range(4)], dtype=object)
    metrics = _metrics_fixture()
    chart_config = ChartConfig(with_event_marks=True, show_derivations=True)
    filter_result = SimpleNamespace(display="Filters: test", label="test")

    @contextmanager
    def fake_context(*args, **kwargs):
        yield fig, axes, "out.png"

    mocks = {}
    with (
        patch("samplepath.plots.core.layout_context", side_effect=fake_context),
        patch("samplepath.plots.core.NPanel.render") as mock_N,
        patch("samplepath.plots.core.LPanel.render") as mock_L,
        patch("samplepath.plots.core.ThetaPanel.render") as mock_Theta,
        patch("samplepath.plots.core.WPrimePanel.render") as mock_w_prime,
    ):
        core.plot_departure_flow_metrics_stack(
            metrics, filter_result, chart_config, "/tmp/out"
        )
        mocks["N"] = mock_N
        mocks["L"] = mock_L
        mocks["Theta"] = mock_Theta
        mocks["w_prime"] = mock_w_prime
    return mocks


def _capture_departure_flow_metrics_stack_layout_context():
    fig = MagicMock()
    axes = np.array([object() for _ in range(4)], dtype=object)
    metrics = _metrics_fixture()
    chart_config = ChartConfig()
    filter_result = SimpleNamespace(display="Filters: test", label="test")
    captured = {}

    @contextmanager
    def fake_context(out_path=None, **kwargs):
        captured["out_path"] = out_path
        captured.update(kwargs)
        yield fig, axes, "out.png"

    with (
        patch("samplepath.plots.core.layout_context", side_effect=fake_context),
        patch("samplepath.plots.core.NPanel.render"),
        patch("samplepath.plots.core.LPanel.render"),
        patch("samplepath.plots.core.ThetaPanel.render"),
        patch("samplepath.plots.core.WPrimePanel.render"),
    ):
        core.plot_departure_flow_metrics_stack(
            metrics, filter_result, chart_config, "/tmp/out"
        )
    captured["outer_chart_config"] = chart_config
    return captured


def test_plot_LT_derivation_stack_out_path_is_none():
    captured = _capture_LT_derivation_stack_layout_context()
    assert captured["out_path"] is None


def test_plot_LT_derivation_stack_passes_chart_config():
    captured = _capture_LT_derivation_stack_layout_context()
    assert captured["chart_config"] == captured["outer_chart_config"]


def test_plot_LT_derivation_stack_uses_correct_out_dir():
    captured = _capture_LT_derivation_stack_layout_context()
    assert captured["out_dir"] == "/tmp/out"


def test_plot_LT_derivation_stack_uses_core_subdir():
    captured = _capture_LT_derivation_stack_layout_context()
    assert captured["subdir"] == "core"


def test_plot_LT_derivation_stack_uses_correct_base_name():
    captured = _capture_LT_derivation_stack_layout_context()
    assert captured["base_name"] == "lt_derivation_stack"


def test_plot_LT_derivation_stack_layout_has_4_rows():
    captured = _capture_LT_derivation_stack_layout_context()
    assert captured["layout"].nrows == 4


def test_plot_LT_derivation_stack_layout_has_1_col():
    captured = _capture_LT_derivation_stack_layout_context()
    assert captured["layout"].ncols == 1


def test_plot_LT_derivation_stack_layout_figsize():
    captured = _capture_LT_derivation_stack_layout_context()
    assert captured["layout"].figsize == (12.0, 11.0)


def test_plot_LT_derivation_stack_layout_sharex():
    captured = _capture_LT_derivation_stack_layout_context()
    assert captured["layout"].sharex is True


def test_plot_LT_derivation_stack_suptitle():
    captured = _capture_LT_derivation_stack_layout_context()
    assert captured["decor"].suptitle == "L(T) Derivation from Cumulative Flow Diagram"


def test_plot_LT_derivation_stack_caption():
    captured = _capture_LT_derivation_stack_layout_context()
    assert captured["decor"].caption == "Filters: test"


def test_plot_LT_derivation_stack_format_axis_fn_set():
    captured = _capture_LT_derivation_stack_layout_context()
    assert captured["format_axis_fn"] is not None


def test_plot_LT_derivation_stack_format_targets():
    captured = _capture_LT_derivation_stack_layout_context()
    assert captured["format_targets"] == "bottom_row"


def test_plot_departure_flow_metrics_stack_calls_N_render():
    mocks = _call_departure_flow_metrics_stack_with_mocks()
    mocks["N"].assert_called_once()


def test_plot_departure_flow_metrics_stack_calls_L_render():
    mocks = _call_departure_flow_metrics_stack_with_mocks()
    mocks["L"].assert_called_once()


def test_plot_departure_flow_metrics_stack_calls_Theta_render():
    mocks = _call_departure_flow_metrics_stack_with_mocks()
    mocks["Theta"].assert_called_once()


def test_plot_departure_flow_metrics_stack_calls_w_prime_render():
    mocks = _call_departure_flow_metrics_stack_with_mocks()
    mocks["w_prime"].assert_called_once()


def test_plot_departure_flow_metrics_stack_out_path_is_none():
    captured = _capture_departure_flow_metrics_stack_layout_context()
    assert captured["out_path"] is None


def test_plot_departure_flow_metrics_stack_passes_chart_config():
    captured = _capture_departure_flow_metrics_stack_layout_context()
    assert captured["chart_config"] == captured["outer_chart_config"]


def test_plot_departure_flow_metrics_stack_uses_correct_out_dir():
    captured = _capture_departure_flow_metrics_stack_layout_context()
    assert captured["out_dir"] == "/tmp/out"


def test_plot_departure_flow_metrics_stack_uses_correct_base_name():
    captured = _capture_departure_flow_metrics_stack_layout_context()
    assert captured["base_name"] == "departure_flow_metrics"


def test_plot_departure_flow_metrics_stack_uses_core_subdir():
    captured = _capture_departure_flow_metrics_stack_layout_context()
    assert captured["subdir"] == "core"


def test_plot_departure_flow_metrics_stack_layout_has_4_rows():
    captured = _capture_departure_flow_metrics_stack_layout_context()
    assert captured["layout"].nrows == 4


def test_plot_departure_flow_metrics_stack_layout_has_1_col():
    captured = _capture_departure_flow_metrics_stack_layout_context()
    assert captured["layout"].ncols == 1


def test_plot_departure_flow_metrics_stack_layout_figsize():
    captured = _capture_departure_flow_metrics_stack_layout_context()
    assert captured["layout"].figsize == (12.0, 11.0)


def test_plot_departure_flow_metrics_stack_layout_sharex():
    captured = _capture_departure_flow_metrics_stack_layout_context()
    assert captured["layout"].sharex is True


def test_plot_departure_flow_metrics_stack_suptitle():
    captured = _capture_departure_flow_metrics_stack_layout_context()
    assert captured["decor"].suptitle == "Departure-Focused Flow Metrics"


def test_plot_departure_flow_metrics_stack_caption():
    captured = _capture_departure_flow_metrics_stack_layout_context()
    assert captured["decor"].caption == "Filters: test"


def test_plot_departure_flow_metrics_stack_format_axis_fn_set():
    captured = _capture_departure_flow_metrics_stack_layout_context()
    assert captured["format_axis_fn"] is not None


def test_plot_departure_flow_metrics_stack_format_targets():
    captured = _capture_departure_flow_metrics_stack_layout_context()
    assert captured["format_targets"] == "bottom_row"


def _call_LT_derivation_stack_capturing_panel_classes(
    with_event_marks: bool = True, show_derivations: bool = True
):
    """Helper to capture panel constructor kwargs from plot_LT_derivation_stack."""
    fig = MagicMock()
    axes = np.array([object() for _ in range(4)], dtype=object)
    metrics = _metrics_fixture()
    chart_config = ChartConfig(
        with_event_marks=with_event_marks,
        show_derivations=show_derivations,
    )
    filter_result = SimpleNamespace(display="Filters: test", label="test")

    @contextmanager
    def fake_context(*args, **kwargs):
        yield fig, axes, "out.png"

    mocks = {}
    with (
        patch("samplepath.plots.core.layout_context", side_effect=fake_context),
        patch("samplepath.plots.core.CFDPanel") as mock_CFD,
        patch("samplepath.plots.core.NPanel") as mock_N,
        patch("samplepath.plots.core.HPanel") as mock_H,
        patch("samplepath.plots.core.LPanel") as mock_L,
    ):
        core.plot_LT_derivation_stack(metrics, filter_result, chart_config, "/tmp/out")
        mocks["CFD"] = mock_CFD
        mocks["N"] = mock_N
        mocks["H"] = mock_H
        mocks["L"] = mock_L
    return mocks


def test_plot_LT_derivation_stack_passes_event_marks_to_CFD():
    mocks = _call_LT_derivation_stack_capturing_panel_classes(with_event_marks=True)
    assert mocks["CFD"].call_args.kwargs["with_event_marks"] is True


def test_plot_LT_derivation_stack_passes_show_derivations_to_CFD():
    mocks = _call_LT_derivation_stack_capturing_panel_classes(show_derivations=True)
    assert mocks["CFD"].call_args.kwargs["show_derivations"] is True


def test_plot_LT_derivation_stack_passes_event_marks_to_N():
    mocks = _call_LT_derivation_stack_capturing_panel_classes(with_event_marks=True)
    assert mocks["N"].call_args.kwargs["with_event_marks"] is True


def test_plot_LT_derivation_stack_passes_show_derivations_to_N():
    mocks = _call_LT_derivation_stack_capturing_panel_classes(show_derivations=True)
    assert mocks["N"].call_args.kwargs["show_derivations"] is True


def test_plot_LT_derivation_stack_passes_show_derivations_to_H():
    mocks = _call_LT_derivation_stack_capturing_panel_classes(show_derivations=True)
    assert mocks["H"].call_args.kwargs["show_derivations"] is True


def test_plot_LT_derivation_stack_passes_event_marks_to_H():
    mocks = _call_LT_derivation_stack_capturing_panel_classes(with_event_marks=True)
    assert mocks["H"].call_args.kwargs["with_event_marks"] is True


def test_plot_LT_derivation_stack_passes_event_marks_to_L():
    mocks = _call_LT_derivation_stack_capturing_panel_classes(with_event_marks=True)
    assert mocks["L"].call_args.kwargs["with_event_marks"] is True


def test_plot_LT_derivation_stack_passes_show_derivations_to_L():
    mocks = _call_LT_derivation_stack_capturing_panel_classes(show_derivations=True)
    assert mocks["L"].call_args.kwargs["show_derivations"] is True


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# ChartConfig sampling_frequency + panel passthrough tests
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def test_chart_config_sampling_frequency_defaults_to_none():
    config = ChartConfig()
    assert config.sampling_frequency is None


def test_chart_config_init_from_args_reads_sampling_frequency():
    args = SimpleNamespace(sampling_frequency="week")
    config = ChartConfig.init_from_args(args)
    assert config.sampling_frequency == "week"


def test_LPanel_passes_sampling_frequency_to_render_line_chart():
    ax = MagicMock()
    times = [_t("2024-01-01")]
    values = np.array([2.0])
    with patch("samplepath.plots.core.render_line_chart") as mock_render:
        core.LPanel(sampling_frequency="week").render(ax, times, values)
    _, kwargs = mock_render.call_args
    assert kwargs["sampling_frequency"] == "week"
