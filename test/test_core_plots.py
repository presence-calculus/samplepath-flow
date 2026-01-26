# -*- coding: utf-8 -*-
# Copyright (c) 2025 Krishna Kumar
# SPDX-License-Identifier: MIT
from contextlib import contextmanager
import os
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from matplotlib import colors as mcolors
import numpy as np
import pandas as pd

from samplepath.plots import core
from samplepath.plots.chart_config import ChartConfig
from samplepath.plots.core import ClipOptions


def _t(s: str) -> pd.Timestamp:
    return pd.Timestamp(s)


def test_render_N_colors_grey_when_overlays():
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
    assert kwargs["color"] == "grey"
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


def test_render_L_colors_grey_when_overlays():
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
    assert kwargs["color"] == "grey"
    assert kwargs["overlays"] == overlays
    ax.set_title.assert_called_once()
    ax.set_ylabel.assert_called_once_with("L(T)")


def test_render_Lambda_passes_clip_options():
    ax = MagicMock()
    times = [_t("2024-01-01"), _t("2024-01-02")]
    values = np.array([1.0, 2.0])
    clip = ClipOptions(pctl_upper=99.0, pctl_lower=1.0, warmup_hours=0.5)
    with (
        patch("samplepath.plots.core.render_line_chart") as mock_line,
        patch("samplepath.plots.core._clip_axis_to_percentile") as mock_clip,
    ):
        core.LambdaPanel(clip_opts=clip).render(ax, times, values)
    mock_line.assert_called_once()
    mock_clip.assert_called_once_with(ax, list(times), values, 99.0, 1.0, 0.5)
    ax.set_ylabel.assert_called_once_with("Λ(T) [1/hr]")


def test_render_w_sets_defaults():
    ax = MagicMock()
    times = [_t("2024-01-01")]
    values = np.array([0.5])
    with patch("samplepath.plots.core.render_line_chart") as mock_line:
        core.WPanel().render(ax, times, values)
    mock_line.assert_called_once()
    ax.set_ylabel.assert_called_once_with("w(T) [hrs]")


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
        yield fig, ax

    with (
        patch("samplepath.plots.core.figure_context", side_effect=fake_context),
        patch("samplepath.plots.core.NPanel.render") as mock_render,
    ):
        core.NPanel(with_event_marks=True).plot(
            None,
            ChartConfig(with_event_marks=True),
            SimpleNamespace(display="Filters: test", label="test"),
            _metrics_fixture(),
            "/tmp/out",
        )
    mock_render.assert_called_once()


def test_plot_single_panel_L_calls_renderer():
    fig = MagicMock()
    ax = MagicMock()

    @contextmanager
    def fake_context(*args, **kwargs):
        yield fig, ax

    with (
        patch("samplepath.plots.core.figure_context", side_effect=fake_context),
        patch("samplepath.plots.core.LPanel.plot") as mock_render,
    ):
        core.LPanel().plot("out.png", [_t("2024-01-01")], np.array([2.0]))
    mock_render.assert_called_once()


def test_plot_single_panel_Lambda_calls_renderer():
    fig = MagicMock()
    ax = MagicMock()

    @contextmanager
    def fake_context(*args, **kwargs):
        yield fig, ax

    with (
        patch("samplepath.plots.core.figure_context", side_effect=fake_context),
        patch("samplepath.plots.core.LambdaPanel.plot") as mock_render,
    ):
        core.LambdaPanel().plot("out.png", [_t("2024-01-01")], np.array([3.0]))
    mock_render.assert_called_once()


def test_plot_single_panel_w_calls_renderer():
    fig = MagicMock()
    ax = MagicMock()

    @contextmanager
    def fake_context(*args, **kwargs):
        yield fig, ax

    with (
        patch("samplepath.plots.core.figure_context", side_effect=fake_context),
        patch("samplepath.plots.core.WPanel.plot") as mock_render,
    ):
        core.WPanel().plot("out.png", [_t("2024-01-01")], np.array([4.0]))
    mock_render.assert_called_once()


def test_plot_single_panel_H_calls_renderer():
    fig = MagicMock()
    ax = MagicMock()

    @contextmanager
    def fake_context(*args, **kwargs):
        yield fig, ax

    with (
        patch("samplepath.plots.core.figure_context", side_effect=fake_context),
        patch("samplepath.plots.core.HPanel.plot") as mock_render,
    ):
        core.HPanel().plot("out.png", [_t("2024-01-01")], np.array([5.0]))
    mock_render.assert_called_once()


def test_plot_single_panel_CFD_calls_renderer():
    fig = MagicMock()
    ax = MagicMock()

    @contextmanager
    def fake_context(*args, **kwargs):
        yield fig, ax

    with (
        patch("samplepath.plots.core.figure_context", side_effect=fake_context),
        patch("samplepath.plots.core.CFDPanel.render") as mock_render,
    ):
        core.CFDPanel().plot(
            "out.png",
            [_t("2024-01-01")],
            np.array([1.0]),
            np.array([0.0]),
            arrival_times=[_t("2024-01-01")],
            departure_times=[_t("2024-01-01")],
        )
    mock_render.assert_called_once()


def test_plot_CFD_passes_unit():
    fig = MagicMock()
    ax = MagicMock()

    @contextmanager
    def fake_context(*args, **kwargs):
        yield fig, ax

    with patch("samplepath.plots.core.figure_context", side_effect=fake_context) as ctx:
        core.CFDPanel().plot(
            "out.png",
            [_t("2024-01-01")],
            np.array([1.0]),
            np.array([0.0]),
            unit="W",
        )
    assert ctx.call_args.kwargs["unit"] == "W"


def test_plot_CFD_passes_caption():
    fig = MagicMock()
    ax = MagicMock()

    @contextmanager
    def fake_context(*args, **kwargs):
        yield fig, ax

    with patch("samplepath.plots.core.figure_context", side_effect=fake_context) as ctx:
        core.CFDPanel().plot(
            "out.png",
            [_t("2024-01-01")],
            np.array([1.0]),
            np.array([0.0]),
            caption="Filters: test",
        )
    assert ctx.call_args.kwargs["caption"] == "Filters: test"


def test_plot_CFD_uses_single_panel_layout():
    fig = MagicMock()
    ax = MagicMock()

    @contextmanager
    def fake_context(*args, **kwargs):
        yield fig, ax

    with patch("samplepath.plots.core.figure_context", side_effect=fake_context) as ctx:
        core.CFDPanel().plot(
            "out.png",
            [_t("2024-01-01")],
            np.array([1.0]),
            np.array([0.0]),
        )
    assert ctx.call_args.kwargs["nrows"] == 1


def test_plot_core_stack_calls_all_renderers():
    fig = MagicMock()
    axes = np.array([object() for _ in range(4)], dtype=object)
    metrics = _metrics_fixture()
    chart_config = ChartConfig(
        with_event_marks=True,
        lambda_pctl_upper=99.0,
        lambda_pctl_lower=1.0,
        lambda_warmup_hours=0.5,
    )
    filter_result = SimpleNamespace(display="Filters: test", label="test")

    @contextmanager
    def fake_context(*args, **kwargs):
        yield fig, axes

    with (
        patch("samplepath.plots.core.layout_context", side_effect=fake_context),
        patch("samplepath.plots.core.NPanel.render") as mock_N,
        patch("samplepath.plots.core.LPanel.render") as mock_L,
        patch("samplepath.plots.core.LambdaPanel.render") as mock_Lam,
        patch("samplepath.plots.core.WPanel.render") as mock_w,
    ):
        core.plot_core_stack(
            None,
            chart_config,
            filter_result,
            metrics,
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

    @contextmanager
    def fake_context(out_path, *, layout, decor, unit, format_axis_fn, format_targets):
        assert out_path == "/tmp/out/sample_path_flow_metrics.png"
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
        yield fig, axes

    with (
        patch("samplepath.plots.core.layout_context", side_effect=fake_context),
        patch("samplepath.plots.core.NPanel.render"),
        patch("samplepath.plots.core.LPanel.render"),
        patch("samplepath.plots.core.LambdaPanel.render"),
        patch("samplepath.plots.core.WPanel.render"),
    ):
        core.plot_core_stack(
            None,
            chart_config,
            filter_result,
            metrics,
            "/tmp/out",
        )


def test_plot_core_stack_uses_tighter_layout_without_caption():
    fig = MagicMock()
    axes = np.array([object() for _ in range(4)], dtype=object)
    metrics = _metrics_fixture()
    chart_config = ChartConfig()
    filter_result = SimpleNamespace(display="Filters: ", label="")

    @contextmanager
    def fake_context(out_path, *, layout, decor, unit, format_axis_fn, format_targets):
        assert decor.caption == "Filters: None"
        assert decor.tight_layout_rect == (0, 0, 1, 0.96)
        yield fig, axes

    with (
        patch("samplepath.plots.core.layout_context", side_effect=fake_context),
        patch("samplepath.plots.core.NPanel.render"),
        patch("samplepath.plots.core.LPanel.render"),
        patch("samplepath.plots.core.LambdaPanel.render"),
        patch("samplepath.plots.core.WPanel.render"),
    ):
        core.plot_core_stack(
            None,
            chart_config,
            filter_result,
            metrics,
            "/tmp/out",
        )


def _metrics_fixture(freq: str | None = "D"):
    return SimpleNamespace(
        times=[_t("2024-01-01")],
        N=np.array([1.0]),
        L=np.array([2.0]),
        Lambda=np.array([3.0]),
        w=np.array([4.0]),
        H=np.array([5.0]),
        Arrivals=np.array([1.0]),
        Departures=np.array([0.0]),
        arrival_times=[_t("2024-01-01")],
        departure_times=[_t("2024-01-02")],
        freq=freq,
    )


def test_core_driver_returns_expected_paths():
    metrics = _metrics_fixture()
    out_dir = "/tmp/out"
    args = SimpleNamespace(lambda_pctl=99.0, lambda_lower_pctl=1.0, lambda_warmup=0.5)
    chart_config = ChartConfig.init_from_args(args)
    filter_result = SimpleNamespace(display="Filters: test", label="test")
    expected = [
        os.path.join(out_dir, "core/sample_path_N.png"),
        os.path.join(out_dir, "core/time_average_N_L.png"),
        os.path.join(out_dir, "core/cumulative_arrival_rate_Lambda.png"),
        os.path.join(out_dir, "core/average_residence_time_w.png"),
        os.path.join(out_dir, "core/cumulative_presence_mass_H.png"),
        os.path.join(out_dir, "core/cumulative_flow_diagram.png"),
        os.path.join(out_dir, "core/littles_law_invariant.png"),
        os.path.join(out_dir, "sample_path_flow_metrics.png"),
    ]
    with (
        patch("samplepath.plots.core.plot_core_stack"),
        patch("samplepath.plots.core.NPanel.plot") as mock_plot_N,
        patch("samplepath.plots.core.LPanel.plot") as mock_plot_L,
        patch("samplepath.plots.core.LambdaPanel.plot") as mock_plot_Lam,
        patch("samplepath.plots.core.WPanel.plot") as mock_plot_w,
        patch("samplepath.plots.core.HPanel.plot") as mock_plot_H,
        patch("samplepath.plots.core.CFDPanel.plot") as mock_plot_CFD,
        patch("samplepath.plots.core.LLWPanel.plot") as mock_plot_llw,
    ):
        written = core.plot_core_flow_metrics_charts(
            None, chart_config, filter_result, metrics, out_dir
        )
    assert written == expected
    mock_plot_N.assert_called_once()
    mock_plot_L.assert_called_once()
    mock_plot_Lam.assert_called_once()
    mock_plot_w.assert_called_once()
    mock_plot_H.assert_called_once()
    mock_plot_CFD.assert_called_once()
    mock_plot_llw.assert_called_once()


def test_core_driver_calls_plot_core_stack_with_expected_args():
    metrics = _metrics_fixture()
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
        patch("samplepath.plots.core.NPanel.plot") as mock_plot_N,
        patch("samplepath.plots.core.LPanel.plot") as mock_plot_L,
        patch("samplepath.plots.core.LambdaPanel.plot") as mock_plot_Lam,
        patch("samplepath.plots.core.WPanel.plot") as mock_plot_w,
        patch("samplepath.plots.core.HPanel.plot") as mock_plot_H,
        patch("samplepath.plots.core.CFDPanel.plot") as mock_plot_CFD,
        patch("samplepath.plots.core.LLWPanel.plot"),
    ):
        core.plot_core_flow_metrics_charts(
            None, chart_config, filter_result, metrics, out_dir
        )
    mock_stack.assert_called_once_with(
        None, chart_config, filter_result, metrics, out_dir
    )
    mock_plot_N.assert_called_once()
    mock_plot_L.assert_called_once()
    mock_plot_Lam.assert_called_once()
    mock_plot_w.assert_called_once()
    mock_plot_H.assert_called_once()
    mock_plot_CFD.assert_called_once()


def test_core_driver_passes_event_marks_to_Lambda_and_w():
    metrics = _metrics_fixture()
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
        patch("samplepath.plots.core.NPanel.plot") as mock_plot_N,
        patch("samplepath.plots.core.LPanel.plot") as mock_plot_L,
        patch("samplepath.plots.core.HPanel.plot"),
        patch("samplepath.plots.core.CFDPanel") as mock_cfd_cls,
        patch("samplepath.plots.core.LLWPanel"),
        patch("samplepath.plots.core.LambdaPanel") as mock_lam_cls,
        patch("samplepath.plots.core.WPanel") as mock_w_cls,
    ):
        core.plot_core_flow_metrics_charts(
            None, chart_config, filter_result, metrics, out_dir
        )
    assert mock_lam_cls.call_args.kwargs["with_event_marks"] is True
    assert mock_w_cls.call_args.kwargs["with_event_marks"] is True
    assert mock_cfd_cls.call_args.kwargs["with_event_marks"] is True
    mock_plot_N.assert_called_once()
    mock_plot_L.assert_called_once()
    mock_lam_cls.assert_called_once()
    mock_w_cls.assert_called_once()


def test_core_driver_passes_show_derivations_to_CFD():
    metrics = _metrics_fixture()
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
        patch("samplepath.plots.core.NPanel.plot") as mock_plot_N,
        patch("samplepath.plots.core.LPanel.plot") as mock_plot_L,
        patch("samplepath.plots.core.LambdaPanel.plot") as mock_plot_Lam,
        patch("samplepath.plots.core.WPanel.plot") as mock_plot_w,
        patch("samplepath.plots.core.HPanel.plot") as mock_plot_H,
        patch("samplepath.plots.core.LLWPanel.plot"),
        patch("samplepath.plots.core.CFDPanel") as mock_cfd_cls,
    ):
        core.plot_core_flow_metrics_charts(
            None, chart_config, filter_result, metrics, out_dir
        )
    assert mock_cfd_cls.call_args.kwargs["show_derivations"] is True
    mock_plot_N.assert_called_once()
    mock_plot_L.assert_called_once()
    mock_plot_Lam.assert_called_once()
    mock_plot_w.assert_called_once()
    mock_plot_H.assert_called_once()


def test_core_driver_uses_metrics_freq_for_unit():
    metrics = _metrics_fixture(freq="W")
    out_dir = "/tmp/out"
    args = SimpleNamespace(lambda_pctl=99.0, lambda_lower_pctl=1.0, lambda_warmup=0.5)
    chart_config = ChartConfig.init_from_args(args)
    filter_result = SimpleNamespace(display="Filters: test", label="test")
    with (
        patch("samplepath.plots.core.plot_core_stack"),
        patch("samplepath.plots.core.NPanel.plot") as mock_plot,
    ):
        with (
            patch("samplepath.plots.core.LPanel.plot"),
            patch("samplepath.plots.core.LambdaPanel.plot"),
            patch("samplepath.plots.core.WPanel.plot"),
            patch("samplepath.plots.core.HPanel.plot"),
            patch("samplepath.plots.core.CFDPanel.plot"),
            patch("samplepath.plots.core.LLWPanel.plot"),
        ):
            core.plot_core_flow_metrics_charts(
                None, chart_config, filter_result, metrics, out_dir
            )
    assert mock_plot.call_args.args[1] == chart_config


def test_core_driver_falls_back_to_timestamp_unit():
    metrics = _metrics_fixture(freq=None)
    out_dir = "/tmp/out"
    args = SimpleNamespace(lambda_pctl=99.0, lambda_lower_pctl=1.0, lambda_warmup=0.5)
    chart_config = ChartConfig.init_from_args(args)
    filter_result = SimpleNamespace(display="Filters: test", label="test")
    with (
        patch("samplepath.plots.core.plot_core_stack"),
        patch("samplepath.plots.core.LambdaPanel.plot") as mock_plot,
    ):
        with (
            patch("samplepath.plots.core.NPanel.plot"),
            patch("samplepath.plots.core.LPanel.plot"),
            patch("samplepath.plots.core.WPanel.plot"),
            patch("samplepath.plots.core.HPanel.plot"),
            patch("samplepath.plots.core.CFDPanel.plot"),
            patch("samplepath.plots.core.LLWPanel.plot"),
        ):
            core.plot_core_flow_metrics_charts(
                None, chart_config, filter_result, metrics, out_dir
            )
    assert mock_plot.call_args.kwargs["unit"] == "timestamp"


def test_NPanel_plot_uses_metrics_freq_for_unit():
    fig = MagicMock()
    ax = MagicMock()
    metrics = _metrics_fixture(freq="W")

    @contextmanager
    def fake_context(*args, **kwargs):
        yield fig, ax

    with patch("samplepath.plots.core.figure_context", side_effect=fake_context) as ctx:
        core.NPanel().plot(
            None,
            ChartConfig(),
            SimpleNamespace(display="Filters: test", label="test"),
            metrics,
            "/tmp/out",
        )
    assert ctx.call_args.kwargs["unit"] == "W"


def test_NPanel_plot_falls_back_to_timestamp_unit():
    fig = MagicMock()
    ax = MagicMock()
    metrics = _metrics_fixture(freq=None)

    @contextmanager
    def fake_context(*args, **kwargs):
        yield fig, ax

    with patch("samplepath.plots.core.figure_context", side_effect=fake_context) as ctx:
        core.NPanel().plot(
            None,
            ChartConfig(),
            SimpleNamespace(display="Filters: test", label="test"),
            metrics,
            "/tmp/out",
        )
    assert ctx.call_args.kwargs["unit"] == "timestamp"


def test_core_driver_uses_filter_display_caption():
    metrics = _metrics_fixture()
    out_dir = "/tmp/out"
    args = SimpleNamespace(lambda_pctl=99.0, lambda_lower_pctl=1.0, lambda_warmup=0.5)
    chart_config = ChartConfig.init_from_args(args)
    filter_result = SimpleNamespace(display="Filters: test", label="test")
    with (
        patch("samplepath.plots.core.plot_core_stack"),
        patch("samplepath.plots.core.WPanel.plot") as mock_plot,
    ):
        with (
            patch("samplepath.plots.core.NPanel.plot"),
            patch("samplepath.plots.core.LPanel.plot"),
            patch("samplepath.plots.core.LambdaPanel.plot"),
            patch("samplepath.plots.core.HPanel.plot"),
            patch("samplepath.plots.core.CFDPanel.plot"),
            patch("samplepath.plots.core.LLWPanel.plot"),
        ):
            core.plot_core_flow_metrics_charts(
                None, chart_config, filter_result, metrics, out_dir
            )
    assert mock_plot.call_args.kwargs["caption"] == "Filters: test"


def test_core_driver_calls_plot_H_under_core_dir():
    metrics = _metrics_fixture()
    out_dir = "/tmp/out"
    args = SimpleNamespace(lambda_pctl=99.0, lambda_lower_pctl=1.0, lambda_warmup=0.5)
    chart_config = ChartConfig.init_from_args(args)
    filter_result = SimpleNamespace(display="Filters: test", label="test")
    with (
        patch("samplepath.plots.core.plot_core_stack"),
        patch("samplepath.plots.core.HPanel.plot") as mock_plot,
    ):
        with (
            patch("samplepath.plots.core.NPanel.plot"),
            patch("samplepath.plots.core.LPanel.plot"),
            patch("samplepath.plots.core.LambdaPanel.plot"),
            patch("samplepath.plots.core.WPanel.plot"),
            patch("samplepath.plots.core.CFDPanel.plot"),
            patch("samplepath.plots.core.LLWPanel.plot"),
        ):
            core.plot_core_flow_metrics_charts(
                None, chart_config, filter_result, metrics, out_dir
            )
    assert mock_plot.call_args.args[0] == os.path.join(
        out_dir, "core/cumulative_presence_mass_H.png"
    )


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
    with (
        patch("samplepath.plots.core.plot_core_stack"),
        patch("samplepath.plots.core.NPanel.plot"),
        patch("samplepath.plots.core.LPanel.plot"),
        patch("samplepath.plots.core.LambdaPanel.plot"),
        patch("samplepath.plots.core.WPanel.plot"),
        patch("samplepath.plots.core.HPanel.plot"),
        patch("samplepath.plots.core.LLWPanel.plot"),
        patch("samplepath.plots.core.CFDPanel.plot") as mock_plot,
    ):
        core.plot_core_flow_metrics_charts(
            None, chart_config, filter_result, metrics, out_dir
        )
    assert mock_plot.call_args.args[0] == os.path.join(
        out_dir, "core/cumulative_flow_diagram.png"
    )


def test_core_driver_passes_event_marks_to_CFD():
    metrics = _metrics_fixture()
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
        patch("samplepath.plots.core.NPanel.plot"),
        patch("samplepath.plots.core.LPanel.plot"),
        patch("samplepath.plots.core.LambdaPanel.plot"),
        patch("samplepath.plots.core.WPanel.plot"),
        patch("samplepath.plots.core.HPanel.plot"),
        patch("samplepath.plots.core.LLWPanel"),
        patch("samplepath.plots.core.CFDPanel") as mock_cfd_cls,
    ):
        core.plot_core_flow_metrics_charts(
            None, chart_config, filter_result, metrics, out_dir
        )
    assert mock_cfd_cls.call_args.kwargs["with_event_marks"] is True


def test_LLWPanel_renders_invariant_chart():
    fig = MagicMock()
    ax = MagicMock()
    with (
        patch("samplepath.plots.core.plt.subplots", return_value=(fig, ax)),
        patch("samplepath.plots.core.add_caption") as mock_caption,
    ):
        core.LLWPanel(title="L(T) vs Λ(T).w(T)").plot(
            "out.png",
            [_t("2024-01-01"), _t("2024-01-02")],
            np.array([1.0, 2.0]),
            np.array([1.0, 1.5]),
            np.array([2.0, 1.0]),
            caption="Filters: test",
        )
    ax.plot.assert_called_once()
    ax.set_aspect.assert_called_once_with("equal", adjustable="box")
    ax.grid.assert_not_called()
    ax.set_xlabel.assert_called_once_with("L(T)")
    ax.set_ylabel.assert_called_once_with("Λ(T)·w(T)")
    ax.set_title.assert_called_once_with("L(T) vs Λ(T).w(T)")
    mock_caption.assert_called_once_with(fig, "Filters: test")
    fig.tight_layout.assert_called_once_with(rect=(0.05, 0, 1, 1))
    fig.savefig.assert_called_once_with("out.png")


def test_LLWPanel_skips_reference_line_on_nonfinite():
    fig = MagicMock()
    ax = MagicMock()
    with (patch("samplepath.plots.core.plt.subplots", return_value=(fig, ax)),):
        core.LLWPanel().plot(
            "out.png",
            [_t("2024-01-01")],
            np.array([np.nan]),
            np.array([np.nan]),
            np.array([np.nan]),
        )
    ax.plot.assert_not_called()
    ax.scatter.assert_not_called()


def test_LLWPanel_event_marks_colors_arrivals_purple():
    fig = MagicMock()
    ax = MagicMock()
    times = [_t("2024-01-01"), _t("2024-01-02")]
    with patch("samplepath.plots.core.plt.subplots", return_value=(fig, ax)):
        core.LLWPanel(with_event_marks=True).plot(
            "out.png",
            times,
            np.array([1.0, 2.0]),
            np.array([1.0, 1.0]),
            np.array([1.0, 1.0]),
            arrival_times=[times[0]],
            departure_times=[times[1]],
        )
    colors = ax.scatter.call_args.kwargs["color"]
    assert colors[0][:3] == mcolors.to_rgba("purple")[:3]


def test_LLWPanel_event_marks_colors_departures_green():
    fig = MagicMock()
    ax = MagicMock()
    times = [_t("2024-01-01"), _t("2024-01-02")]
    with patch("samplepath.plots.core.plt.subplots", return_value=(fig, ax)):
        core.LLWPanel(with_event_marks=True).plot(
            "out.png",
            times,
            np.array([1.0, 2.0]),
            np.array([1.0, 1.0]),
            np.array([1.0, 1.0]),
            arrival_times=[times[0]],
            departure_times=[times[1]],
        )
    colors = ax.scatter.call_args.kwargs["color"]
    assert colors[1][:3] == mcolors.to_rgba("green")[:3]


def test_LLWPanel_event_marks_alpha_increases():
    fig = MagicMock()
    ax = MagicMock()
    times = [_t("2024-01-01"), _t("2024-01-02")]
    with patch("samplepath.plots.core.plt.subplots", return_value=(fig, ax)):
        core.LLWPanel(with_event_marks=True).plot(
            "out.png",
            times,
            np.array([1.0, 2.0]),
            np.array([1.0, 1.0]),
            np.array([1.0, 1.0]),
            arrival_times=[times[0]],
            departure_times=[times[1]],
        )
    colors = ax.scatter.call_args.kwargs["color"]
    assert colors[0][3] < colors[1][3]


def test_LLWPanel_event_marks_drop_lines_arrival_color():
    fig = MagicMock()
    ax = MagicMock()
    times = [_t("2024-01-01"), _t("2024-01-02")]
    with patch("samplepath.plots.core.plt.subplots", return_value=(fig, ax)):
        core.LLWPanel(with_event_marks=True).plot(
            "out.png",
            times,
            np.array([1.0, 2.0]),
            np.array([1.0, 1.0]),
            np.array([1.0, 1.0]),
            arrival_times=[times[0]],
            departure_times=[times[1]],
        )
    colors = ax.vlines.call_args.kwargs["colors"]
    assert colors[0] == mcolors.to_rgba("purple", alpha=0.25)


def test_LLWPanel_event_marks_drop_lines_departure_color():
    fig = MagicMock()
    ax = MagicMock()
    times = [_t("2024-01-01"), _t("2024-01-02")]
    with patch("samplepath.plots.core.plt.subplots", return_value=(fig, ax)):
        core.LLWPanel(with_event_marks=True).plot(
            "out.png",
            times,
            np.array([1.0, 2.0]),
            np.array([1.0, 1.0]),
            np.array([1.0, 1.0]),
            arrival_times=[times[0]],
            departure_times=[times[1]],
        )
    colors = ax.vlines.call_args.kwargs["colors"]
    assert colors[1] == mcolors.to_rgba("green", alpha=0.25)


def test_LLWPanel_event_marks_hlines_arrival_color():
    fig = MagicMock()
    ax = MagicMock()
    times = [_t("2024-01-01"), _t("2024-01-02")]
    with patch("samplepath.plots.core.plt.subplots", return_value=(fig, ax)):
        core.LLWPanel(with_event_marks=True).plot(
            "out.png",
            times,
            np.array([1.0, 2.0]),
            np.array([1.0, 1.0]),
            np.array([1.0, 1.0]),
            arrival_times=[times[0]],
            departure_times=[times[1]],
        )
    colors = ax.hlines.call_args.kwargs["colors"]
    assert colors[0] == mcolors.to_rgba("purple", alpha=0.25)


def test_LLWPanel_event_marks_hlines_departure_color():
    fig = MagicMock()
    ax = MagicMock()
    times = [_t("2024-01-01"), _t("2024-01-02")]
    with patch("samplepath.plots.core.plt.subplots", return_value=(fig, ax)):
        core.LLWPanel(with_event_marks=True).plot(
            "out.png",
            times,
            np.array([1.0, 2.0]),
            np.array([1.0, 1.0]),
            np.array([1.0, 1.0]),
            arrival_times=[times[0]],
            departure_times=[times[1]],
        )
    colors = ax.hlines.call_args.kwargs["colors"]
    assert colors[1] == mcolors.to_rgba("green", alpha=0.25)


def test_LLWPanel_event_marks_adds_legend():
    fig = MagicMock()
    ax = MagicMock()
    times = [_t("2024-01-01"), _t("2024-01-02")]
    with patch("samplepath.plots.core.plt.subplots", return_value=(fig, ax)):
        core.LLWPanel(with_event_marks=True).plot(
            "out.png",
            times,
            np.array([1.0, 2.0]),
            np.array([1.0, 1.0]),
            np.array([1.0, 1.0]),
            arrival_times=[times[0]],
            departure_times=[times[1]],
        )
    ax.legend.assert_called_once()


def test_LLWPanel_no_event_marks_alpha_increases():
    fig = MagicMock()
    ax = MagicMock()
    times = [_t("2024-01-01"), _t("2024-01-02")]
    with patch("samplepath.plots.core.plt.subplots", return_value=(fig, ax)):
        core.LLWPanel(with_event_marks=False).plot(
            "out.png",
            times,
            np.array([1.0, 2.0]),
            np.array([1.0, 1.0]),
            np.array([1.0, 1.0]),
        )
    colors = ax.scatter.call_args.kwargs["color"]
    assert colors[0][3] < colors[1][3]


def test_LLWPanel_no_event_marks_has_no_legend():
    fig = MagicMock()
    ax = MagicMock()
    times = [_t("2024-01-01"), _t("2024-01-02")]
    with patch("samplepath.plots.core.plt.subplots", return_value=(fig, ax)):
        core.LLWPanel(with_event_marks=False).plot(
            "out.png",
            times,
            np.array([1.0, 2.0]),
            np.array([1.0, 1.0]),
            np.array([1.0, 1.0]),
        )
    ax.legend.assert_not_called()


def test_LLWPanel_departure_overrides_arrival_color():
    fig = MagicMock()
    ax = MagicMock()
    times = [_t("2024-01-01")]
    with patch("samplepath.plots.core.plt.subplots", return_value=(fig, ax)):
        core.LLWPanel(with_event_marks=True).plot(
            "out.png",
            times,
            np.array([1.0]),
            np.array([1.0]),
            np.array([1.0]),
            arrival_times=times,
            departure_times=times,
        )
    colors = ax.scatter.call_args.kwargs["color"]
    assert colors[0][:3] == mcolors.to_rgba("green")[:3]


def test_core_driver_calls_invariant_plot_under_core_dir():
    metrics = _metrics_fixture()
    out_dir = "/tmp/out"
    args = SimpleNamespace(lambda_pctl=99.0, lambda_lower_pctl=1.0, lambda_warmup=0.5)
    chart_config = ChartConfig.init_from_args(args)
    filter_result = SimpleNamespace(display="Filters: test", label="test")
    with (
        patch("samplepath.plots.core.plot_core_stack"),
        patch("samplepath.plots.core.LLWPanel") as mock_panel,
    ):
        with (
            patch("samplepath.plots.core.NPanel.plot"),
            patch("samplepath.plots.core.LPanel.plot"),
            patch("samplepath.plots.core.LambdaPanel.plot"),
            patch("samplepath.plots.core.WPanel.plot"),
            patch("samplepath.plots.core.HPanel.plot"),
            patch("samplepath.plots.core.CFDPanel.plot"),
        ):
            core.plot_core_flow_metrics_charts(
                None, chart_config, filter_result, metrics, out_dir
            )
    mock_panel.assert_called_once_with(
        with_event_marks=False,
        title="L(T) vs Λ(T).w(T)",
    )
    mock_panel.return_value.plot.assert_called_once_with(
        os.path.join(out_dir, "core/littles_law_invariant.png"),
        metrics.times,
        metrics.L,
        metrics.Lambda,
        metrics.w,
        arrival_times=metrics.arrival_times,
        departure_times=metrics.departure_times,
        caption="Filters: test",
    )


def test_core_driver_omits_caption_when_label_empty():
    metrics = _metrics_fixture()
    out_dir = "/tmp/out"
    args = SimpleNamespace(lambda_pctl=99.0, lambda_lower_pctl=1.0, lambda_warmup=0.5)
    chart_config = ChartConfig.init_from_args(args)
    filter_result = SimpleNamespace(display="Filters: ", label="")
    with (
        patch("samplepath.plots.core.plot_core_stack") as mock_stack,
        patch("samplepath.plots.core.NPanel.plot"),
        patch("samplepath.plots.core.LPanel.plot"),
        patch("samplepath.plots.core.LambdaPanel.plot"),
        patch("samplepath.plots.core.WPanel.plot"),
        patch("samplepath.plots.core.HPanel.plot"),
        patch("samplepath.plots.core.CFDPanel.plot"),
        patch("samplepath.plots.core.LLWPanel.plot"),
    ):
        core.plot_core_flow_metrics_charts(
            None, chart_config, filter_result, metrics, out_dir
        )
    mock_stack.assert_called_once_with(
        None, chart_config, filter_result, metrics, out_dir
    )
