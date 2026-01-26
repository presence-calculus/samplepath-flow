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
        core.render_N(
            ax,
            times,
            values,
            arrival_times=[times[0]],
            departure_times=[times[0]],
            with_event_marks=True,
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
        core.render_N(ax, times, values, with_event_marks=False, show_title=False)
    ax.set_title.assert_not_called()


def test_render_L_colors_grey_when_overlays():
    ax = MagicMock()
    times = [_t("2024-01-01")]
    values = np.array([2.0])
    overlays = ["overlay"]
    with (
        patch("samplepath.plots.core.build_event_overlays", return_value=overlays),
        patch("samplepath.plots.core.render_line_chart") as mock_render,
    ):
        core.render_L(
            ax,
            times,
            values,
            arrival_times=[times[0]],
            departure_times=[times[0]],
            with_event_marks=True,
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
        core.render_Lambda(ax, times, values, clip_opts=clip)
    mock_line.assert_called_once()
    mock_clip.assert_called_once_with(ax, list(times), values, 99.0, 1.0, 0.5)
    ax.set_ylabel.assert_called_once_with("Λ(T) [1/hr]")


def test_render_w_sets_defaults():
    ax = MagicMock()
    times = [_t("2024-01-01")]
    values = np.array([0.5])
    with patch("samplepath.plots.core.render_line_chart") as mock_line:
        core.render_w(ax, times, values)
    mock_line.assert_called_once()
    ax.set_ylabel.assert_called_once_with("w(T) [hrs]")


def test_render_Lambda_arrival_overlays_when_enabled():
    ax = MagicMock()
    times = [_t("2024-01-01")]
    values = np.array([1.0])
    arrivals = [times[0]]
    with patch("samplepath.plots.core.render_line_chart") as mock_line:
        core.render_Lambda(
            ax,
            times,
            values,
            arrival_times=arrivals,
            with_event_marks=True,
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
        core.render_w(
            ax,
            times,
            values,
            arrival_times=arrivals,
            departure_times=departures,
            with_event_marks=True,
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
        core.render_w(
            ax,
            times,
            values,
            arrival_times=arrivals,
            departure_times=departures,
            with_event_marks=True,
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
        core.render_w(
            ax,
            times,
            values,
            arrival_times=arrivals,
            departure_times=departures,
            with_event_marks=True,
        )
    overlays = mock_line.call_args.kwargs["overlays"]
    assert overlays[1].drop_lines is False


def test_render_H_sets_defaults():
    ax = MagicMock()
    times = [_t("2024-01-01")]
    values = np.array([1.0])
    with patch("samplepath.plots.core.render_line_chart") as mock_line:
        core.render_H(ax, times, values)
    mock_line.assert_called_once()
    ax.set_ylabel.assert_called_once_with("H(T) [hrs·items]")


def _render_cfd_with_mocks(with_event_marks: bool = False):
    ax = MagicMock()
    times = [_t("2024-01-01")]
    arrivals = np.array([1.0])
    departures = np.array([0.0])
    with patch("samplepath.plots.core.render_step_chart") as mock_step:
        core.render_CFD(
            ax,
            times,
            arrivals,
            departures,
            arrival_times=times,
            departure_times=times,
            with_event_marks=with_event_marks,
        )
    return ax, mock_step


def test_render_CFD_calls_step_twice():
    _, mock_step = _render_cfd_with_mocks()
    assert mock_step.call_count == 2


def test_render_CFD_arrivals_label():
    _, mock_step = _render_cfd_with_mocks()
    assert mock_step.call_args_list[0].kwargs["label"] == "A(t): cumulative arrivals"


def test_render_CFD_arrivals_color():
    _, mock_step = _render_cfd_with_mocks()
    assert mock_step.call_args_list[0].kwargs["color"] == "purple"


def test_render_CFD_arrivals_fill_false():
    _, mock_step = _render_cfd_with_mocks()
    assert mock_step.call_args_list[0].kwargs["fill"] is False


def test_render_CFD_departures_label():
    _, mock_step = _render_cfd_with_mocks()
    assert mock_step.call_args_list[1].kwargs["label"] == "D(t): cumulative departures"


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
        core.render_CFD(
            ax,
            times,
            arrivals,
            departures,
            arrival_times=arrival_times,
            departure_times=departure_times,
            with_event_marks=True,
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
        core.render_CFD(
            ax,
            times,
            arrivals,
            departures,
            arrival_times=arrival_times,
            departure_times=departure_times,
            with_event_marks=True,
        )
    overlay = mock_step.call_args_list[1].kwargs["overlays"][0]
    assert overlay.x == departure_times


def test_render_CFD_sets_title():
    ax, _ = _render_cfd_with_mocks()
    ax.set_title.assert_called_once_with("Cumulative Arrivals vs Cumulative Departures")


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
        core.render_CFD(ax, times, arrivals, departures)
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
        patch("samplepath.plots.core.render_N") as mock_render,
    ):
        core.plot_N(
            "out.png", [_t("2024-01-01")], np.array([1.0]), with_event_marks=True
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
        patch("samplepath.plots.core.render_L") as mock_render,
    ):
        core.plot_L("out.png", [_t("2024-01-01")], np.array([2.0]))
    mock_render.assert_called_once()


def test_plot_single_panel_Lambda_calls_renderer():
    fig = MagicMock()
    ax = MagicMock()

    @contextmanager
    def fake_context(*args, **kwargs):
        yield fig, ax

    with (
        patch("samplepath.plots.core.figure_context", side_effect=fake_context),
        patch("samplepath.plots.core.render_Lambda") as mock_render,
    ):
        core.plot_Lambda("out.png", [_t("2024-01-01")], np.array([3.0]))
    mock_render.assert_called_once()


def test_plot_single_panel_w_calls_renderer():
    fig = MagicMock()
    ax = MagicMock()

    @contextmanager
    def fake_context(*args, **kwargs):
        yield fig, ax

    with (
        patch("samplepath.plots.core.figure_context", side_effect=fake_context),
        patch("samplepath.plots.core.render_w") as mock_render,
    ):
        core.plot_w("out.png", [_t("2024-01-01")], np.array([4.0]))
    mock_render.assert_called_once()


def test_plot_single_panel_H_calls_renderer():
    fig = MagicMock()
    ax = MagicMock()

    @contextmanager
    def fake_context(*args, **kwargs):
        yield fig, ax

    with (
        patch("samplepath.plots.core.figure_context", side_effect=fake_context),
        patch("samplepath.plots.core.render_H") as mock_render,
    ):
        core.plot_H("out.png", [_t("2024-01-01")], np.array([5.0]))
    mock_render.assert_called_once()


def test_plot_single_panel_CFD_calls_renderer():
    fig = MagicMock()
    ax = MagicMock()

    @contextmanager
    def fake_context(*args, **kwargs):
        yield fig, ax

    with (
        patch("samplepath.plots.core.figure_context", side_effect=fake_context),
        patch("samplepath.plots.core.render_CFD") as mock_render,
    ):
        core.plot_CFD(
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
        core.plot_CFD(
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
        core.plot_CFD(
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
        core.plot_CFD(
            "out.png",
            [_t("2024-01-01")],
            np.array([1.0]),
            np.array([0.0]),
        )
    assert ctx.call_args.kwargs["nrows"] == 1


def test_plot_core_stack_calls_all_renderers():
    fig = MagicMock()
    axes = np.array([object() for _ in range(4)], dtype=object)

    @contextmanager
    def fake_context(*args, **kwargs):
        yield fig, axes

    with (
        patch("samplepath.plots.core.layout_context", side_effect=fake_context),
        patch("samplepath.plots.core.render_N") as mock_N,
        patch("samplepath.plots.core.render_L") as mock_L,
        patch("samplepath.plots.core.render_Lambda") as mock_Lam,
        patch("samplepath.plots.core.render_w") as mock_w,
    ):
        core.plot_core_stack(
            "out.png",
            [_t("2024-01-01")],
            np.array([1.0]),
            np.array([2.0]),
            np.array([3.0]),
            np.array([4.0]),
            lambda_pctl_upper=99.0,
            lambda_pctl_lower=1.0,
            lambda_warmup_hours=0.5,
            with_event_marks=True,
        )

    mock_N.assert_called_once()
    mock_L.assert_called_once()
    mock_Lam.assert_called_once()
    mock_w.assert_called_once()


def test_plot_core_stack_applies_layout_and_caption():
    fig = MagicMock()
    axes = np.array([object() for _ in range(4)], dtype=object)

    @contextmanager
    def fake_context(out_path, *, layout, decor, unit, format_axis_fn, format_targets):
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
        patch("samplepath.plots.core.render_N"),
        patch("samplepath.plots.core.render_L"),
        patch("samplepath.plots.core.render_Lambda"),
        patch("samplepath.plots.core.render_w"),
    ):
        core.plot_core_stack(
            "out.png",
            [_t("2024-01-01")],
            np.array([1.0]),
            np.array([2.0]),
            np.array([3.0]),
            np.array([4.0]),
            caption="Filters: test",
        )


def test_plot_core_stack_uses_tighter_layout_without_caption():
    fig = MagicMock()
    axes = np.array([object() for _ in range(4)], dtype=object)

    @contextmanager
    def fake_context(out_path, *, layout, decor, unit, format_axis_fn, format_targets):
        assert decor.caption is None
        assert decor.tight_layout_rect == (0, 0, 1, 0.96)
        yield fig, axes

    with (
        patch("samplepath.plots.core.layout_context", side_effect=fake_context),
        patch("samplepath.plots.core.render_N"),
        patch("samplepath.plots.core.render_L"),
        patch("samplepath.plots.core.render_Lambda"),
        patch("samplepath.plots.core.render_w"),
    ):
        core.plot_core_stack(
            "out.png",
            [_t("2024-01-01")],
            np.array([1.0]),
            np.array([2.0]),
            np.array([3.0]),
            np.array([4.0]),
            caption=None,
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
        patch("samplepath.plots.core.plot_N"),
        patch("samplepath.plots.core.plot_L"),
        patch("samplepath.plots.core.plot_Lambda"),
        patch("samplepath.plots.core.plot_w"),
        patch("samplepath.plots.core.plot_H"),
        patch("samplepath.plots.core.plot_CFD"),
        patch("samplepath.plots.core.plot_L_vs_Lambda_w"),
    ):
        written = core.plot_core_flow_metrics_charts(
            None, args, filter_result, metrics, out_dir
        )
    assert written == expected


def test_core_driver_calls_plot_core_stack_with_expected_args():
    metrics = _metrics_fixture()
    out_dir = "/tmp/out"
    args = SimpleNamespace(
        lambda_pctl=98.0,
        lambda_lower_pctl=2.0,
        lambda_warmup=1.5,
        with_event_marks=True,
    )
    filter_result = SimpleNamespace(display="Filters: test", label="test")
    with (
        patch("samplepath.plots.core.plot_core_stack") as mock_stack,
        patch("samplepath.plots.core.plot_N"),
        patch("samplepath.plots.core.plot_L"),
        patch("samplepath.plots.core.plot_Lambda"),
        patch("samplepath.plots.core.plot_w"),
        patch("samplepath.plots.core.plot_H"),
        patch("samplepath.plots.core.plot_CFD"),
        patch("samplepath.plots.core.plot_L_vs_Lambda_w"),
    ):
        core.plot_core_flow_metrics_charts(None, args, filter_result, metrics, out_dir)
    mock_stack.assert_called_once_with(
        os.path.join(out_dir, "sample_path_flow_metrics.png"),
        metrics.times,
        metrics.N,
        metrics.L,
        metrics.Lambda,
        metrics.w,
        lambda_pctl_upper=98.0,
        lambda_pctl_lower=2.0,
        lambda_warmup_hours=1.5,
        caption="Filters: test",
        arrival_times=metrics.arrival_times,
        departure_times=metrics.departure_times,
        with_event_marks=True,
        unit="D",
    )


def test_core_driver_passes_event_marks_to_Lambda_and_w():
    metrics = _metrics_fixture()
    out_dir = "/tmp/out"
    args = SimpleNamespace(
        lambda_pctl=99.0,
        lambda_lower_pctl=1.0,
        lambda_warmup=0.5,
        with_event_marks=True,
    )
    filter_result = SimpleNamespace(display="Filters: test", label="test")
    with (
        patch("samplepath.plots.core.plot_core_stack"),
        patch("samplepath.plots.core.plot_N"),
        patch("samplepath.plots.core.plot_L"),
        patch("samplepath.plots.core.plot_H"),
        patch("samplepath.plots.core.plot_CFD"),
        patch("samplepath.plots.core.plot_L_vs_Lambda_w"),
        patch("samplepath.plots.core.plot_Lambda") as mock_lam,
        patch("samplepath.plots.core.plot_w") as mock_w,
    ):
        core.plot_core_flow_metrics_charts(None, args, filter_result, metrics, out_dir)
    assert mock_lam.call_args.kwargs["with_event_marks"] is True
    assert mock_w.call_args.kwargs["with_event_marks"] is True


def test_core_driver_uses_metrics_freq_for_unit():
    metrics = _metrics_fixture(freq="W")
    out_dir = "/tmp/out"
    args = SimpleNamespace(lambda_pctl=99.0, lambda_lower_pctl=1.0, lambda_warmup=0.5)
    filter_result = SimpleNamespace(display="Filters: test", label="test")
    with (
        patch("samplepath.plots.core.plot_core_stack"),
        patch("samplepath.plots.core.plot_N") as mock_plot,
    ):
        with (
            patch("samplepath.plots.core.plot_L"),
            patch("samplepath.plots.core.plot_Lambda"),
            patch("samplepath.plots.core.plot_w"),
            patch("samplepath.plots.core.plot_H"),
            patch("samplepath.plots.core.plot_CFD"),
            patch("samplepath.plots.core.plot_L_vs_Lambda_w"),
        ):
            core.plot_core_flow_metrics_charts(
                None, args, filter_result, metrics, out_dir
            )
    assert mock_plot.call_args.kwargs["unit"] == "W"


def test_core_driver_falls_back_to_timestamp_unit():
    metrics = _metrics_fixture(freq=None)
    out_dir = "/tmp/out"
    args = SimpleNamespace(lambda_pctl=99.0, lambda_lower_pctl=1.0, lambda_warmup=0.5)
    filter_result = SimpleNamespace(display="Filters: test", label="test")
    with (
        patch("samplepath.plots.core.plot_core_stack"),
        patch("samplepath.plots.core.plot_Lambda") as mock_plot,
    ):
        with (
            patch("samplepath.plots.core.plot_N"),
            patch("samplepath.plots.core.plot_L"),
            patch("samplepath.plots.core.plot_w"),
            patch("samplepath.plots.core.plot_H"),
            patch("samplepath.plots.core.plot_CFD"),
            patch("samplepath.plots.core.plot_L_vs_Lambda_w"),
        ):
            core.plot_core_flow_metrics_charts(
                None, args, filter_result, metrics, out_dir
            )
    assert mock_plot.call_args.kwargs["unit"] == "timestamp"


def test_core_driver_uses_filter_display_caption():
    metrics = _metrics_fixture()
    out_dir = "/tmp/out"
    args = SimpleNamespace(lambda_pctl=99.0, lambda_lower_pctl=1.0, lambda_warmup=0.5)
    filter_result = SimpleNamespace(display="Filters: test", label="test")
    with (
        patch("samplepath.plots.core.plot_core_stack"),
        patch("samplepath.plots.core.plot_w") as mock_plot,
    ):
        with (
            patch("samplepath.plots.core.plot_N"),
            patch("samplepath.plots.core.plot_L"),
            patch("samplepath.plots.core.plot_Lambda"),
            patch("samplepath.plots.core.plot_H"),
            patch("samplepath.plots.core.plot_CFD"),
            patch("samplepath.plots.core.plot_L_vs_Lambda_w"),
        ):
            core.plot_core_flow_metrics_charts(
                None, args, filter_result, metrics, out_dir
            )
    assert mock_plot.call_args.kwargs["caption"] == "Filters: test"


def test_core_driver_calls_plot_H_under_core_dir():
    metrics = _metrics_fixture()
    out_dir = "/tmp/out"
    args = SimpleNamespace(lambda_pctl=99.0, lambda_lower_pctl=1.0, lambda_warmup=0.5)
    filter_result = SimpleNamespace(display="Filters: test", label="test")
    with (
        patch("samplepath.plots.core.plot_core_stack"),
        patch("samplepath.plots.core.plot_H") as mock_plot,
    ):
        with (
            patch("samplepath.plots.core.plot_N"),
            patch("samplepath.plots.core.plot_L"),
            patch("samplepath.plots.core.plot_Lambda"),
            patch("samplepath.plots.core.plot_w"),
            patch("samplepath.plots.core.plot_CFD"),
            patch("samplepath.plots.core.plot_L_vs_Lambda_w"),
        ):
            core.plot_core_flow_metrics_charts(
                None, args, filter_result, metrics, out_dir
            )
    assert mock_plot.call_args.args[0] == os.path.join(
        out_dir, "core/cumulative_presence_mass_H.png"
    )


def test_core_driver_calls_plot_CFD_under_core_dir():
    out_dir = "/tmp/out"
    args = SimpleNamespace(lambda_pctl=None, lambda_lower_pctl=None, lambda_warmup=0.0)
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
        patch("samplepath.plots.core.plot_N"),
        patch("samplepath.plots.core.plot_L"),
        patch("samplepath.plots.core.plot_Lambda"),
        patch("samplepath.plots.core.plot_w"),
        patch("samplepath.plots.core.plot_H"),
        patch("samplepath.plots.core.plot_L_vs_Lambda_w"),
        patch("samplepath.plots.core.plot_CFD") as mock_plot,
    ):
        core.plot_core_flow_metrics_charts(None, args, filter_result, metrics, out_dir)
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
    filter_result = SimpleNamespace(display="Filters: test", label="test")
    with (
        patch("samplepath.plots.core.plot_core_stack"),
        patch("samplepath.plots.core.plot_N"),
        patch("samplepath.plots.core.plot_L"),
        patch("samplepath.plots.core.plot_Lambda"),
        patch("samplepath.plots.core.plot_w"),
        patch("samplepath.plots.core.plot_H"),
        patch("samplepath.plots.core.plot_L_vs_Lambda_w"),
        patch("samplepath.plots.core.plot_CFD") as mock_plot,
    ):
        core.plot_core_flow_metrics_charts(None, args, filter_result, metrics, out_dir)
    assert mock_plot.call_args.kwargs["with_event_marks"] is True


def test_plot_L_vs_Lambda_w_renders_invariant_chart():
    fig = MagicMock()
    ax = MagicMock()
    with (
        patch("samplepath.plots.core.plt.subplots", return_value=(fig, ax)),
        patch("samplepath.plots.core.add_caption") as mock_caption,
    ):
        core.plot_L_vs_Lambda_w(
            [_t("2024-01-01"), _t("2024-01-02")],
            np.array([1.0, 2.0]),
            np.array([1.0, 1.5]),
            np.array([2.0, 1.0]),
            title="L(T) vs Λ(T).w(T)",
            out_path="out.png",
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


def test_plot_L_vs_Lambda_w_skips_reference_line_on_nonfinite():
    fig = MagicMock()
    ax = MagicMock()
    with (patch("samplepath.plots.core.plt.subplots", return_value=(fig, ax)),):
        core.plot_L_vs_Lambda_w(
            [_t("2024-01-01")],
            np.array([np.nan]),
            np.array([np.nan]),
            np.array([np.nan]),
            title="L(T) vs Λ(T).w(T)",
            out_path="out.png",
        )
    ax.plot.assert_not_called()
    ax.scatter.assert_not_called()


def test_plot_L_vs_Lambda_w_event_marks_colors_arrivals_purple():
    fig = MagicMock()
    ax = MagicMock()
    times = [_t("2024-01-01"), _t("2024-01-02")]
    with patch("samplepath.plots.core.plt.subplots", return_value=(fig, ax)):
        core.plot_L_vs_Lambda_w(
            times,
            np.array([1.0, 2.0]),
            np.array([1.0, 1.0]),
            np.array([1.0, 1.0]),
            arrival_times=[times[0]],
            departure_times=[times[1]],
            with_event_marks=True,
            title="L(T) vs Λ(T).w(T)",
            out_path="out.png",
        )
    colors = ax.scatter.call_args.kwargs["color"]
    assert colors[0][:3] == mcolors.to_rgba("purple")[:3]


def test_plot_L_vs_Lambda_w_event_marks_colors_departures_green():
    fig = MagicMock()
    ax = MagicMock()
    times = [_t("2024-01-01"), _t("2024-01-02")]
    with patch("samplepath.plots.core.plt.subplots", return_value=(fig, ax)):
        core.plot_L_vs_Lambda_w(
            times,
            np.array([1.0, 2.0]),
            np.array([1.0, 1.0]),
            np.array([1.0, 1.0]),
            arrival_times=[times[0]],
            departure_times=[times[1]],
            with_event_marks=True,
            title="L(T) vs Λ(T).w(T)",
            out_path="out.png",
        )
    colors = ax.scatter.call_args.kwargs["color"]
    assert colors[1][:3] == mcolors.to_rgba("green")[:3]


def test_plot_L_vs_Lambda_w_event_marks_alpha_increases():
    fig = MagicMock()
    ax = MagicMock()
    times = [_t("2024-01-01"), _t("2024-01-02")]
    with patch("samplepath.plots.core.plt.subplots", return_value=(fig, ax)):
        core.plot_L_vs_Lambda_w(
            times,
            np.array([1.0, 2.0]),
            np.array([1.0, 1.0]),
            np.array([1.0, 1.0]),
            arrival_times=[times[0]],
            departure_times=[times[1]],
            with_event_marks=True,
            title="L(T) vs Λ(T).w(T)",
            out_path="out.png",
        )
    colors = ax.scatter.call_args.kwargs["color"]
    assert colors[0][3] < colors[1][3]


def test_plot_L_vs_Lambda_w_event_marks_drop_lines_arrival_color():
    fig = MagicMock()
    ax = MagicMock()
    times = [_t("2024-01-01"), _t("2024-01-02")]
    with patch("samplepath.plots.core.plt.subplots", return_value=(fig, ax)):
        core.plot_L_vs_Lambda_w(
            times,
            np.array([1.0, 2.0]),
            np.array([1.0, 1.0]),
            np.array([1.0, 1.0]),
            arrival_times=[times[0]],
            departure_times=[times[1]],
            with_event_marks=True,
            title="L(T) vs Λ(T).w(T)",
            out_path="out.png",
        )
    colors = ax.vlines.call_args.kwargs["colors"]
    assert colors[0] == mcolors.to_rgba("purple", alpha=0.25)


def test_plot_L_vs_Lambda_w_event_marks_drop_lines_departure_color():
    fig = MagicMock()
    ax = MagicMock()
    times = [_t("2024-01-01"), _t("2024-01-02")]
    with patch("samplepath.plots.core.plt.subplots", return_value=(fig, ax)):
        core.plot_L_vs_Lambda_w(
            times,
            np.array([1.0, 2.0]),
            np.array([1.0, 1.0]),
            np.array([1.0, 1.0]),
            arrival_times=[times[0]],
            departure_times=[times[1]],
            with_event_marks=True,
            title="L(T) vs Λ(T).w(T)",
            out_path="out.png",
        )
    colors = ax.vlines.call_args.kwargs["colors"]
    assert colors[1] == mcolors.to_rgba("green", alpha=0.25)


def test_plot_L_vs_Lambda_w_event_marks_hlines_arrival_color():
    fig = MagicMock()
    ax = MagicMock()
    times = [_t("2024-01-01"), _t("2024-01-02")]
    with patch("samplepath.plots.core.plt.subplots", return_value=(fig, ax)):
        core.plot_L_vs_Lambda_w(
            times,
            np.array([1.0, 2.0]),
            np.array([1.0, 1.0]),
            np.array([1.0, 1.0]),
            arrival_times=[times[0]],
            departure_times=[times[1]],
            with_event_marks=True,
            title="L(T) vs Λ(T).w(T)",
            out_path="out.png",
        )
    colors = ax.hlines.call_args.kwargs["colors"]
    assert colors[0] == mcolors.to_rgba("purple", alpha=0.25)


def test_plot_L_vs_Lambda_w_event_marks_hlines_departure_color():
    fig = MagicMock()
    ax = MagicMock()
    times = [_t("2024-01-01"), _t("2024-01-02")]
    with patch("samplepath.plots.core.plt.subplots", return_value=(fig, ax)):
        core.plot_L_vs_Lambda_w(
            times,
            np.array([1.0, 2.0]),
            np.array([1.0, 1.0]),
            np.array([1.0, 1.0]),
            arrival_times=[times[0]],
            departure_times=[times[1]],
            with_event_marks=True,
            title="L(T) vs Λ(T).w(T)",
            out_path="out.png",
        )
    colors = ax.hlines.call_args.kwargs["colors"]
    assert colors[1] == mcolors.to_rgba("green", alpha=0.25)


def test_plot_L_vs_Lambda_w_event_marks_adds_legend():
    fig = MagicMock()
    ax = MagicMock()
    times = [_t("2024-01-01"), _t("2024-01-02")]
    with patch("samplepath.plots.core.plt.subplots", return_value=(fig, ax)):
        core.plot_L_vs_Lambda_w(
            times,
            np.array([1.0, 2.0]),
            np.array([1.0, 1.0]),
            np.array([1.0, 1.0]),
            arrival_times=[times[0]],
            departure_times=[times[1]],
            with_event_marks=True,
            title="L(T) vs Λ(T).w(T)",
            out_path="out.png",
        )
    ax.legend.assert_called_once()


def test_plot_L_vs_Lambda_w_no_event_marks_alpha_increases():
    fig = MagicMock()
    ax = MagicMock()
    times = [_t("2024-01-01"), _t("2024-01-02")]
    with patch("samplepath.plots.core.plt.subplots", return_value=(fig, ax)):
        core.plot_L_vs_Lambda_w(
            times,
            np.array([1.0, 2.0]),
            np.array([1.0, 1.0]),
            np.array([1.0, 1.0]),
            with_event_marks=False,
            title="L(T) vs Λ(T).w(T)",
            out_path="out.png",
        )
    colors = ax.scatter.call_args.kwargs["color"]
    assert colors[0][3] < colors[1][3]


def test_plot_L_vs_Lambda_w_no_event_marks_has_no_legend():
    fig = MagicMock()
    ax = MagicMock()
    times = [_t("2024-01-01"), _t("2024-01-02")]
    with patch("samplepath.plots.core.plt.subplots", return_value=(fig, ax)):
        core.plot_L_vs_Lambda_w(
            times,
            np.array([1.0, 2.0]),
            np.array([1.0, 1.0]),
            np.array([1.0, 1.0]),
            with_event_marks=False,
            title="L(T) vs Λ(T).w(T)",
            out_path="out.png",
        )
    ax.legend.assert_not_called()


def test_plot_L_vs_Lambda_w_departure_overrides_arrival_color():
    fig = MagicMock()
    ax = MagicMock()
    times = [_t("2024-01-01")]
    with patch("samplepath.plots.core.plt.subplots", return_value=(fig, ax)):
        core.plot_L_vs_Lambda_w(
            times,
            np.array([1.0]),
            np.array([1.0]),
            np.array([1.0]),
            arrival_times=times,
            departure_times=times,
            with_event_marks=True,
            title="L(T) vs Λ(T).w(T)",
            out_path="out.png",
        )
    colors = ax.scatter.call_args.kwargs["color"]
    assert colors[0][:3] == mcolors.to_rgba("green")[:3]


def test_core_driver_calls_invariant_plot_under_core_dir():
    metrics = _metrics_fixture()
    out_dir = "/tmp/out"
    args = SimpleNamespace(lambda_pctl=99.0, lambda_lower_pctl=1.0, lambda_warmup=0.5)
    filter_result = SimpleNamespace(display="Filters: test", label="test")
    with (
        patch("samplepath.plots.core.plot_core_stack"),
        patch("samplepath.plots.core.plot_L_vs_Lambda_w") as mock_plot,
    ):
        with (
            patch("samplepath.plots.core.plot_N"),
            patch("samplepath.plots.core.plot_L"),
            patch("samplepath.plots.core.plot_Lambda"),
            patch("samplepath.plots.core.plot_w"),
            patch("samplepath.plots.core.plot_H"),
            patch("samplepath.plots.core.plot_CFD"),
        ):
            core.plot_core_flow_metrics_charts(
                None, args, filter_result, metrics, out_dir
            )
    mock_plot.assert_called_once_with(
        metrics.times,
        metrics.L,
        metrics.Lambda,
        metrics.w,
        arrival_times=metrics.arrival_times,
        departure_times=metrics.departure_times,
        with_event_marks=False,
        title="L(T) vs Λ(T).w(T)",
        out_path=os.path.join(out_dir, "core/littles_law_invariant.png"),
        caption="Filters: test",
    )


def test_core_driver_omits_caption_when_label_empty():
    metrics = _metrics_fixture()
    out_dir = "/tmp/out"
    args = SimpleNamespace(lambda_pctl=99.0, lambda_lower_pctl=1.0, lambda_warmup=0.5)
    filter_result = SimpleNamespace(display="Filters: ", label="")
    with (
        patch("samplepath.plots.core.plot_core_stack") as mock_stack,
        patch("samplepath.plots.core.plot_N"),
        patch("samplepath.plots.core.plot_L"),
        patch("samplepath.plots.core.plot_Lambda"),
        patch("samplepath.plots.core.plot_w"),
        patch("samplepath.plots.core.plot_H"),
        patch("samplepath.plots.core.plot_CFD"),
        patch("samplepath.plots.core.plot_L_vs_Lambda_w"),
    ):
        core.plot_core_flow_metrics_charts(None, args, filter_result, metrics, out_dir)
    assert mock_stack.call_args.kwargs["caption"] == "Filters: "
