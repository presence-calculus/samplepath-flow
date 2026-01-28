# -*- coding: utf-8 -*-
# Copyright (c) 2025 Krishna Kumar
# SPDX-License-Identifier: MIT
"""Tests for plotting functionality."""
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from samplepath.plots.helpers import (
    ScatterOverlay,
    build_event_overlays,
    draw_step_chart,
    render_lambda_chart,
    render_line_chart,
    render_LT_chart,
    render_N_chart,
    render_step_chart,
)


def _t(s: str) -> pd.Timestamp:
    return pd.Timestamp(s)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# ScatterOverlay tests
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def test_scatter_overlay_creation():
    """ScatterOverlay dataclass instantiates correctly."""
    overlay = ScatterOverlay(
        x=[_t("2024-01-01 00:00")],
        y=[1.0],
        color="purple",
        label="Test",
    )
    assert overlay.x == [_t("2024-01-01 00:00")]
    assert overlay.y == [1.0]
    assert overlay.color == "purple"
    assert overlay.label == "Test"


def test_scatter_overlay_drop_lines_default_false():
    """drop_lines defaults to False."""
    overlay = ScatterOverlay(
        x=[_t("2024-01-01 00:00")],
        y=[1.0],
        color="green",
        label="Test",
    )
    assert overlay.drop_lines is False


def test_scatter_overlay_drop_lines_can_be_true():
    """drop_lines can be set to True."""
    overlay = ScatterOverlay(
        x=[_t("2024-01-01 00:00")],
        y=[1.0],
        color="green",
        label="Test",
        drop_lines=True,
    )
    assert overlay.drop_lines is True


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# draw_step_chart integration tests
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


@pytest.fixture
def simple_chart_data():
    """Simple time series for chart testing."""
    times = [_t("2024-01-01 00:00"), _t("2024-01-01 02:00")]
    values = np.array([1.0, 0.0])
    return times, values


@patch("samplepath.plots.helpers.plt")
def test_draw_step_chart_no_overlays_no_scatter(mock_plt, simple_chart_data, tmp_path):
    """When overlays=None, no scatter calls are made."""
    times, values = simple_chart_data
    out_path = str(tmp_path / "chart.png")

    mock_ax = MagicMock()
    mock_fig = MagicMock()
    mock_plt.subplots.return_value = (mock_fig, mock_ax)

    draw_step_chart(times, values, "Title", "Y", out_path, overlays=None)

    mock_ax.scatter.assert_not_called()


@patch("samplepath.plots.helpers.plt")
def test_draw_step_chart_with_overlays_calls_scatter(
    mock_plt, simple_chart_data, tmp_path
):
    """When overlays provided, scatter is called for each overlay."""
    times, values = simple_chart_data
    out_path = str(tmp_path / "chart.png")
    overlays = [
        ScatterOverlay(
            x=[_t("2024-01-01 00:00")],
            y=[1.0],
            color="purple",
            label="Arrival",
        ),
        ScatterOverlay(
            x=[_t("2024-01-01 02:00")],
            y=[0.0],
            color="green",
            label="Departure",
        ),
    ]

    mock_ax = MagicMock()
    mock_fig = MagicMock()
    mock_plt.subplots.return_value = (mock_fig, mock_ax)

    draw_step_chart(times, values, "Title", "Y", out_path, overlays=overlays)

    assert mock_ax.scatter.call_count == 2


@patch("samplepath.plots.helpers.plt")
def test_draw_step_chart_overlay_uses_specified_color(mock_plt, tmp_path):
    """Overlay color is passed to scatter."""
    times = [_t("2024-01-01 00:00")]
    values = np.array([1.0])
    out_path = str(tmp_path / "chart.png")
    overlays = [
        ScatterOverlay(
            x=[_t("2024-01-01 00:00")],
            y=[1.0],
            color="purple",
            label="Test",
        ),
    ]

    mock_ax = MagicMock()
    mock_fig = MagicMock()
    mock_plt.subplots.return_value = (mock_fig, mock_ax)

    draw_step_chart(times, values, "Title", "Y", out_path, overlays=overlays)

    call_kwargs = mock_ax.scatter.call_args_list[0][1]
    assert call_kwargs["color"] == "purple"


@patch("samplepath.plots.helpers.plt")
def test_draw_step_chart_overlay_drop_lines_calls_vlines(mock_plt, tmp_path):
    """drop_lines=True renders vertical lines."""
    times = [_t("2024-01-01 00:00")]
    values = np.array([1.0])
    out_path = str(tmp_path / "chart.png")
    overlays = [
        ScatterOverlay(
            x=[_t("2024-01-01 00:00")],
            y=[1.0],
            color="green",
            label="Test",
            drop_lines=True,
        ),
    ]

    mock_ax = MagicMock()
    mock_fig = MagicMock()
    mock_plt.subplots.return_value = (mock_fig, mock_ax)

    draw_step_chart(times, values, "Title", "Y", out_path, overlays=overlays)

    mock_ax.vlines.assert_called_once()


@patch("samplepath.plots.helpers.plt")
def test_draw_step_chart_overlay_no_drop_lines_no_vlines(mock_plt, tmp_path):
    """drop_lines=False does not render vertical lines."""
    times = [_t("2024-01-01 00:00")]
    values = np.array([1.0])
    out_path = str(tmp_path / "chart.png")
    overlays = [
        ScatterOverlay(
            x=[_t("2024-01-01 00:00")],
            y=[1.0],
            color="green",
            label="Test",
            drop_lines=False,
        ),
    ]

    mock_ax = MagicMock()
    mock_fig = MagicMock()
    mock_plt.subplots.return_value = (mock_fig, mock_ax)

    draw_step_chart(times, values, "Title", "Y", out_path, overlays=overlays)

    mock_ax.vlines.assert_not_called()


@patch("samplepath.plots.helpers.plt")
def test_draw_step_chart_empty_overlay_skipped(mock_plt, tmp_path):
    """Overlay with empty x list is skipped."""
    times = [_t("2024-01-01 00:00")]
    values = np.array([1.0])
    out_path = str(tmp_path / "chart.png")
    overlays = [
        ScatterOverlay(
            x=[],
            y=[],
            color="green",
            label="Empty",
        ),
    ]

    mock_ax = MagicMock()
    mock_fig = MagicMock()
    mock_plt.subplots.return_value = (mock_fig, mock_ax)

    draw_step_chart(times, values, "Title", "Y", out_path, overlays=overlays)

    mock_ax.scatter.assert_not_called()


@patch("samplepath.plots.helpers.plt")
def test_draw_step_chart_multiple_overlays(mock_plt, tmp_path):
    """Multiple overlays each get rendered."""
    times = [_t("2024-01-01 00:00"), _t("2024-01-01 01:00"), _t("2024-01-01 02:00")]
    values = np.array([1.0, 2.0, 0.0])
    out_path = str(tmp_path / "chart.png")
    overlays = [
        ScatterOverlay(
            x=[_t("2024-01-01 00:00")],
            y=[1.0],
            color="purple",
            label="First",
        ),
        ScatterOverlay(
            x=[_t("2024-01-01 01:00")],
            y=[2.0],
            color="green",
            label="Second",
        ),
        ScatterOverlay(
            x=[_t("2024-01-01 02:00")],
            y=[0.0],
            color="red",
            label="Third",
        ),
    ]

    mock_ax = MagicMock()
    mock_fig = MagicMock()
    mock_plt.subplots.return_value = (mock_fig, mock_ax)

    draw_step_chart(times, values, "Title", "Y", out_path, overlays=overlays)

    assert mock_ax.scatter.call_count == 3


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Overlay helper and renderer tests
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def test_build_event_overlays_returns_none_if_missing_events():
    times = [_t("2024-01-01 00:00")]
    values = np.array([1.0])
    assert build_event_overlays(times, values, None, []) is None
    assert build_event_overlays(times, values, [], None) is None


def test_build_event_overlays_filters_and_maps_values():
    times = [_t("2024-01-01 00:00"), _t("2024-01-01 01:00")]
    values = np.array([1.0, 2.0])
    arrivals = [_t("2024-01-01 00:00"), _t("2024-01-02 00:00")]  # second should drop
    departures = [_t("2024-01-01 01:00")]

    overlays = build_event_overlays(times, values, arrivals, departures)
    assert overlays is not None
    arrival_overlay, departure_overlay = overlays
    assert arrival_overlay.x == [_t("2024-01-01 00:00")]
    assert arrival_overlay.y == [1.0]
    assert departure_overlay.x == [_t("2024-01-01 01:00")]
    assert departure_overlay.y == [2.0]


def test_build_event_overlays_sets_colors_labels_and_drop_lines():
    times = [_t("2024-01-01 00:00")]
    values = np.array([1.0])
    overlays = build_event_overlays(
        times,
        values,
        [_t("2024-01-01 00:00")],
        [_t("2024-01-01 00:00")],
        drop_lines_for_arrivals=True,
        drop_lines_for_departures=True,
    )
    assert overlays is not None
    arrival_overlay, departure_overlay = overlays
    assert arrival_overlay.color == "purple"
    assert arrival_overlay.label == "Arrival"
    assert arrival_overlay.drop_lines is True
    assert departure_overlay.color == "green"
    assert departure_overlay.label == "Departure"
    assert departure_overlay.drop_lines is True


def test_render_step_chart_renders_fill_and_overlays():
    ax = MagicMock()
    overlays = [
        ScatterOverlay(
            x=[_t("2024-01-01 00:00")],
            y=[1.0],
            color="red",
            label="evt",
            drop_lines=True,
        )
    ]
    times = [_t("2024-01-01 00:00"), _t("2024-01-01 01:00")]
    values = np.array([1.0, 2.0])

    render_step_chart(
        ax,
        times,
        values,
        label="N(t)",
        color="orange",
        fill=True,
        overlays=overlays,
    )

    ax.step.assert_called_once()
    ax.fill_between.assert_called_once()
    ax.vlines.assert_called_once()
    ax.scatter.assert_called_once()


def test_render_step_chart_skips_overlay_calls_when_none():
    ax = MagicMock()
    times = [_t("2024-01-01 00:00")]
    values = np.array([1.0])

    render_step_chart(ax, times, values, label="N(t)", color="blue", overlays=None)

    ax.scatter.assert_not_called()
    ax.vlines.assert_not_called()


def test_render_line_chart_plots_with_label_and_color():
    ax = MagicMock()
    times = [_t("2024-01-01 00:00")]
    values = np.array([1.0])

    render_line_chart(ax, times, values, label="L(T)", color="cyan")

    ax.plot.assert_called_once()
    call_args, call_kwargs = ax.plot.call_args
    assert call_args[0] == times
    assert call_args[1].tolist() == values.tolist()
    assert call_kwargs["label"] == "L(T)"
    assert call_kwargs["color"] == "cyan"


@patch("samplepath.plots.helpers._clip_axis_to_percentile")
def test_render_lambda_chart_invokes_clipping(mock_clip):
    ax = MagicMock()
    times = [_t("2024-01-01 00:00"), _t("2024-01-01 01:00")]
    values = np.array([1.0, 2.0])

    render_lambda_chart(
        ax,
        times,
        values,
        label="Λ(T)",
        pctl_upper=99.0,
        pctl_lower=1.0,
        warmup_hours=0.5,
    )

    mock_clip.assert_called_once_with(ax, list(times), values, 99.0, 1.0, 0.5)


@patch("samplepath.plots.helpers.render_step_chart")
@patch("samplepath.plots.helpers.build_event_overlays")
def test_render_N_chart_with_event_marks_uses_overlays(
    mock_build_overlays, mock_render_step_chart
):
    ax = MagicMock()
    times = [_t("2024-01-01 00:00")]
    values = np.array([1.0])
    overlays = [ScatterOverlay(x=[times[0]], y=[1.0], color="purple", label="Arrival")]
    mock_build_overlays.return_value = overlays

    render_N_chart(
        ax,
        times,
        values,
        arrival_times=[times[0]],
        departure_times=[times[0]],
        with_event_marks=True,
    )

    mock_render_step_chart.assert_called_once()
    _, kwargs = mock_render_step_chart.call_args
    assert kwargs["color"] == "grey"
    assert kwargs["overlays"] == overlays
    ax.set_title.assert_called_once()
    ax.set_ylabel.assert_called_once()


@patch("samplepath.plots.helpers.render_step_chart")
def test_render_N_chart_without_event_marks_uses_default_color(mock_render_step_chart):
    ax = MagicMock()
    times = [_t("2024-01-01 00:00")]
    values = np.array([1.0])

    render_N_chart(ax, times, values, with_event_marks=False)

    mock_render_step_chart.assert_called_once()
    _, kwargs = mock_render_step_chart.call_args
    assert kwargs["color"] == "tab:blue"
    assert kwargs["overlays"] is None


@patch("samplepath.plots.helpers.render_line_chart")
@patch("samplepath.plots.helpers.build_event_overlays")
def test_render_LT_chart_with_event_marks_uses_overlays(
    mock_build_overlays, mock_render_line_chart
):
    ax = MagicMock()
    times = [_t("2024-01-01 00:00")]
    values = np.array([2.0])
    overlays = [ScatterOverlay(x=[times[0]], y=[2.0], color="purple", label="Arrival")]
    mock_build_overlays.return_value = overlays

    render_LT_chart(
        ax,
        times,
        values,
        arrival_times=[times[0]],
        departure_times=[times[0]],
        with_event_marks=True,
    )

    mock_render_line_chart.assert_called_once()
    _, kwargs = mock_render_line_chart.call_args
    assert kwargs["color"] == "grey"
    assert kwargs["overlays"] == overlays
    ax.set_title.assert_called_once()
    ax.set_ylabel.assert_called_once()


@patch("samplepath.plots.helpers.render_line_chart")
def test_render_LT_chart_without_event_marks_uses_default_color(mock_render_line_chart):
    ax = MagicMock()
    times = [_t("2024-01-01 00:00")]
    values = np.array([2.0])

    render_LT_chart(ax, times, values, with_event_marks=False)

    mock_render_line_chart.assert_called_once()
    _, kwargs = mock_render_line_chart.call_args
    assert kwargs["color"] == "tab:blue"
    assert kwargs["overlays"] is None
