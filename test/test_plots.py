# -*- coding: utf-8 -*-
# Copyright (c) 2025 Krishna Kumar
# SPDX-License-Identifier: MIT
"""Tests for plotting functionality."""
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from samplepath.plots.helpers import ScatterOverlay, draw_step_chart


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
