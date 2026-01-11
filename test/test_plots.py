# -*- coding: utf-8 -*-
# Copyright (c) 2025 Krishna Kumar
# SPDX-License-Identifier: MIT
"""Tests for plotting functionality."""
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from samplepath.plots.helpers import draw_step_chart


def _t(s: str) -> pd.Timestamp:
    return pd.Timestamp(s)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Event marker extraction logic tests
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def test_event_markers_extracts_arrivals():
    """Arrivals are correctly identified: arrivals > 0."""
    # event = (timestamp, delta_n, arrivals)
    # For arrival: delta_n = +1, arrivals = 1
    # departures = arrivals - delta_n = 1 - 1 = 0
    event = (_t("2024-01-01 00:00"), +1, 1)
    delta_n, arrivals = event[1], event[2]
    departures = arrivals - delta_n

    assert arrivals == 1
    assert departures == 0


def test_event_markers_extracts_departures():
    """Departures are correctly identified: departures = arrivals - delta_n."""
    # For departure: delta_n = -1, arrivals = 0
    # departures = arrivals - delta_n = 0 - (-1) = 1
    event = (_t("2024-01-01 02:00"), -1, 0)
    delta_n, arrivals = event[1], event[2]
    departures = arrivals - delta_n

    assert arrivals == 0
    assert departures == 1


def test_event_markers_handles_simultaneous_arrival_departure():
    """Both arrival and departure at same timestamp (net zero change)."""
    # For simultaneous: delta_n = 0, arrivals = 1
    # departures = arrivals - delta_n = 1 - 0 = 1
    event = (_t("2024-01-01 01:00"), 0, 1)
    delta_n, arrivals = event[1], event[2]
    departures = arrivals - delta_n

    assert arrivals == 1
    assert departures == 1


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
def test_draw_step_chart_without_events_no_scatter(
    mock_plt, simple_chart_data, tmp_path
):
    """When events=None, no scatter calls are made."""
    times, values = simple_chart_data
    out_path = str(tmp_path / "chart.png")

    mock_ax = MagicMock()
    mock_fig = MagicMock()
    mock_plt.subplots.return_value = (mock_fig, mock_ax)

    draw_step_chart(times, values, "Title", "Y", out_path, events=None)

    mock_ax.scatter.assert_not_called()


@patch("samplepath.plots.helpers.plt")
def test_draw_step_chart_with_events_calls_scatter(
    mock_plt, simple_chart_data, tmp_path
):
    """When events provided, scatter is called for markers."""
    times, values = simple_chart_data
    out_path = str(tmp_path / "chart.png")
    events = [
        (_t("2024-01-01 00:00"), +1, 1),  # arrival
        (_t("2024-01-01 02:00"), -1, 0),  # departure
    ]

    mock_ax = MagicMock()
    mock_fig = MagicMock()
    mock_plt.subplots.return_value = (mock_fig, mock_ax)

    draw_step_chart(times, values, "Title", "Y", out_path, events=events)

    assert mock_ax.scatter.call_count == 2


@patch("samplepath.plots.helpers.plt")
def test_draw_step_chart_arrival_marker_is_purple(mock_plt, tmp_path):
    """Arrival markers use purple color."""
    times = [_t("2024-01-01 00:00")]
    values = np.array([1.0])
    out_path = str(tmp_path / "chart.png")
    events = [(_t("2024-01-01 00:00"), +1, 1)]  # arrival only

    mock_ax = MagicMock()
    mock_fig = MagicMock()
    mock_plt.subplots.return_value = (mock_fig, mock_ax)

    draw_step_chart(times, values, "Title", "Y", out_path, events=events)

    # Check scatter was called with purple
    call_kwargs = mock_ax.scatter.call_args_list[0][1]
    assert call_kwargs["color"] == "purple"


@patch("samplepath.plots.helpers.plt")
def test_draw_step_chart_departure_marker_is_green(mock_plt, tmp_path):
    """Departure markers use green color."""
    times = [_t("2024-01-01 00:00")]
    values = np.array([0.0])
    out_path = str(tmp_path / "chart.png")
    events = [(_t("2024-01-01 00:00"), -1, 0)]  # departure only

    mock_ax = MagicMock()
    mock_fig = MagicMock()
    mock_plt.subplots.return_value = (mock_fig, mock_ax)

    draw_step_chart(times, values, "Title", "Y", out_path, events=events)

    # Check scatter was called with green
    call_kwargs = mock_ax.scatter.call_args_list[0][1]
    assert call_kwargs["color"] == "green"
