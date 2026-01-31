# -*- coding: utf-8 -*-
# Copyright (c) 2025 Krishna Kumar
# SPDX-License-Identifier: MIT
import os
from unittest.mock import MagicMock, patch

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")

from samplepath.plots.chart_config import ChartConfig
from samplepath.plots.figure_context import (
    FigureDecorSpec,
    LayoutSpec,
    _format_axis_label,
    figure_context,
    layout_context,
)

# - Basic layout contracts


def test_returns_single_axis_for_1x1(tmp_path):
    out_path = tmp_path / "chart.png"
    with figure_context(str(out_path), nrows=1, ncols=1, unit="timestamp") as (
        fig,
        axes,
        _,
    ):
        assert fig is not None
        # Matplotlib returns a single Axes instance for 1x1
        assert not isinstance(axes, (list, tuple))
        assert not isinstance(axes, np.ndarray)


def test_returns_array_for_multi_axes(tmp_path):
    out_path = tmp_path / "chart.png"
    with figure_context(str(out_path), nrows=2, ncols=2, unit="timestamp") as (
        fig,
        axes,
        _,
    ):
        assert fig is not None
        assert isinstance(axes, np.ndarray)
        assert axes.shape == (2, 2)


# Formatting x-axis dates
def test_format_axis_label_date_like_calls_autofmt_and_label():
    ax = MagicMock()
    with patch("samplepath.plots.figure_context.is_date_axis", return_value=True):
        _format_axis_label(ax, unit="W-SUN")
    ax.set_xlabel.assert_called_once_with("Week (SUN)")
    ax.figure.autofmt_xdate.assert_called_once()


def test_format_axis_label_non_date_sets_raw_label():
    ax = MagicMock()
    with patch("samplepath.plots.figure_context.is_date_axis", return_value=False):
        _format_axis_label(ax, unit="hours")
    ax.set_xlabel.assert_called_once_with("hours")
    ax.figure.autofmt_xdate.assert_not_called()


def test_sharex_formats_only_bottom_axis(tmp_path):
    out_path = tmp_path / "chart.png"
    formatter = MagicMock()
    with figure_context(
        str(out_path), nrows=2, sharex=True, unit="timestamp", format_axis_fn=formatter
    ) as (_, axes, _):
        pass

    assert formatter.call_count == 1
    formatter.assert_called_once_with(axes[-1], unit="timestamp")


def test_non_sharex_formats_all_axes(tmp_path):
    out_path = tmp_path / "chart.png"
    formatter = MagicMock()
    with figure_context(
        str(out_path), nrows=2, sharex=False, unit="timestamp", format_axis_fn=formatter
    ) as (_, axes, _):
        pass

    assert formatter.call_count == 2
    formatter.assert_any_call(axes[0], unit="timestamp")
    formatter.assert_any_call(axes[1], unit="timestamp")


def test_invalid_freq_like_unit_still_formats_axes(tmp_path):
    out_path = tmp_path / "chart.png"
    formatter = MagicMock()
    with figure_context(str(out_path), unit="not-a-freq", format_axis_fn=formatter) as (
        _,
        ax,
        _,
    ):
        pass

    formatter.assert_called_once_with(ax, unit="not-a-freq")


def test_is_date_axis_token_fallback(tmp_path):
    out_path = tmp_path / "chart.png"
    with figure_context(str(out_path), unit="timestamp") as (_, ax, _):
        pass
    assert ax.get_xlabel() == "Event Timeline"


def test_multi_column_axes_formatting(tmp_path):
    out_path = tmp_path / "chart.png"
    formatter = MagicMock()
    with figure_context(
        str(out_path),
        nrows=1,
        ncols=2,
        sharex=False,
        unit="timestamp",
        format_axis_fn=formatter,
    ) as (_, axes, _):
        pass
    assert formatter.call_count == 2
    formatter.assert_any_call(axes[0], unit="timestamp")
    formatter.assert_any_call(axes[1], unit="timestamp")


def test_figsize_scales_with_rows(tmp_path):
    out_path = tmp_path / "chart.png"
    with figure_context(str(out_path), nrows=3, ncols=1, unit="timestamp") as (
        fig,
        _,
        _,
    ):
        width, height = fig.get_size_inches()
        assert height >= 3.4 * 3 - 0.1


@patch("samplepath.plots.figure_context.plt")
def test_chart_config_dpi_applied(mock_plt, tmp_path):
    fig = MagicMock()
    ax = MagicMock()
    mock_plt.subplots.return_value = (fig, ax)

    out_path = tmp_path / "chart.png"
    with figure_context(
        str(out_path),
        chart_config=ChartConfig(chart_format="png", chart_dpi=200),
        unit="timestamp",
    ):
        pass

    fig.savefig.assert_called_once_with(str(out_path), format="png", dpi=200)
    mock_plt.close.assert_called_once_with(fig)


@patch("samplepath.plots.figure_context.add_caption")
@patch("samplepath.plots.figure_context.plt")
def test_tight_layout_runs_before_caption(mock_plt, mock_add_caption, tmp_path):
    fig = MagicMock()
    ax = MagicMock()
    mock_plt.subplots.return_value = (fig, ax)
    order = []
    fig.tight_layout.side_effect = lambda: order.append("tight")
    mock_add_caption.side_effect = lambda f, c: order.append("caption")

    out_path = tmp_path / "chart.png"
    with figure_context(str(out_path), unit="timestamp", caption="cap"):
        pass

    assert order == ["tight", "caption"]


# tight layout
@patch("samplepath.plots.figure_context.plt")
def test_figure_context_respects_tight_layout_flag(mock_plt, tmp_path):
    fig = MagicMock()
    ax = MagicMock()
    mock_plt.subplots.return_value = (fig, ax)

    out_path = tmp_path / "chart.png"
    with figure_context(str(out_path), unit="timestamp", tight_layout=False):
        pass

    fig.tight_layout.assert_not_called()
    fig.savefig.assert_called_once_with(str(out_path), format="png", dpi=150)
    mock_plt.close.assert_called_once_with(fig)


# save and close


@patch("samplepath.plots.figure_context.add_caption")
@patch("samplepath.plots.figure_context.plt")
def test_figure_context_saves_and_closes(mock_plt, mock_add_caption, tmp_path):
    fig = MagicMock()
    ax = MagicMock()
    mock_plt.subplots.return_value = (fig, ax)

    out_path = tmp_path / "chart.png"
    with figure_context(str(out_path), unit="timestamp", caption="cap"):
        pass

    fig.savefig.assert_called_once_with(str(out_path), format="png", dpi=150)
    mock_plt.close.assert_called_once_with(fig)
    mock_add_caption.assert_called_once_with(fig, "cap")
    fig.tight_layout.assert_called_once()


@patch("samplepath.plots.figure_context.plt")
def test_figure_context_always_closes_on_error(mock_plt, tmp_path):
    fig = MagicMock()
    ax = MagicMock()
    mock_plt.subplots.return_value = (fig, ax)

    out_path = tmp_path / "chart.png"
    with pytest.raises(RuntimeError):
        with figure_context(str(out_path), unit="timestamp"):
            raise RuntimeError("boom")

    mock_plt.close.assert_called_once_with(fig)


@patch("samplepath.plots.figure_context.plt")
def test_layout_context_caption_top_defaults_y(mock_plt, tmp_path):
    fig = MagicMock()
    ax = MagicMock()
    mock_plt.subplots.return_value = (fig, ax)
    out_path = tmp_path / "chart.png"
    decor = FigureDecorSpec(caption="Filters: test", caption_position="top")
    layout = LayoutSpec()
    with layout_context(str(out_path), layout=layout, decor=decor):
        pass
    fig.text.assert_called_once_with(0.5, 0.945, "Filters: test", ha="center", va="top")


@patch("samplepath.plots.figure_context.add_caption")
@patch("samplepath.plots.figure_context.plt")
def test_layout_context_caption_bottom_defaults_y(mock_plt, mock_add_caption, tmp_path):
    fig = MagicMock()
    ax = MagicMock()
    mock_plt.subplots.return_value = (fig, ax)
    out_path = tmp_path / "chart.png"
    decor = FigureDecorSpec(caption="cap", caption_position="bottom")
    layout = LayoutSpec()
    with layout_context(str(out_path), layout=layout, decor=decor):
        pass
    mock_add_caption.assert_called_once_with(fig, "cap")


@patch("samplepath.plots.figure_context.plt")
def test_layout_context_format_targets_bottom_row(mock_plt, tmp_path):
    fig = MagicMock()
    axes = np.array([[object(), object()], [object(), object()]], dtype=object)
    mock_plt.subplots.return_value = (fig, axes)
    formatter = MagicMock()
    out_path = tmp_path / "chart.png"
    layout = LayoutSpec(nrows=2, ncols=2, sharex=True)
    with layout_context(
        str(out_path),
        layout=layout,
        format_axis_fn=formatter,
        format_targets="bottom_row",
    ):
        pass
    assert formatter.call_count == 2
    assert formatter.call_count == 2


@patch("samplepath.plots.figure_context.plt")
def test_layout_context_format_targets_left_col(mock_plt, tmp_path):
    fig = MagicMock()
    axes = np.array([[object(), object()], [object(), object()]], dtype=object)
    mock_plt.subplots.return_value = (fig, axes)
    formatter = MagicMock()
    out_path = tmp_path / "chart.png"
    layout = LayoutSpec(nrows=2, ncols=2, sharey=True)
    with layout_context(
        str(out_path),
        layout=layout,
        format_axis_fn=formatter,
        format_targets="left_col",
    ):
        pass
    assert formatter.call_count == 2


@patch("samplepath.plots.figure_context.plt")
def test_layout_context_format_targets_bottom_left(mock_plt, tmp_path):
    fig = MagicMock()
    axes = np.array([[object(), object()], [object(), object()]], dtype=object)
    mock_plt.subplots.return_value = (fig, axes)
    formatter = MagicMock()
    out_path = tmp_path / "chart.png"
    layout = LayoutSpec(nrows=2, ncols=2, sharex=True, sharey=True)
    with layout_context(
        str(out_path),
        layout=layout,
        format_axis_fn=formatter,
        format_targets="bottom_left",
    ):
        pass
    assert formatter.call_count == 3


@patch("samplepath.plots.figure_context.plt")
def test_layout_context_uses_tight_layout_rect(mock_plt, tmp_path):
    fig = MagicMock()
    ax = MagicMock()
    mock_plt.subplots.return_value = (fig, ax)
    out_path = tmp_path / "chart.png"
    decor = FigureDecorSpec(tight_layout=True, tight_layout_rect=(0, 0, 1, 0.9))
    layout = LayoutSpec()
    with layout_context(str(out_path), layout=layout, decor=decor):
        pass
    fig.tight_layout.assert_called_once_with(rect=(0, 0, 1, 0.9))


@patch("samplepath.plots.figure_context.plt")
def test_layout_context_applies_suptitle(mock_plt, tmp_path):
    fig = MagicMock()
    ax = MagicMock()
    mock_plt.subplots.return_value = (fig, ax)
    out_path = tmp_path / "chart.png"
    decor = FigureDecorSpec(suptitle="Title", suptitle_y=0.9)
    layout = LayoutSpec()
    with layout_context(str(out_path), layout=layout, decor=decor):
        pass
    fig.suptitle.assert_called_once_with("Title", y=0.9)


def test_layout_context_figsize_autoscale(tmp_path):
    out_path = tmp_path / "chart.png"
    layout = LayoutSpec(nrows=3, ncols=1, figsize=None)
    with layout_context(str(out_path), layout=layout) as (fig, _, _):
        width, height = fig.get_size_inches()
        assert height >= 3.4 * 3 - 0.1


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# _format_axis_label calendar tick integration
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

import matplotlib.dates as mdates


def test_format_axis_label_calendar_unit_sets_locator_and_xlabel():
    ax = MagicMock()
    _format_axis_label(ax, unit="MS")
    ax.xaxis.set_major_locator.assert_called_once()
    locator = ax.xaxis.set_major_locator.call_args[0][0]
    assert isinstance(locator, mdates.MonthLocator)
    ax.set_xlabel.assert_called_once_with("Month")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# ChartConfig.freq_display_label tests
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


@pytest.mark.parametrize(
    "unit, expected",
    [
        (None, "Event Timeline"),
        ("timestamp", "Event Timeline"),
        ("D", "Day"),
        ("W-MON", "Week (MON)"),
        ("W-SUN", "Week (SUN)"),
        ("MS", "Month"),
        ("QS-JAN", "Quarter (JAN)"),
        ("QS-APR", "Quarter (APR)"),
        ("YS-JAN", "Year (JAN)"),
        ("YS-JUL", "Year (JUL)"),
    ],
)
def test_freq_display_label(unit, expected):
    assert ChartConfig.freq_display_label(unit) == expected


def test_freq_display_label_unrecognised_freq_falls_back():
    assert ChartConfig.freq_display_label("not-a-freq") == "Event Timeline"
