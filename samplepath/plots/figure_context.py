# -*- coding: utf-8 -*-
# Copyright (c) 2025 Krishna Kumar
# SPDX-License-Identifier: MIT
"""
Figure-level context helpers for plotting.

This module provides a simple context manager that owns Matplotlib figure
creation, optional shared x-axis date formatting, caption placement, saving,
and cleanup. Chart/layout code can focus on rendering onto the provided axes.
"""

from contextlib import contextmanager
from dataclasses import dataclass
import logging
import os
from typing import Callable, Iterator, Literal, Optional, Tuple, Union

from matplotlib import pyplot as plt
import numpy as np
from pandas.tseries.frequencies import to_offset

from samplepath.plots.chart_config import ChartConfig
from samplepath.plots.helpers import add_caption

logger = logging.getLogger(__name__)


def resolve_chart_path(
    out_dir: str, subdir: Optional[str], base_name: str, chart_format: str
) -> str:
    """Resolve an output path with subdir and the requested format extension."""
    root, ext = os.path.splitext(base_name)
    extension = chart_format.lstrip(".")
    filename = f"{root}.{extension}" if ext else f"{base_name}.{extension}"
    parts = [out_dir]
    if subdir:
        parts.append(subdir)
    parts.append(filename)
    return os.path.join(*parts)


def build_chart_save_kwargs(
    chart_format: str, chart_dpi: Optional[int]
) -> dict[str, object]:
    """Build savefig kwargs for the requested chart format."""
    kwargs: dict[str, object] = {"format": chart_format}
    if chart_format == "png" and chart_dpi is not None:
        kwargs["dpi"] = chart_dpi
    return kwargs


def _resolve_output(
    *,
    chart_config: Optional[ChartConfig],
    out_path: Optional[str],
    out_dir: Optional[str],
    subdir: Optional[str],
    base_name: Optional[str],
) -> tuple[str, dict[str, object]]:
    cfg = chart_config or ChartConfig()
    resolved_out_path = out_path
    if resolved_out_path is None:
        if not out_dir or not base_name:
            raise ValueError("layout_context requires out_path or (out_dir, base_name)")
        resolved_out_path = resolve_chart_path(
            out_dir, subdir, base_name, cfg.chart_format
        )
    resolved_save_kwargs = build_chart_save_kwargs(cfg.chart_format, cfg.chart_dpi)
    return resolved_out_path, resolved_save_kwargs


def is_date_axis(unit: Optional[str]) -> bool:
    """Return True if the axis label implies a date/time or pandas frequency axis."""
    if not unit:
        return False
    try:
        to_offset(unit)
        return True
    except Exception as exc:  # pragma: no cover - defensive logging
        logger.debug(
            "is_date_axis: failed to parse unit %r as pandas offset: %s", unit, exc
        )
    # Fallback to token-based detection on unit string.
    u = unit.lower()
    if any(
        token in u for token in ["timestamp", "datetime", "date", "day", "week", "time"]
    ):
        return True
    return False


def _first_axis(axes: Union[plt.Axes, np.ndarray]) -> plt.Axes:
    """Return the first axis from a single Axes or array of Axes."""
    if isinstance(axes, np.ndarray):
        return axes.flat[0]
    return axes


def _format_axis_label(ax, unit: Optional[str]) -> None:
    """
    Set a sensible x-axis label based on a pandas frequency alias or date-like unit.
    Examples:
      - D -> "Date (D)"
      - W-SUN -> "Date (W-SUN)"
      - MS -> "Date (MS)"
    Falls back to the raw unit string when not date-like.
    """
    label_val = unit
    if is_date_axis(unit):
        ax.set_xlabel(f"Date ({label_val})")
        try:
            ax.figure.autofmt_xdate()
        except Exception:
            logger.debug("figure_context: autofmt_xdate failed for label %r", label_val)
    elif unit:
        ax.set_xlabel(str(unit))


@dataclass
class LayoutSpec:
    nrows: int = 1
    ncols: int = 1
    figsize: Optional[Tuple[float, float]] = None
    sharex: bool = False
    sharey: bool = False


@dataclass
class FigureDecorSpec:
    suptitle: Optional[str] = None
    suptitle_y: float = 0.97
    caption: Optional[str] = None
    caption_position: Literal["top", "bottom"] = "bottom"
    caption_y: Optional[float] = None
    tight_layout: bool = True
    tight_layout_rect: Optional[Tuple[float, float, float, float]] = None


def _flatten_axes(axes: Union[plt.Axes, np.ndarray]) -> np.ndarray:
    if isinstance(axes, np.ndarray):
        return axes.ravel()
    if isinstance(axes, (list, tuple)):
        return np.array(axes, dtype=object).ravel()
    return np.array([axes], dtype=object).ravel()


def _format_targets(
    axes: Union[plt.Axes, np.ndarray],
    *,
    policy: Literal["all", "bottom_row", "left_col", "bottom_left"],
) -> np.ndarray:
    if not isinstance(axes, np.ndarray):
        return _flatten_axes(axes)
    if axes.ndim == 1:
        if policy in ("bottom_row", "bottom_left"):
            return axes[-1:].ravel()
        return axes.ravel()

    if policy == "all":
        return axes.ravel()
    if policy == "bottom_row":
        return axes[-1, :].ravel()
    if policy == "left_col":
        return axes[:, 0].ravel()
    if policy == "bottom_left":
        targets = np.concatenate([axes[-1, :].ravel(), axes[:, 0].ravel()])
        seen = set()
        uniq = []
        for ax in targets:
            key = id(ax)
            if key in seen:
                continue
            seen.add(key)
            uniq.append(ax)
        return np.array(uniq, dtype=object)
    return axes.ravel()


@contextmanager
def layout_context(
    out_path: Optional[str] = None,
    *,
    chart_config: Optional[ChartConfig] = None,
    layout: LayoutSpec,
    decor: Optional[FigureDecorSpec] = None,
    # x-axis formatting
    unit: Optional[str] = "timestamp",
    format_targets: Literal[
        "all", "bottom_row", "left_col", "bottom_left"
    ] = "bottom_row",
    format_axis_fn: Callable[[plt.Axes, Optional[str]], None] = _format_axis_label,
    # save configuration
    out_dir: Optional[str] = None,
    subdir: Optional[str] = None,
    base_name: Optional[str] = None,
) -> Iterator[Tuple[plt.Figure, Union[plt.Axes, np.ndarray], str]]:
    resolved_out_path, resolved_save_kwargs = _resolve_output(
        chart_config=chart_config,
        out_path=out_path,
        out_dir=out_dir,
        subdir=subdir,
        base_name=base_name,
    )
    effective_figsize = layout.figsize or (10.0, 3.4 * max(1, layout.nrows))
    fig, axes = plt.subplots(
        nrows=layout.nrows,
        ncols=layout.ncols,
        figsize=effective_figsize,
        sharex=layout.sharex,
        sharey=layout.sharey,
    )
    try:
        yield fig, axes, resolved_out_path

        targets = _format_targets(axes, policy=format_targets)
        for ax in targets:
            format_axis_fn(ax, unit=unit)

        if decor:
            if decor.suptitle:
                fig.suptitle(decor.suptitle, y=decor.suptitle_y)
            if decor.tight_layout:
                if decor.tight_layout_rect:
                    fig.tight_layout(rect=decor.tight_layout_rect)
                else:
                    fig.tight_layout()
            if decor.caption:
                if decor.caption_position == "top":
                    caption_y = 0.945 if decor.caption_y is None else decor.caption_y
                    fig.text(0.5, caption_y, decor.caption, ha="center", va="top")
                else:
                    add_caption(fig, decor.caption)
        fig.savefig(resolved_out_path, **resolved_save_kwargs)
    finally:
        plt.close(fig)


@contextmanager
def figure_context(
    out_path: Optional[str] = None,
    *,
    chart_config: Optional[ChartConfig] = None,
    # plot sizing
    nrows: int = 1,
    ncols: int = 1,
    figsize: Optional[Tuple[float, float]] = None,
    tight_layout: bool = True,
    # chart decorators
    caption: Optional[str] = None,
    # x-axis formatting
    unit: Optional[str] = "timestamp",
    sharex: bool = False,
    format_axis_fn: Callable[[plt.Axes, Optional[str]], None] = _format_axis_label,
    # save configuration
    out_dir: Optional[str] = None,
    subdir: Optional[str] = None,
    base_name: Optional[str] = None,
) -> Iterator[Tuple[plt.Figure, Union[plt.Axes, np.ndarray], str]]:
    layout = LayoutSpec(
        nrows=nrows,
        ncols=ncols,
        figsize=figsize,
        sharex=sharex,
        sharey=False,
    )
    decor = FigureDecorSpec(
        caption=caption,
        caption_position="bottom",
        tight_layout=tight_layout,
        tight_layout_rect=None,
    )
    format_policy = "bottom_row" if sharex else "all"
    with layout_context(
        out_path,
        chart_config=chart_config,
        layout=layout,
        decor=decor,
        unit=unit,
        format_targets=format_policy,
        format_axis_fn=format_axis_fn,
        out_dir=out_dir,
        subdir=subdir,
        base_name=base_name,
    ) as (fig, axes, resolved_out_path):
        yield fig, axes, resolved_out_path
