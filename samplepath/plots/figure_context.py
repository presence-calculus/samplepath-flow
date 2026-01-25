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
from typing import Callable, Iterator, Literal, Optional, Tuple, Union

from matplotlib import pyplot as plt
import numpy as np
from pandas.tseries.frequencies import to_offset

from samplepath.plots.helpers import add_caption

logger = logging.getLogger(__name__)


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
    out_path: str,
    *,
    layout: LayoutSpec,
    decor: Optional[FigureDecorSpec] = None,
    unit: Optional[str] = "timestamp",
    format_axis_fn: Callable[[plt.Axes, Optional[str]], None] = _format_axis_label,
    format_targets: Literal[
        "all", "bottom_row", "left_col", "bottom_left"
    ] = "bottom_row",
    save_kwargs: Optional[dict] = None,
) -> Iterator[Tuple[plt.Figure, Union[plt.Axes, np.ndarray]]]:
    effective_figsize = layout.figsize or (10.0, 3.4 * max(1, layout.nrows))
    fig, axes = plt.subplots(
        nrows=layout.nrows,
        ncols=layout.ncols,
        figsize=effective_figsize,
        sharex=layout.sharex,
        sharey=layout.sharey,
    )
    try:
        yield fig, axes

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
        fig.savefig(out_path, **(save_kwargs or {}))
    finally:
        plt.close(fig)


@contextmanager
def figure_context(
    out_path: str,
    *,
    nrows: int = 1,
    ncols: int = 1,
    figsize: Optional[Tuple[float, float]] = None,
    sharex: bool = False,
    caption: Optional[str] = None,
    unit: Optional[str] = "timestamp",
    save_kwargs: Optional[dict] = None,
    tight_layout: bool = True,
    format_axis_fn: Callable[[plt.Axes, Optional[str]], None] = _format_axis_label,
) -> Iterator[Tuple[plt.Figure, Union[plt.Axes, np.ndarray]]]:
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
        layout=layout,
        decor=decor,
        unit=unit,
        format_axis_fn=format_axis_fn,
        format_targets=format_policy,
        save_kwargs=save_kwargs,
    ) as (fig, axes):
        yield fig, axes
