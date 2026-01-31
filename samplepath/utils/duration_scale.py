# -*- coding: utf-8 -*-
# Copyright (c) 2026 Krishna Kumar
# SPDX-License-Identifier: MIT
"""Duration scale inference for display-layer unit conversion.

All internal computations use seconds.  This module picks the best
human-readable unit (seconds, minutes, hours, days, weeks) from a data
range and provides the divisor / labels needed by the chart layer.
"""
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class DurationScale:
    """Immutable descriptor for a display time-unit.

    Attributes
    ----------
    name : str
        Machine-readable name (``"seconds"``, ``"minutes"``, ``"hours"``, …).
    label : str
        Short label used in y-axis annotations (``"sec"``, ``"min"``, ``"hrs"``, …).
    divisor : float
        Factor to convert from seconds to this unit (divide durations,
        multiply rates).
    rate_label : str
        Label for the inverse unit (``"1/min"``, ``"1/hr"``, …).
    """

    name: str
    label: str
    divisor: float
    rate_label: str


SECONDS = DurationScale(name="seconds", label="sec", divisor=1.0, rate_label="1/sec")
MINUTES = DurationScale(name="minutes", label="min", divisor=60.0, rate_label="1/min")
HOURS = DurationScale(name="hours", label="hrs", divisor=3600.0, rate_label="1/hr")
DAYS = DurationScale(name="days", label="days", divisor=86_400.0, rate_label="1/day")
WEEKS = DurationScale(
    name="weeks", label="weeks", divisor=604_800.0, rate_label="1/week"
)
MONTHS = DurationScale(
    name="months", label="months", divisor=2_592_000.0, rate_label="1/month"
)
YEARS = DurationScale(
    name="years", label="years", divisor=31_536_000.0, rate_label="1/year"
)

_SCALES = [SECONDS, MINUTES, HOURS, DAYS, WEEKS, MONTHS, YEARS]


def infer_duration_scale(values_seconds: np.ndarray) -> DurationScale:
    """Pick the best display unit from a duration array (in seconds).

    Walks up the scale ladder (seconds → minutes → hours → days → weeks) and chooses the largest unit where the
    scaled maximum remains in a readable range (< ~1000 and >= 1). If
    nothing satisfies both bounds, fall back to the largest unit with
    scaled max >= 1. Defaults to HOURS when the input is empty or all-NaN.

    Parameters
    ----------
    values_seconds : np.ndarray
        Duration values in seconds. NaN and inf are ignored.
    """

    def select_scale(max_seconds: float) -> DurationScale:
        """Select the largest scale that keeps the max within a readable range."""
        readable_choice: DurationScale | None = None
        fallback_choice: DurationScale | None = None
        for scale in _SCALES:
            scaled = max_seconds / scale.divisor
            if scaled >= 1.0:
                fallback_choice = scale
            if 1.0 <= scaled < 1000.0:
                readable_choice = scale
            elif scaled < 1.0:
                break
        if readable_choice is not None:
            return readable_choice
        if fallback_choice is not None:
            return fallback_choice
        return HOURS

    finite = np.abs(values_seconds[np.isfinite(values_seconds)])
    if finite.size == 0:
        return HOURS
    max_val = float(np.max(finite))
    if max_val < 1.0:
        return SECONDS

    return select_scale(max_val)
