# -*- coding: utf-8 -*-
# Copyright (c) 2026 Krishna Kumar
# SPDX-License-Identifier: MIT
from dataclasses import dataclass, replace
from typing import Optional

import pandas as pd
from pandas.tseries.frequencies import to_offset

from samplepath.utils.duration_scale import DurationScale


@dataclass(frozen=True)
class ColorConfig:
    arrival_color: str = "rebeccapurple"
    departure_color: str = "forestgreen"


@dataclass(frozen=True)
class ChartConfig:
    """Chart-focused configuration derived from CLI args."""

    with_event_marks: bool = False
    show_derivations: bool = False
    lambda_pctl_upper: Optional[float] = None
    lambda_pctl_lower: Optional[float] = None
    lambda_warmup_seconds: float = 0.0
    epsilon: float = 0.05
    horizon_seconds: float = 28.0 * 86400.0
    duration_scale: Optional[DurationScale] = None
    chart_format: str = "png"
    chart_dpi: Optional[int] = 150
    sampling_frequency: Optional[str] = None
    grid_lines: bool = True

    @classmethod
    def init_from_args(cls, args: Optional[object]) -> "ChartConfig":
        if args is None:
            return cls()
        lambda_warmup_hours = float(getattr(args, "lambda_warmup", 0.0) or 0.0)
        horizon_days = float(getattr(args, "horizon_days", 28.0) or 28.0)
        return cls(
            with_event_marks=getattr(args, "with_event_marks", False),
            show_derivations=getattr(args, "show_derivations", False),
            lambda_pctl_upper=getattr(args, "lambda_pctl", None),
            lambda_pctl_lower=getattr(args, "lambda_lower_pctl", None),
            lambda_warmup_seconds=lambda_warmup_hours * 3600.0,
            epsilon=getattr(args, "epsilon", 0.05),
            horizon_seconds=horizon_days * 86400.0,
            chart_format=getattr(args, "chart_format", "png"),
            chart_dpi=getattr(args, "chart_dpi", 150),
            sampling_frequency=getattr(args, "sampling_frequency", None),
            grid_lines=getattr(args, "grid_lines", True),
        )

    def with_duration_scale(self, scale: DurationScale) -> "ChartConfig":
        return replace(self, duration_scale=scale)

    @staticmethod
    def freq_display_label(unit: Optional[str]) -> str:
        """Reverse-map a pandas frequency alias to a human-readable x-axis label.

        Returns ``"Timestamp"`` for ``None``, ``"timestamp"``, or unrecognised
        values.  Otherwise returns ``"Time (<readable>)"`` where *readable* is
        a friendly name derived from the pandas offset type.
        """
        _FALLBACK = "Event Timeline"
        if unit is None or unit == "timestamp":
            return _FALLBACK
        try:
            offset = to_offset(unit)
        except (ValueError, TypeError):
            return _FALLBACK

        _WEEKDAY_NAMES = ["MON", "TUE", "WED", "THU", "FRI", "SAT", "SUN"]
        _MONTH_NAMES = [
            "JAN",
            "FEB",
            "MAR",
            "APR",
            "MAY",
            "JUN",
            "JUL",
            "AUG",
            "SEP",
            "OCT",
            "NOV",
            "DEC",
        ]

        if isinstance(offset, pd.tseries.offsets.Day):
            return f"Day"
        if isinstance(offset, pd.tseries.offsets.Week):
            anchor = _WEEKDAY_NAMES[offset.weekday]
            return f"Week ({anchor})"
        if isinstance(offset, pd.tseries.offsets.MonthBegin):
            return "Month"
        if isinstance(offset, pd.tseries.offsets.QuarterBegin):
            anchor = _MONTH_NAMES[offset.startingMonth - 1]
            return f"Quarter ({anchor})"
        if isinstance(offset, pd.tseries.offsets.YearBegin):
            anchor = _MONTH_NAMES[offset.month - 1]
            return f"Year ({anchor})"

        return _FALLBACK
