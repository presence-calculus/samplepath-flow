# -*- coding: utf-8 -*-
# Copyright (c) 2026 Krishna Kumar
# SPDX-License-Identifier: MIT
from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class ChartConfig:
    """Chart-focused configuration derived from CLI args."""

    with_event_marks: bool = False
    show_derivations: bool = False
    lambda_pctl_upper: Optional[float] = None
    lambda_pctl_lower: Optional[float] = None
    lambda_warmup_hours: float = 0.0
    epsilon: float = 0.05
    horizon_days: float = 28.0
    chart_format: str = "png"
    chart_dpi: Optional[int] = 150

    @classmethod
    def init_from_args(cls, args: Optional[object]) -> "ChartConfig":
        if args is None:
            return cls()
        return cls(
            with_event_marks=getattr(args, "with_event_marks", False),
            show_derivations=getattr(args, "show_derivations", False),
            lambda_pctl_upper=getattr(args, "lambda_pctl", None),
            lambda_pctl_lower=getattr(args, "lambda_lower_pctl", None),
            lambda_warmup_hours=getattr(args, "lambda_warmup", 0.0),
            epsilon=getattr(args, "epsilon", 0.05),
            horizon_days=getattr(args, "horizon_days", 28.0),
            chart_format=getattr(args, "chart_format", "png"),
            chart_dpi=getattr(args, "chart_dpi", 150),
        )
