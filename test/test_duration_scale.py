# -*- coding: utf-8 -*-
# Copyright (c) 2026 Krishna Kumar
# SPDX-License-Identifier: MIT
"""Tests for duration scale inference."""

import numpy as np
import pytest

from samplepath.utils.duration_scale import (
    HOURS,
    SECONDS,
    DurationScale,
    infer_duration_scale,
)


def test_duration_scale_fields():
    scale = DurationScale(name="hours", label="hrs", divisor=3600.0, rate_label="1/hr")
    assert scale.name == "hours"
    assert scale.label == "hrs"
    assert scale.divisor == 3600.0
    assert scale.rate_label == "1/hr"


def test_duration_scale_is_frozen():
    scale = DurationScale(name="hours", label="hrs", divisor=3600.0, rate_label="1/hr")
    with pytest.raises(AttributeError):
        scale.name = "days"


@pytest.mark.parametrize(
    "max_seconds, expected_name",
    [
        (0.5, "seconds"),
        (30.0, "seconds"),
        (59.0, "seconds"),
        (60.0, "minutes"),
        (120.0, "minutes"),
        (3599.0, "minutes"),
        (3600.0, "hours"),
        (7200.0, "hours"),
        (86399.0, "hours"),
        (86400.0, "days"),
        (200_000.0, "days"),
        (604799.0, "days"),
        (604_800.0, "weeks"),
        (2_000_000.0, "weeks"),
        (2_592_000.0, "months"),
        (10_000_000.0, "months"),
        (31_536_000.0, "years"),
        (100_000_000.0, "years"),
    ],
)
def test_infer_duration_scale_from_max_value(max_seconds, expected_name):
    values = np.array([0.0, max_seconds / 2, max_seconds])
    result = infer_duration_scale(values)
    assert result.name == expected_name


def test_infer_duration_scale_empty_array_returns_hours():
    result = infer_duration_scale(np.array([]))
    assert result.name == "hours"


def test_infer_duration_scale_all_nan_returns_hours():
    result = infer_duration_scale(np.array([np.nan, np.nan]))
    assert result.name == "hours"


def test_infer_duration_scale_all_zero_returns_minutes():
    result = infer_duration_scale(np.array([0.0, 0.0]))
    assert result.name == "seconds"


def test_infer_duration_scale_negative_values_uses_abs():
    values = np.array([-200_000.0, 100.0])
    result = infer_duration_scale(values)
    assert result.name == "days"


def test_infer_duration_scale_very_small_returns_minutes():
    values = np.array([0.5, 1.0])
    result = infer_duration_scale(values)
    assert result.name == "seconds"


def test_hours_constant_is_hours_scale():
    assert HOURS.name == "hours"
    assert HOURS.divisor == 3600.0


def test_seconds_constant_is_seconds_scale():
    assert SECONDS.name == "seconds"
    assert SECONDS.divisor == 1.0
