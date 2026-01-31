import pandas as pd
import pytest

from samplepath.metrics import _align_to_boundary


def _t(s: str) -> pd.Timestamp:
    return pd.Timestamp(s)


def test_fixed_freq_aligns_to_midnight():
    ts = _t("2024-01-15 14:30")
    result = _align_to_boundary(ts, "D")
    assert result == _t("2024-01-15 00:00")


def test_week_aligns_to_preceding_sunday():
    # 2024-01-10 is a Wednesday
    ts = _t("2024-01-10 14:30")
    result = _align_to_boundary(ts, "W-SUN")
    # Preceding W-SUN boundary is Sunday 2024-01-07
    assert result == _t("2024-01-07")
    assert result.weekday() == 6


def test_month_start_aligns_to_first_of_month():
    ts = _t("2024-03-15 10:00")
    result = _align_to_boundary(ts, "MS")
    assert result == _t("2024-03-01")


def test_timestamp_on_boundary_returns_same():
    ts = _t("2024-03-01 00:00")
    result = _align_to_boundary(ts, "MS")
    assert result == ts


def test_quarter_start_aligns_to_quarter_boundary():
    ts = _t("2024-05-20 10:00")
    result = _align_to_boundary(ts, "QS-JAN")
    assert result == _t("2024-04-01")


def test_year_start_aligns_to_year_boundary():
    ts = _t("2024-06-15 10:00")
    result = _align_to_boundary(ts, "YS-JAN")
    assert result == _t("2024-01-01")
