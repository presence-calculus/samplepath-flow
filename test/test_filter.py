# -*- coding: utf-8 -*-
# SPDX-License-Identifier: MIT
# test/spath/test_filter.py

import numpy as np
import pandas as pd
import pytest
import argparse

from spath.filter import (
    _parse_classes,
    _completed_base_label,
    _require_cols,
    _f_completed_only,
    _f_incomplete_only,
    _f_classes,
    _f_outlier_hours,
    _f_outlier_pctl,
    _f_outlier_iqr,
    run_filters,
    apply_filters,
    FilterSpec,
    FilterResult,
)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Fixtures
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

@pytest.fixture
def df_basic():
    """3 rows: one completed short, one completed long, one incomplete."""
    return pd.DataFrame(
        {
            "id": ["A", "B", "C"],
            "start_ts": pd.to_datetime(
                ["2024-01-01 00:00", "2024-01-02 00:00", "2024-01-03 00:00"]
            ),
            "end_ts": pd.to_datetime(
                ["2024-01-01 02:00", "2024-01-02 06:00", pd.NaT]
            ),
            "duration_hr": [2.0, 6.0, np.nan],
            "class": ["alpha", "beta", "gamma"],
        }
    )

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Helpers
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def test_parse_classes_none_returns_none():
    assert _parse_classes(None) is None


def test_parse_classes_normalizes_and_deduplicates():
    assert _parse_classes(" A, b ,A ") == ["a", "b"]


def test_completed_base_label_completed():
    spec = FilterSpec(completed_only=True)
    assert _completed_base_label(spec) == "closed elements"


def test_completed_base_label_incomplete():
    spec = FilterSpec(incomplete_only=True)
    assert _completed_base_label(spec) == "open elements"


def test_require_cols_raises_on_missing(df_basic):
    with pytest.raises(ValueError):
        _require_cols(df_basic[["id"]], ["start_ts"])

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Individual filters (signatures per filter.py)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def test_f_completed_only_keeps_only_completed(df_basic):
    spec = FilterSpec(completed_only=True)
    mask = pd.Series(True, index=df_basic.index)
    applied, dropped = [], {}
    new_mask = _f_completed_only(df_basic, mask, spec, applied, dropped)
    assert new_mask.sum() == 2


def test_f_incomplete_only_keeps_only_incomplete(df_basic):
    spec = FilterSpec(incomplete_only=True)
    mask = pd.Series(True, index=df_basic.index)
    applied, dropped = [], {}
    new_mask = _f_incomplete_only(df_basic, mask, spec, applied, dropped)
    assert new_mask.sum() == 1


def test_f_classes_selects_subset(df_basic):
    spec = FilterSpec(classes="alpha,gamma")
    mask = pd.Series(True, index=df_basic.index)
    applied, dropped = [], {}
    new_mask, label_add = _f_classes(df_basic, mask, spec, applied, dropped)
    kept = set(df_basic.loc[new_mask, "class"])
    assert kept == {"alpha", "gamma"}


def test_f_outlier_hours_keeps_shorter_and_incomplete(df_basic):
    spec = FilterSpec(outlier_hours=4)
    mask = pd.Series(True, index=df_basic.index)
    applied, dropped, outlier_tags = [], {}, []
    new_mask = _f_outlier_hours(df_basic, mask, spec, applied, dropped, outlier_tags)
    kept = df_basic.loc[new_mask]
    assert kept["duration_hr"].max() <= 4 or kept["end_ts"].isna().any()


def test_f_outlier_pctl_invalid_threshold_raises(df_basic):
    spec = FilterSpec(outlier_pctl=0)
    mask = pd.Series(True, index=df_basic.index)
    applied, dropped, thresholds, outlier_tags = [], {}, {}, []
    with pytest.raises(ValueError):
        _f_outlier_pctl(df_basic, mask, spec, applied, dropped, thresholds, outlier_tags)


def test_f_outlier_pctl_applies_to_completed_only(df_basic):
    spec = FilterSpec(outlier_pctl=90)
    mask = pd.Series(True, index=df_basic.index)
    applied, dropped, thresholds, outlier_tags = [], {}, {}, []
    new_mask = _f_outlier_pctl(df_basic, mask, spec, applied, dropped, thresholds, outlier_tags)
    completed = df_basic.loc[df_basic["end_ts"].notna() & new_mask]
    assert (completed["duration_hr"] <= completed["duration_hr"].max()).all()


def test_f_outlier_iqr_returns_original_mask_for_small_sample(df_basic):
    spec = FilterSpec(outlier_iqr=1.5)
    mask = pd.Series(True, index=df_basic.index)
    applied, dropped, thresholds, outlier_tags = [], {}, {}, []
    new_mask = _f_outlier_iqr(df_basic, mask, spec, applied, dropped, thresholds, outlier_tags)
    assert new_mask.equals(mask)


def test_f_outlier_iqr_two_sided_with_large_sample():
    df = pd.DataFrame(
        {
            "duration_hr": [1, 2, 3, 4, 5, 6, 7, 8],
            "end_ts": pd.date_range("2024-01-01", periods=8, freq="H"),
        }
    )
    spec = FilterSpec(outlier_iqr=1.5, outlier_iqr_two_sided=True)
    mask = pd.Series(True, index=df.index)
    applied, dropped, thresholds, outlier_tags = [], {}, {}, []
    new_mask = _f_outlier_iqr(df, mask, spec, applied, dropped, thresholds, outlier_tags)
    kept = df.loc[new_mask, "duration_hr"]
    low = thresholds.get("iqr_low_hr", kept.min())
    high = thresholds.get("iqr_high_hr", kept.max())
    assert kept.between(low, high).all()

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Driver + result objects
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def test_run_filters_mutual_exclusion(df_basic):
    spec = FilterSpec(completed_only=True, incomplete_only=True)
    with pytest.raises(ValueError):
        run_filters(df_basic, spec)


def test_run_filters_requires_columns(df_basic):
    spec = FilterSpec()
    bad = df_basic.drop(columns=["start_ts"])
    with pytest.raises(ValueError):
        run_filters(bad, spec)


def test_run_filters_applies_completed_only(df_basic):
    spec = FilterSpec(completed_only=True)
    res = run_filters(df_basic, spec)
    assert len(res.df) == 2


def test_run_filters_copy_result_true_returns_copy(df_basic):
    spec = FilterSpec(completed_only=True, copy_result=True)
    res = run_filters(df_basic, spec)
    assert res.df is not df_basic


def test_filterresult_label_includes_closed_elements(df_basic):
    spec = FilterSpec(completed_only=True)
    res = run_filters(df_basic, spec)
    assert "closed elements" in res.label


def test_filterresult_display_format():
    res = FilterResult(pd.DataFrame(), [], {}, {}, "demo")
    assert res.display == "Filters: demo"


def test_apply_filters_maps_namespace(df_basic):
    args = argparse.Namespace(
        completed=True,
        incomplete=False,
        classes=None,
        outlier_hours=None,
        outlier_pctl=None,
        outlier_iqr=None,
        outlier_iqr_two_sided=False,
    )
    res = apply_filters(df_basic, args)
    assert isinstance(res, FilterResult)
