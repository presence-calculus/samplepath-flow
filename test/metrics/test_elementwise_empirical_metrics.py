# -*- coding: utf-8 -*-
# Copyright (c) 2025 Krishna Kumar
# SPDX-License-Identifier: MIT
# test/spath/metrics/test_empirical_series.py
import numpy as np
import pandas as pd
import pytest

from spath.metrics import compute_elementwise_empirical_metrics


def _t(s: str) -> pd.Timestamp:
    return pd.Timestamp(s)


@pytest.fixture
def tiny_df():
    # Three items: two complete, one incomplete
    # durations: [2h], [3h], [NaT]
    return pd.DataFrame(
        {
            "start_ts": [
                _t("2024-01-01 00:00"),
                _t("2024-01-01 01:00"),
                _t("2024-01-01 03:00"),
            ],
            "end_ts": [
                _t("2024-01-01 02:00"),
                _t("2024-01-01 04:00"),
                pd.NaT,
            ],
        }
    )


@pytest.fixture
def times():
    # t0, first completion, after both completions
    return [
        _t("2024-01-01 00:00"),
        _t("2024-01-01 02:00"),
        _t("2024-01-01 05:00"),
    ]


def test_shapes_match_times(tiny_df, times):
    ew = compute_elementwise_empirical_metrics(tiny_df, times)
    assert ew.W_star.shape == ew.lam_star.shape == (len(times),)


def test_initial_values_nan_when_elapsed_zero(tiny_df, times):
    ew = compute_elementwise_empirical_metrics(tiny_df, times)
    assert np.isnan(ew.W_star[0]) and np.isnan(ew.lam_star[0])


def test_W_star_after_first_completion_is_item_duration_mean(tiny_df, times):
    ew = compute_elementwise_empirical_metrics(tiny_df, times)
    # At 02:00, only first item completed: mean duration = 2.0 hours
    assert np.isclose(ew.W_star[1], 2.0)


def test_W_star_after_two_completions_is_mean_of_two(tiny_df, times):
    ew = compute_elementwise_empirical_metrics(tiny_df, times)
    # At 05:00, completed durations are 2h and 3h → mean = 2.5
    assert np.isclose(ew.W_star[2], 2.5)


def test_incomplete_items_ignored_in_W_star(tiny_df, times):
    ew = compute_elementwise_empirical_metrics(tiny_df, times)
    # The third (incomplete) item must not affect W*(t)
    # If it did, mean would deviate from 2.5 at t=05:00
    assert np.isclose(ew.W_star[2], 2.5)


def test_lam_star_uses_starts_per_elapsed_hour_at_first_completion(tiny_df, times):
    ew = compute_elementwise_empirical_metrics(tiny_df, times)
    # By 02:00, there are 2 starts (00:00 and 01:00); elapsed=2h → 2/2 = 1.0
    assert np.isclose(ew.lam_star[1], 1.0)


def test_lam_star_counts_incomplete_starts(tiny_df, times):
    ew = compute_elementwise_empirical_metrics(tiny_df, times)
    # By 05:00, there are 3 starts (including the incomplete one at 03:00); elapsed=5h → 3/5 = 0.6
    assert np.isclose(ew.lam_star[2], 0.6)


def test_empty_times_returns_empty_arrays(tiny_df):
    ew = compute_elementwise_empirical_metrics(tiny_df, [])
    assert ew.W_star.size == 0 and ew.lam_star.size == 0
