# -*- coding: utf-8 -*-
# Copyright (c) 2025 Krishna Kumar
# SPDX-License-Identifier: MIT
# test/spath/metrics/test_end_effects.py
import numpy as np
import pandas as pd
import pytest

from spath.metrics import (
    compute_end_effect_series,
    compute_sample_path_metrics,
    compute_elementwise_empirical_metrics,
)

def _t(s: str) -> pd.Timestamp:
    return pd.Timestamp(s)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Fixtures: one simple 2h item; and shared arrays for A_vals, W_star
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

@pytest.fixture
def df_one_item():
    return pd.DataFrame(
        {
            "start_ts": [_t("2024-01-01 00:00")],
            "end_ts":   [_t("2024-01-01 02:00")],
        }
    )

@pytest.fixture
def times_three():
    # t0 (start), t_mid (1h), t1 (2h, completion)
    return [
        _t("2024-01-01 00:00"),
        _t("2024-01-01 01:00"),
        _t("2024-01-01 02:00"),
    ]

@pytest.fixture
def events_one_item(df_one_item):
    t0, t1 = df_one_item["start_ts"][0], df_one_item["end_ts"][0]
    # (time, dN, arrivals_increment)
    return [(t0, +1, 1), (t1, -1, 0)]

@pytest.fixture
def A_vals(events_one_item, times_three):
    # Use the module’s sampler to produce A aligned to times
    _, _, _, _, _, A, _, _ = compute_sample_path_metrics(events_one_item, times_three)
    return A

@pytest.fixture
def W_star_series(df_one_item, times_three):
    ew = compute_elementwise_empirical_metrics(df_one_item, times_three)
    return ew.W_star


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Shape & NaN-at-start behavior
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def test_shapes_match_times(df_one_item, times_three, A_vals, W_star_series):
    rA, rB, rho = compute_end_effect_series(df_one_item, times_three, A_vals, W_star_series)
    assert rA.shape == rB.shape == rho.shape == (len(times_three),)


def test_start_index_has_nans_when_elapsed_zero(df_one_item, times_three, A_vals, W_star_series):
    rA, rB, rho = compute_end_effect_series(df_one_item, times_three, A_vals, W_star_series)
    assert np.isnan(rA[0]) and np.isnan(rB[0]) and np.isnan(rho[0])


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# rA(T): end-effect share of area
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def test_rA_is_one_before_first_completion(df_one_item, times_three, A_vals, W_star_series):
    # At t_mid: A_full=0 (no fully contained items), A_T>0 → rA = 1
    rA, _, _ = compute_end_effect_series(df_one_item, times_three, A_vals, W_star_series)
    assert np.isclose(rA[1], 1.0)


def test_rA_is_zero_when_window_contains_only_full_items(df_one_item, times_three, A_vals, W_star_series):
    # At t1: the only item is fully contained → E=0 → rA=0
    rA, _, _ = compute_end_effect_series(df_one_item, times_three, A_vals, W_star_series)
    assert np.isclose(rA[2], 0.0)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# rB(T): boundary share of items started
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def test_rB_is_one_while_item_is_incomplete(df_one_item, times_three, A_vals, W_star_series):
    # At t_mid: total_started=1, incomplete_by_t=1 → rB=1
    _, rB, _ = compute_end_effect_series(df_one_item, times_three, A_vals, W_star_series)
    assert np.isclose(rB[1], 1.0)


def test_rB_is_zero_after_completion(df_one_item, times_three, A_vals, W_star_series):
    # At t1: total_started=1, incomplete_by_t=0 → rB=0
    _, rB, _ = compute_end_effect_series(df_one_item, times_three, A_vals, W_star_series)
    assert np.isclose(rB[2], 0.0)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# rho(T): elapsed_hours / W*(t)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def test_rho_nan_before_any_completion(df_one_item, times_three, A_vals, W_star_series):
    # W*(t_mid) is NaN (no completions yet) → rho is NaN
    _, _, rho = compute_end_effect_series(df_one_item, times_three, A_vals, W_star_series)
    assert np.isnan(rho[1])


def test_rho_equals_elapsed_over_mean_duration_at_completion(df_one_item, times_three, A_vals, W_star_series):
    # At t1: elapsed=2h, W*(t1)=2h → rho=1
    _, _, rho = compute_end_effect_series(df_one_item, times_three, A_vals, W_star_series)
    assert np.isclose(rho[2], 1.0)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Empty times
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def test_empty_times_return_empty_arrays(df_one_item):
    rA, rB, rho = compute_end_effect_series(df_one_item, [], np.array([]), np.array([]))
    assert rA.size == 0 and rB.size == 0 and rho.size == 0
