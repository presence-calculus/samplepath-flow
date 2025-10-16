# -*- coding: utf-8 -*-
# Copyright (c) 2025 Krishna Kumar
# SPDX-License-Identifier: MIT
# test/spath/metrics/test_sample_path_metrics.py
import numpy as np
import pandas as pd
import pytest

from spath.metrics import compute_sample_path_metrics


def _t(s: str) -> pd.Timestamp:
    return pd.Timestamp(s)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  SINGLE BOXCAR (one item active for 2 hours)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

@pytest.fixture
def single_boxcar():
    t0 = _t("2024-01-01 00:00")
    t1 = _t("2024-01-01 02:00")
    events = [(t0, +1, 1), (t1, -1, 0)]
    times = [t0, t1]
    return compute_sample_path_metrics(events, times)


def test_single_boxcar_shapes(single_boxcar):
    T, L, Lam, w, N, A, Arr, Dep = single_boxcar
    assert all(len(x) == 2 for x in (L, Lam, w, N, A, Arr, Dep))


def test_single_boxcar_initial_values(single_boxcar):
    _, L, Lam, w, N, A, Arr, Dep = single_boxcar
    assert np.isnan(L[0]) and np.isnan(Lam[0]) and np.isclose(w[0], 0.0)


def test_single_boxcar_initial_counts(single_boxcar):
    _, _, _, _, N, A, Arr, Dep = single_boxcar
    assert N[0] == 1 and np.isclose(A[0], 0.0) and Arr[0] == 1 and Dep[0] == 0


def test_single_boxcar_final_area(single_boxcar):
    _, _, _, _, _, A, _, _ = single_boxcar
    assert np.isclose(A[1], 2.0)


def test_single_boxcar_final_L(single_boxcar):
    _, L, _, _, _, _, _, _ = single_boxcar
    assert np.isclose(L[1], 1.0)


def test_single_boxcar_final_Lambda(single_boxcar):
    _, _, Lam, _, _, _, _, _ = single_boxcar
    assert np.isclose(Lam[1], 0.5)


def test_single_boxcar_final_w(single_boxcar):
    _, _, _, w, _, _, _, _ = single_boxcar
    assert np.isclose(w[1], 2.0)


def test_single_boxcar_final_counts(single_boxcar):
    _, _, _, _, N, _, Arr, Dep = single_boxcar
    assert N[1] == 0 and Arr[1] == 1 and Dep[1] == 1


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  OVERLAPPING ITEMS (peak N = 2)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

@pytest.fixture
def overlapping_items():
    t0 = _t("2024-01-01 00:00")
    t1 = _t("2024-01-01 01:00")
    t2 = _t("2024-01-01 03:30")
    t3 = _t("2024-01-01 05:00")
    events = [(t0, +1, 1), (t1, +1, 1), (t2, -1, 0), (t3, -1, 0)]
    times = [t0, t1, t2, t3]
    return compute_sample_path_metrics(events, times)


def test_overlapping_items_N_sequence(overlapping_items):
    _, _, _, _, N, _, _, _ = overlapping_items
    assert np.all(N == np.array([1, 2, 1, 0]))


def test_overlapping_items_final_area(overlapping_items):
    T, _, _, _, _, A, _, _ = overlapping_items
    t0, t1, t2, t3 = T
    expected_area = (
        1 * (t1 - t0).total_seconds() / 3600.0
        + 2 * (t2 - t1).total_seconds() / 3600.0
        + 1 * (t3 - t2).total_seconds() / 3600.0
    )
    assert np.isclose(A[-1], expected_area)


def test_overlapping_items_arrivals(overlapping_items):
    _, _, _, _, _, _, Arr, _ = overlapping_items
    assert np.all(Arr == np.array([1, 2, 2, 2]))


def test_overlapping_items_departures(overlapping_items):
    _, _, _, _, _, _, _, Dep = overlapping_items
    assert np.all(Dep == np.array([0, 0, 1, 2]))


def test_overlapping_items_final_L_equals_A_over_elapsed(overlapping_items):
    T, L, _, _, _, A, _, _ = overlapping_items
    elapsed = (T[-1] - T[0]).total_seconds() / 3600.0
    assert np.isclose(L[-1], A[-1] / elapsed)


def test_overlapping_items_final_w_equals_A_over_arrivals(overlapping_items):
    _, _, _, w, _, A, Arr, _ = overlapping_items
    assert np.isclose(w[-1], A[-1] / Arr[-1])


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  SAME-TIMESTAMP TIES (arrival then departure)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

@pytest.fixture
def same_timestamp_tie():
    t0 = _t("2024-01-01 00:00")
    events = [(t0, +1, 1), (t0, -1, 0)]
    times = [t0]
    return compute_sample_path_metrics(events, times)


def test_same_timestamp_tie_final_counts(same_timestamp_tie):
    _, _, _, _, N, _, Arr, Dep = same_timestamp_tie
    assert N[0] == 0 and Arr[0] == 1 and Dep[0] == 1


def test_same_timestamp_tie_nan_metrics(same_timestamp_tie):
    _, L, Lam, w, _, A, _, _ = same_timestamp_tie
    assert np.isnan(L[0]) and np.isnan(Lam[0]) and np.isclose(A[0], 0.0) and np.isclose(w[0], 0.0)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  EMPTY INPUTS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def test_empty_events_returns_empty_arrays():
    t0 = _t("2024-01-01 00:00")
    times = [t0]
    events = []
    T, L, Lam, w, N, A, Arr, Dep = compute_sample_path_metrics(events, times)
    for arr in (L, Lam, w, N, A, Arr, Dep):
        assert arr.size == 0
