# -*- coding: utf-8 -*-
# Copyright (c) 2025 Krishna Kumar
# SPDX-License-Identifier: MIT
# test/spath/metrics/test_tracking_and_coherence.py
import numpy as np
import pandas as pd
import pytest

from spath.metrics import compute_tracking_errors, compute_coherence_score


def _make_times(n=4, start="2024-01-01 00:00", step_hours=1):
    t0 = pd.Timestamp(start)
    return [t0 + pd.Timedelta(hours=i * step_hours) for i in range(n)]


def test_compute_tracking_errors_perfect_match():
    times = _make_times(4)
    # perfect match: zero relative errors everywhere (where denominators > 0)
    W_star = np.array([1.0, 2.0, 3.0, 4.0])
    w_vals = W_star.copy()

    lam_star = np.array([0.5, 1.0, 1.5, 2.0])
    lam_vals = lam_star.copy()

    eW, eLam, elapsed = compute_tracking_errors(times, w_vals, lam_vals, W_star, lam_star)

    assert np.allclose(elapsed, [0.0, 1.0, 2.0, 3.0])
    assert np.allclose(eW[np.isfinite(W_star) & (W_star > 0)], 0.0)
    assert np.allclose(eLam[np.isfinite(lam_star) & (lam_star > 0)], 0.0)


def test_compute_tracking_errors_constant_offset_on_lambda_only():
    times = _make_times(4)

    # w side: identical -> zero eW
    W_star = np.array([1.0, 2.0, 3.0, 4.0])
    w_vals = W_star.copy()

    # lambda side: constant +0.5 offset
    lam_star = np.array([1.0, 2.0, 3.0, 4.0])
    lam_vals = lam_star + 0.5

    eW, eLam, elapsed = compute_tracking_errors(times, w_vals, lam_vals, W_star, lam_star)

    assert np.allclose(eW, 0.0)
    expected_rel = np.abs(lam_vals - lam_star) / lam_star
    assert np.allclose(eLam, expected_rel, rtol=1e-9, atol=1e-12)
    assert np.allclose(elapsed, [0.0, 1.0, 2.0, 3.0])


def test_compute_tracking_errors_with_nans_masked():
    times = _make_times(4)

    W_star  = np.array([np.nan, 2.0, np.nan, 4.0])
    w_vals  = np.array([np.nan, 2.2, 3.0,  np.nan])

    lam_star = np.array([1.0,   np.nan, 3.0,  4.0])
    lam_vals = np.array([1.1,   2.0,    np.nan, 4.4])

    eW, eLam, elapsed = compute_tracking_errors(times, w_vals, lam_vals, W_star, lam_star)

    # eW: only index 1 has finite W_star>0 AND finite w_vals
    assert np.isnan(eW[0])
    assert np.isclose(eW[1], np.abs(2.2 - 2.0) / 2.0)
    assert np.isnan(eW[2])
    assert np.isnan(eW[3])

    # eLam: indices 0 and 3 are computable
    assert np.isclose(eLam[0], np.abs(1.1 - 1.0) / 1.0)
    assert np.isnan(eLam[1])
    assert np.isnan(eLam[2])
    assert np.isclose(eLam[3], np.abs(4.4 - 4.0) / 4.0)

    assert np.allclose(elapsed, [0.0, 1.0, 2.0, 3.0])


def test_compute_coherence_score_basic_pass():
    # eW and eLam both small; elapsed meets horizon on first three
    eW = np.array([0.05, 0.10, 0.20, 0.05])
    eLam = np.array([0.02, 0.05, 0.10, 0.02])
    elapsed = np.array([100.0, 100.0, 100.0, 50.0])
    epsilon = 0.20
    horizon_hours = 80.0

    score, coherent, total = compute_coherence_score(eW, eLam, elapsed, epsilon, horizon_hours)
    assert coherent == 3
    assert total == 3
    assert np.isclose(score, 1.0)


def test_compute_coherence_score_mixed_and_edge_cases():
    eW = np.array([np.nan, 0.5, 0.05, 0.05, 0.25])
    eLam = np.array([0.05, np.nan, 0.10, 0.25, 0.05])
    elapsed = np.array([100.0, 100.0, 20.0, 120.0, 120.0])
    epsilon = 0.20
    horizon_hours = 80.0

    score, coherent, total = compute_coherence_score(eW, eLam, elapsed, epsilon, horizon_hours)
    # Eligible (elapsed >= horizon): indices 0,1,3,4; NaNs excluded
    assert coherent == 0
    assert total == 2  # indices 3 and 4 considered; both fail one side
    assert np.isclose(score, 0.0)


def test_compute_coherence_score_no_eligible_returns_nan_score():
    eW = np.array([0.5, 0.6])
    eLam = np.array([0.5, 0.6])
    elapsed = np.array([10.0, 20.0])  # both < horizon
    epsilon = 0.20
    horizon_hours = 80.0

    score, coherent, total = compute_coherence_score(eW, eLam, elapsed, epsilon, horizon_hours)
    assert total == 0
    assert coherent == 0
    assert np.isnan(score)
