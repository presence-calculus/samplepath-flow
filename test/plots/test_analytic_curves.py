# -*- coding: utf-8 -*-
# Copyright (c) 2026 Krishna Kumar
# SPDX-License-Identifier: MIT
import numpy as np
import pandas as pd

from samplepath.plots.analytic_curves import (
    build_L_curve,
    build_lambda_curve,
    build_theta_curve,
    build_w_curve,
    build_w_prime_curve,
)


def _t(s: str) -> pd.Timestamp:
    return pd.Timestamp(s)


def test_build_L_curve_matches_interval_formula_at_sample_point():
    times = [
        _t("2024-01-01 00:00:00"),
        _t("2024-01-01 00:00:01"),
        _t("2024-01-01 00:00:03"),
    ]
    H = np.array([0.0, 1.0, 3.0])
    N = np.array([1.0, 1.0, 0.0])

    curve_times, curve_vals = build_L_curve(times, H, N, samples_per_interval=2)

    idx = curve_times.index(_t("2024-01-01 00:00:02"))
    assert np.isclose(curve_vals[idx], 1.0)


def test_build_L_curve_drops_t0_point_for_divide_by_t_stability():
    times = [_t("2024-01-01 00:00:00"), _t("2024-01-01 00:00:01")]
    H = np.array([0.0, 1.0])
    N = np.array([1.0, 1.0])

    curve_times, _ = build_L_curve(times, H, N, samples_per_interval=3)

    assert curve_times[0] > times[0]


def test_build_L_curve_returns_non_empty_for_valid_interval():
    times = [_t("2024-01-01 00:00:00"), _t("2024-01-01 00:00:02")]
    H = np.array([0.0, 2.0])
    N = np.array([1.0, 1.0])

    curve_times, _ = build_L_curve(times, H, N, samples_per_interval=4)

    assert len(curve_times) > 0


def test_build_lambda_curve_matches_hyperbola_inside_interval():
    times = [_t("2024-01-01 00:00:00"), _t("2024-01-01 00:00:02")]
    arrivals = np.array([2.0, 2.0])

    curve_times, curve_vals, _ = build_lambda_curve(
        times,
        arrivals,
        samples_per_interval=2,
    )

    idx = curve_times.index(_t("2024-01-01 00:00:01"))
    assert np.isclose(curve_vals[idx], 2.0)


def test_build_lambda_curve_reports_jump_when_arrivals_change():
    times = [
        _t("2024-01-01 00:00:00"),
        _t("2024-01-01 00:00:02"),
        _t("2024-01-01 00:00:04"),
    ]
    arrivals = np.array([1.0, 2.0, 2.0])

    _, _, jumps = build_lambda_curve(times, arrivals, samples_per_interval=2)

    assert jumps == [(_t("2024-01-01 00:00:02"), 0.5, 1.0)]


def test_build_lambda_curve_drops_t0_samples_for_divide_by_t_stability():
    times = [_t("2024-01-01 00:00:00"), _t("2024-01-01 00:00:02")]
    arrivals = np.array([1.0, 1.0])

    curve_times, _, _ = build_lambda_curve(times, arrivals, samples_per_interval=3)

    assert curve_times[0] > times[0]


def test_build_theta_curve_matches_hyperbola_inside_interval():
    times = [_t("2024-01-01 00:00:00"), _t("2024-01-01 00:00:02")]
    departures = np.array([2.0, 2.0])

    curve_times, curve_vals, _ = build_theta_curve(
        times,
        departures,
        samples_per_interval=2,
    )

    idx = curve_times.index(_t("2024-01-01 00:00:01"))
    assert np.isclose(curve_vals[idx], 2.0)


def test_build_theta_curve_reports_jump_when_departures_change():
    times = [
        _t("2024-01-01 00:00:00"),
        _t("2024-01-01 00:00:02"),
        _t("2024-01-01 00:00:04"),
    ]
    departures = np.array([1.0, 2.0, 2.0])

    _, _, jumps = build_theta_curve(times, departures, samples_per_interval=2)

    assert jumps == [(_t("2024-01-01 00:00:02"), 0.5, 1.0)]


def test_build_theta_curve_drops_t0_samples_for_divide_by_t_stability():
    times = [_t("2024-01-01 00:00:00"), _t("2024-01-01 00:00:02")]
    departures = np.array([1.0, 1.0])

    curve_times, _, _ = build_theta_curve(times, departures, samples_per_interval=3)

    assert curve_times[0] > times[0]


def test_build_w_curve_matches_interval_formula_at_sample_point():
    times = [_t("2024-01-01 00:00:00"), _t("2024-01-01 00:00:02")]
    H_vals = np.array([0.0, 2.0])
    N_vals = np.array([1.0, 1.0])
    arrivals = np.array([2.0, 2.0])

    curve_times, curve_vals, _ = build_w_curve(
        times,
        H_vals,
        N_vals,
        arrivals,
        samples_per_interval=2,
    )

    idx = curve_times.index(_t("2024-01-01 00:00:01"))
    assert np.isclose(curve_vals[idx], 0.5)


def test_build_w_curve_reports_jump_when_arrivals_change():
    times = [
        _t("2024-01-01 00:00:00"),
        _t("2024-01-01 00:00:02"),
        _t("2024-01-01 00:00:04"),
    ]
    H_vals = np.array([0.0, 2.0, 4.0])
    N_vals = np.array([1.0, 1.0, 1.0])
    arrivals = np.array([1.0, 2.0, 2.0])

    _, _, jumps = build_w_curve(times, H_vals, N_vals, arrivals, samples_per_interval=2)

    assert jumps == [(_t("2024-01-01 00:00:02"), 2.0, 1.0)]


def test_build_w_curve_skips_undefined_intervals_when_arrivals_are_zero():
    times = [
        _t("2024-01-01 00:00:00"),
        _t("2024-01-01 00:00:02"),
        _t("2024-01-01 00:00:04"),
    ]
    H_vals = np.array([0.0, 2.0, 4.0])
    N_vals = np.array([1.0, 1.0, 1.0])
    arrivals = np.array([0.0, 1.0, 1.0])

    curve_times, _, _ = build_w_curve(
        times,
        H_vals,
        N_vals,
        arrivals,
        samples_per_interval=2,
    )

    assert curve_times[0] >= times[1]


def test_build_w_prime_curve_matches_interval_formula_at_sample_point():
    times = [_t("2024-01-01 00:00:00"), _t("2024-01-01 00:00:02")]
    H_vals = np.array([0.0, 2.0])
    N_vals = np.array([1.0, 1.0])
    departures = np.array([2.0, 2.0])

    curve_times, curve_vals, _ = build_w_prime_curve(
        times,
        H_vals,
        N_vals,
        departures,
        samples_per_interval=2,
    )

    idx = curve_times.index(_t("2024-01-01 00:00:01"))
    assert np.isclose(curve_vals[idx], 0.5)


def test_build_w_prime_curve_reports_jump_when_departures_change():
    times = [
        _t("2024-01-01 00:00:00"),
        _t("2024-01-01 00:00:02"),
        _t("2024-01-01 00:00:04"),
    ]
    H_vals = np.array([0.0, 2.0, 4.0])
    N_vals = np.array([1.0, 1.0, 1.0])
    departures = np.array([1.0, 2.0, 2.0])

    _, _, jumps = build_w_prime_curve(
        times,
        H_vals,
        N_vals,
        departures,
        samples_per_interval=2,
    )

    assert jumps == [(_t("2024-01-01 00:00:02"), 2.0, 1.0)]


def test_build_w_prime_curve_skips_undefined_intervals_when_departures_are_zero():
    times = [
        _t("2024-01-01 00:00:00"),
        _t("2024-01-01 00:00:02"),
        _t("2024-01-01 00:00:04"),
    ]
    H_vals = np.array([0.0, 2.0, 4.0])
    N_vals = np.array([1.0, 1.0, 1.0])
    departures = np.array([0.0, 1.0, 1.0])

    curve_times, _, _ = build_w_prime_curve(
        times,
        H_vals,
        N_vals,
        departures,
        samples_per_interval=2,
    )

    assert curve_times[0] >= times[1]
