# -*- coding: utf-8 -*-
# Copyright (c) 2025 Krishna Kumar
# SPDX-License-Identifier: MIT
# test/samplepath/metrics/test_finite_window_flow_metrics.py
# test/samplepath/metrics/test_finite_window_flow_metrics.py
import numpy as np
import pandas as pd
import pytest

from samplepath.metrics import compute_finite_window_flow_metrics


def _t(s: str) -> pd.Timestamp:
    return pd.Timestamp(s)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Fixtures
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


@pytest.fixture
def simple_events():
    t0 = _t("2024-01-01 00:00")
    t1 = _t("2024-01-01 02:00")
    return [(t0, +1, 1), (t1, -1, 0)]


@pytest.fixture
def overlap_events():
    t0 = _t("2024-01-01 00:00")
    t1 = _t("2024-01-01 01:00")
    t2 = _t("2024-01-01 03:30")
    t3 = _t("2024-01-01 05:00")
    return [(t0, +1, 1), (t1, +1, 1), (t2, -1, 0), (t3, -1, 0)]


@pytest.fixture
def carry_in_events():
    pre = _t("2024-01-01 00:00")
    end = _t("2024-01-02 00:00")
    return [(pre, +1, 1), (end, -1, 0)]


@pytest.fixture
def single_item_one_hour_events():
    t0 = _t("2024-01-01 00:00")
    t1 = _t("2024-01-01 01:00")
    return [(t0, +1, 1), (t1, -1, 0)]


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Event mode (freq=None)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def test_event_mode_sets_mode_event(simple_events):
    res = compute_finite_window_flow_metrics(simple_events, freq=None)
    assert res.mode == "event"


def test_event_mode_times_include_first_and_last_event(simple_events):
    res = compute_finite_window_flow_metrics(simple_events, freq=None)
    assert res.times == sorted([e[0] for e in simple_events])


def test_event_mode_final_identity_L_equals_A_over_elapsed(simple_events):
    res = compute_finite_window_flow_metrics(simple_events, freq=None)
    t0, tn = res.times[0], res.times[-1]
    elapsed = (tn - t0).total_seconds()
    assert np.isclose(res.L[-1], res.H[-1] / elapsed)


def test_event_mode_final_identity_w_equals_A_over_arrivals(simple_events):
    res = compute_finite_window_flow_metrics(simple_events, freq=None)
    assert np.isclose(res.w[-1], res.H[-1] / res.Arrivals[-1])


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Calendar mode
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def test_calendar_mode_sets_mode_calendar(overlap_events):
    res = compute_finite_window_flow_metrics(overlap_events, freq="day")
    assert res.mode == "calendar"


def test_calendar_mode_week_sets_mode_and_freq():
    events = [
        (_t("2024-01-03 10:00"), +1, 1),
        (_t("2024-01-17 10:00"), -1, 0),
    ]
    res = compute_finite_window_flow_metrics(events, freq="week")
    assert res.mode == "calendar"
    assert res.freq == "W-SUN"


def test_calendar_mode_month_sets_mode_and_freq():
    events = [
        (_t("2024-01-15 10:00"), +1, 1),
        (_t("2024-03-15 10:00"), -1, 0),
    ]
    res = compute_finite_window_flow_metrics(events, freq="month")
    assert res.mode == "calendar"
    assert res.freq == "MS"


def test_calendar_mode_week_boundaries_are_aligned():
    events = [
        (_t("2024-01-03 10:00"), +1, 1),
        (_t("2024-01-17 10:00"), -1, 0),
    ]
    res = compute_finite_window_flow_metrics(events, freq="week")
    # W-SUN boundaries fall on Sundays (weekday 6)
    assert all(t.weekday() == 6 for t in res.times)


def test_calendar_mode_month_boundaries_are_month_starts():
    events = [
        (_t("2024-01-15 10:00"), +1, 1),
        (_t("2024-03-15 10:00"), -1, 0),
    ]
    res = compute_finite_window_flow_metrics(events, freq="month")
    assert all(t.day == 1 for t in res.times)


@pytest.mark.parametrize("freq", ["day", "week", "month", "quarter", "year"])
def test_calendar_mode_accepts_all_human_frequencies(freq):
    events = [
        (_t("2024-01-03 10:00"), +1, 1),
        (_t("2025-03-15 10:00"), -1, 0),
    ]
    res = compute_finite_window_flow_metrics(events, freq=freq)
    assert res.mode == "calendar"
    assert len(res.times) > 0


def test_calendar_mode_week_anchor_wed():
    events = [
        (_t("2024-01-03 10:00"), +1, 1),
        (_t("2024-01-24 10:00"), -1, 0),
    ]
    res = compute_finite_window_flow_metrics(events, freq="week", week_anchor="WED")
    # W-WED boundaries fall on Wednesdays (weekday 2)
    assert all(t.weekday() == 2 for t in res.times)


def test_calendar_mode_quarter_anchor_apr():
    events = [
        (_t("2024-01-15 10:00"), +1, 1),
        (_t("2025-01-15 10:00"), -1, 0),
    ]
    res = compute_finite_window_flow_metrics(
        events, freq="quarter", quarter_anchor="APR"
    )
    # QS-APR boundaries fall on Apr, Jul, Oct, Jan
    assert all(t.month in (1, 4, 7, 10) for t in res.times)
    assert all(t.day == 1 for t in res.times)


def test_calendar_mode_year_anchor_jul():
    events = [
        (_t("2023-01-15 10:00"), +1, 1),
        (_t("2025-06-15 10:00"), -1, 0),
    ]
    res = compute_finite_window_flow_metrics(events, freq="year", year_anchor="JUL")
    # YS-JUL boundaries fall on July 1st
    assert all(t.month == 7 and t.day == 1 for t in res.times)


def test_calendar_mode_times_are_midnight_boundaries(overlap_events):
    first = overlap_events[0][0].normalize()
    last = overlap_events[-1][0].normalize()
    res = compute_finite_window_flow_metrics(
        overlap_events, freq="day", start=first, end=last
    )
    assert all(t == t.normalize() for t in res.times)


def test_calendar_mode_carry_in_reflects_in_N0(carry_in_events):
    res = compute_finite_window_flow_metrics(
        carry_in_events,
        freq="day",
        start=_t("2024-01-01 12:00"),
        end=_t("2024-01-02 12:00"),
    )
    assert res.N[0] > 0  # implementation counts carry-in WIP; arrivals not zeroed


def test_calendar_mode_include_next_boundary_adds_one_boundary(overlap_events):
    first = overlap_events[0][0].normalize()
    last = overlap_events[-1][0].normalize()
    res_no = compute_finite_window_flow_metrics(
        overlap_events, freq="day", start=first, end=last, include_next_boundary=False
    )
    res_yes = compute_finite_window_flow_metrics(
        overlap_events, freq="day", start=first, end=last, include_next_boundary=True
    )
    assert len(res_yes.times) == len(res_no.times) + 1


def test_calendar_mode_t0_is_first_boundary(overlap_events):
    first = overlap_events[0][0].normalize()
    res = compute_finite_window_flow_metrics(overlap_events, freq="day", start=first)
    assert res.t0 == res.times[0]


def test_calendar_mode_tn_is_last_boundary(overlap_events):
    last = overlap_events[-1][0].normalize()
    res = compute_finite_window_flow_metrics(overlap_events, freq="day", end=last)
    assert res.tn == res.times[-1]


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Empty inputs
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def test_empty_events_returns_empty_times():
    res = compute_finite_window_flow_metrics([], freq=None)
    assert res.times == []


def test_empty_events_returns_empty_arrays():
    res = compute_finite_window_flow_metrics([], freq="day")
    assert all(
        getattr(res, name).size == 0
        for name in [
            "L",
            "Lambda",
            "Theta",
            "w",
            "w_prime",
            "N",
            "H",
            "Arrivals",
            "Departures",
        ]
    )


def test_w_prime_matches_presence_mass_per_departure():
    t0 = pd.Timestamp("2024-01-01 00:00:00")
    t1 = pd.Timestamp("2024-01-01 01:00:00")
    t2 = pd.Timestamp("2024-01-01 02:00:00")
    events = [
        (t0, 1, 1),  # first arrival
        (t1, 1, 1),  # second arrival
        (t2, -2, 0),  # two departures
    ]
    res = compute_finite_window_flow_metrics(events, freq=None)
    assert res.w_prime[-1] == 5400.0


@pytest.mark.parametrize(
    ("events", "expected_theta"),
    [
        (
            [
                (_t("2024-01-01 00:00"), 1, 1),
                (_t("2024-01-01 01:00"), -1, 0),
            ],
            1 / 3600,
        ),
        (
            [
                (_t("2024-01-01 00:00"), 1, 1),
                (_t("2024-01-01 01:00"), 1, 1),
                (_t("2024-01-01 03:00"), -2, 0),
            ],
            2 / (3 * 3600),
        ),
    ],
)
def test_theta_matches_departures_per_elapsed_second(events, expected_theta):
    res = compute_finite_window_flow_metrics(events, freq=None)
    assert np.isclose(res.Theta[-1], expected_theta)


def test_lambda_matches_arrivals_per_elapsed_second(single_item_one_hour_events):
    res = compute_finite_window_flow_metrics(single_item_one_hour_events, freq=None)
    assert np.isclose(res.Lambda[-1], 1.0 / 3600.0)


def test_lambda_scaled_to_per_hour_matches_expected(single_item_one_hour_events):
    res = compute_finite_window_flow_metrics(single_item_one_hour_events, freq=None)
    assert np.isclose(res.Lambda[-1] * 3600.0, 1.0)


def test_w_matches_seconds_per_arrival(single_item_one_hour_events):
    res = compute_finite_window_flow_metrics(single_item_one_hour_events, freq=None)
    assert np.isclose(res.w[-1], 3600.0)


def test_lambda_w_scaled_product_matches_L(single_item_one_hour_events):
    res = compute_finite_window_flow_metrics(single_item_one_hour_events, freq=None)
    lambda_per_hour = res.Lambda[-1] * 3600.0
    w_hours = res.w[-1] / 3600.0
    assert np.isclose(lambda_per_hour * w_hours, res.L[-1])
