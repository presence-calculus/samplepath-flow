# -*- coding: utf-8 -*-
# Copyright (c) 2025 Krishna Kumar
# SPDX-License-Identifier: MIT
# test/spath/metrics/test_finite_window_flow_metrics.py
# test/spath/metrics/test_finite_window_flow_metrics.py
import numpy as np
import pandas as pd
import pytest

from spath.metrics import compute_finite_window_flow_metrics

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
    elapsed = (tn - t0).total_seconds() / 3600.0
    assert np.isclose(res.L[-1], res.A[-1] / elapsed)

def test_event_mode_final_identity_w_equals_A_over_arrivals(simple_events):
    res = compute_finite_window_flow_metrics(simple_events, freq=None)
    assert np.isclose(res.w[-1], res.A[-1] / res.Arrivals[-1])


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Calendar mode (fixed frequencies only)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def test_calendar_mode_sets_mode_calendar(overlap_events):
    res = compute_finite_window_flow_metrics(overlap_events, freq="day")
    assert res.mode == "calendar"

def test_calendar_mode_rejects_week_nonfixed(overlap_events):
    with pytest.raises(ValueError):
        compute_finite_window_flow_metrics(overlap_events, freq="week")

def test_calendar_mode_rejects_month_nonfixed(overlap_events):
    with pytest.raises(ValueError):
        compute_finite_window_flow_metrics(overlap_events, freq="month")

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
    assert all(getattr(res, name).size == 0
               for name in ["L", "Lambda", "w", "N", "A", "Arrivals", "Departures"])
