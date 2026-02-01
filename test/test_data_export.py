# -*- coding: utf-8 -*-
# Copyright (c) 2025 Krishna Kumar
# SPDX-License-Identifier: MIT
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from samplepath.data_export import (
    _derive_event_types,
    export_data,
    export_element_csv,
    export_flow_metrics_csv,
)
from samplepath.metrics import (
    ElementWiseEmpiricalMetrics,
    FlowMetricsResult,
)


@pytest.fixture
def sample_metrics():
    """Fixture providing sample FlowMetricsResult for testing."""
    times = [
        pd.Timestamp("2024-01-01"),
        pd.Timestamp("2024-01-02"),
        pd.Timestamp("2024-01-03"),
    ]
    return FlowMetricsResult(
        events=[],
        times=times,
        L=np.array([1.0, 2.0, 3.0]),
        Lambda=np.array([0.5, 0.6, 0.7]),
        w=np.array([100.0, 200.0, 300.0]),
        w_prime=np.array([110.0, 210.0, 310.0]),
        N=np.array([1, 2, 3]),
        H=np.array([100.0, 200.0, 300.0]),
        Arrivals=np.array([1, 2, 3]),
        Departures=np.array([0, 1, 2]),
        Theta=np.array([0.0, 0.5, 0.6]),
        mode="event",
        freq=None,
    )


@pytest.fixture
def sample_empirical_metrics():
    """Fixture providing sample ElementWiseEmpiricalMetrics for testing."""
    times = [
        pd.Timestamp("2024-01-01"),
        pd.Timestamp("2024-01-02"),
        pd.Timestamp("2024-01-03"),
    ]
    return ElementWiseEmpiricalMetrics(
        times=times,
        W_star=np.array([50.0, 150.0, 250.0]),
        lam_star=np.array([0.4, 0.5, 0.6]),
        sojourn_vals=np.array([]),
        residence_time_vals=np.array([]),
        residence_completed=np.array([]),
    )


@pytest.fixture
def sample_df():
    """Fixture providing sample DataFrame for element export testing."""
    return pd.DataFrame(
        {
            "id": [1, 2, 3],
            "start_ts": [
                pd.Timestamp("2024-01-01"),
                pd.Timestamp("2024-01-02"),
                pd.Timestamp("2024-01-03"),
            ],
            "end_ts": [
                pd.Timestamp("2024-01-02"),
                pd.Timestamp("2024-01-03"),
                pd.NaT,
            ],
            "duration_td": [
                pd.Timedelta(days=1),
                pd.Timedelta(days=1),
                pd.NaT,
            ],
            "duration_hr": [24.0, 24.0, np.nan],
            "class": ["A", "B", "A"],
        }
    )


def test_derive_event_types_arrival_only():
    """Test event type derivation with arrival-only events."""
    metrics = SimpleNamespace(
        Arrivals=np.array([0, 1, 2]), Departures=np.array([0, 0, 0])
    )
    event_types = _derive_event_types(metrics)
    assert event_types == ["-", "A", "A"]


def test_derive_event_types_departure_only():
    """Test event type derivation with departure-only events."""
    metrics = SimpleNamespace(
        Arrivals=np.array([0, 0, 0]), Departures=np.array([0, 1, 2])
    )
    event_types = _derive_event_types(metrics)
    assert event_types == ["-", "D", "D"]


def test_derive_event_types_mixed():
    """Test event type derivation with mixed arrival/departure events."""
    metrics = SimpleNamespace(
        Arrivals=np.array([0, 1, 2]), Departures=np.array([0, 1, 1])
    )
    event_types = _derive_event_types(metrics)
    assert event_types == ["-", "A/D", "A"]


def test_derive_event_types_baseline():
    """Test event type derivation with no events."""
    metrics = SimpleNamespace(
        Arrivals=np.array([0, 0, 0]), Departures=np.array([0, 0, 0])
    )
    event_types = _derive_event_types(metrics)
    assert event_types == ["-", "-", "-"]


def test_derive_event_types_first_observation_arrival():
    """Test event type when first observation has an arrival."""
    metrics = SimpleNamespace(
        Arrivals=np.array([1, 2, 3]), Departures=np.array([0, 0, 1])
    )
    event_types = _derive_event_types(metrics)
    assert event_types[0] == "A", "First observation with arrival should be 'A'"
    assert event_types == ["A", "A", "A/D"]


def test_derive_event_types_first_observation_departure():
    """Test event type when first observation has a departure."""
    metrics = SimpleNamespace(
        Arrivals=np.array([0, 1, 2]), Departures=np.array([1, 1, 2])
    )
    event_types = _derive_event_types(metrics)
    assert event_types[0] == "D", "First observation with departure should be 'D'"
    assert event_types == ["D", "A", "A/D"]


def test_derive_event_types_first_observation_both():
    """Test event type when first observation has both arrival and departure."""
    metrics = SimpleNamespace(
        Arrivals=np.array([2, 3, 4]), Departures=np.array([1, 1, 2])
    )
    event_types = _derive_event_types(metrics)
    assert event_types[0] == "A/D", "First observation with both should be 'A/D'"
    assert event_types == ["A/D", "A", "A/D"]


def test_export_flow_metrics_csv_event_mode(
    tmp_path, sample_metrics, sample_empirical_metrics
):
    """Test flow metrics CSV export in event mode."""
    export_dir = str(tmp_path)
    path = export_flow_metrics_csv(sample_metrics, sample_empirical_metrics, export_dir)

    # Check file was created
    assert os.path.exists(path)
    assert path.endswith("event_indexed_metrics.csv")

    # Check content
    df = pd.read_csv(path)
    assert len(df) == 3
    assert df.columns.tolist() == [
        "timestamp",
        "event_type",
        "N(T)",
        "H(T)",
        "L(T)",
        "Lambda(T)",
        "Theta(T)",
        "w(T)",
        "w'(T)",
        "W*(T)",
    ]


def test_export_flow_metrics_csv_calendar_mode(tmp_path, sample_empirical_metrics):
    """Test flow metrics CSV export in calendar mode."""
    times = [
        pd.Timestamp("2024-01-01"),
        pd.Timestamp("2024-01-08"),
        pd.Timestamp("2024-01-15"),
    ]
    metrics = FlowMetricsResult(
        events=[],
        times=times,
        L=np.array([1.0, 2.0, 3.0]),
        Lambda=np.array([0.5, 0.6, 0.7]),
        w=np.array([100.0, 200.0, 300.0]),
        w_prime=np.array([110.0, 210.0, 310.0]),
        N=np.array([1, 2, 3]),
        H=np.array([100.0, 200.0, 300.0]),
        Arrivals=np.array([1, 2, 3]),
        Departures=np.array([0, 1, 2]),
        Theta=np.array([0.0, 0.5, 0.6]),
        mode="calendar",
        freq="week",
    )

    export_dir = str(tmp_path)
    path = export_flow_metrics_csv(
        metrics, sample_empirical_metrics, export_dir, sampling_frequency="week"
    )

    # Check file was created with correct name
    assert os.path.exists(path)
    assert path.endswith("week_indexed_metrics.csv")

    # Check content
    df = pd.read_csv(path)
    assert len(df) == 3
    # First column should be the sampling frequency
    assert df.columns[0] == "week"
    # Should NOT have event_type column
    assert "event_type" not in df.columns
    # Should have 9 columns total (no event_type)
    assert len(df.columns) == 9


def test_export_element_csv(tmp_path, sample_df):
    """Test element CSV export."""
    export_dir = str(tmp_path)
    path = export_element_csv(sample_df, export_dir)

    # Check file was created
    assert os.path.exists(path)
    assert path.endswith("elements.csv")

    # Check content
    df = pd.read_csv(path)
    assert len(df) == 3

    # Check columns
    assert "element_id" in df.columns
    assert "sojourn_time" in df.columns
    assert "id" not in df.columns  # Should be renamed
    assert "duration_td" not in df.columns  # Should be dropped
    assert "duration_hr" not in df.columns  # Should be dropped
    assert "class" in df.columns  # Original columns should be preserved

    # Check exact column order per spec: element_id, start_ts, end_ts, <other cols>, sojourn_time
    expected_columns = ["element_id", "start_ts", "end_ts", "class", "sojourn_time"]
    assert df.columns.tolist() == expected_columns, (
        f"Column order must be: element_id, start_ts, end_ts, <other columns>, sojourn_time. "
        f"Got: {df.columns.tolist()}"
    )

    # Check sorting by start_ts
    assert df["element_id"].tolist() == [1, 2, 3]

    # Check sojourn_time calculation
    assert df["sojourn_time"].iloc[0] == 86400.0  # 1 day in seconds
    assert df["sojourn_time"].iloc[1] == 86400.0  # 1 day in seconds
    assert pd.isna(df["sojourn_time"].iloc[2])  # Incomplete item


def test_export_data(tmp_path, sample_df, sample_metrics, sample_empirical_metrics):
    """Test export_data creates both files."""
    export_dir = str(tmp_path)
    paths = export_data(sample_df, sample_metrics, sample_empirical_metrics, export_dir)

    # Check both files were created
    assert len(paths) == 2
    assert all(os.path.exists(p) for p in paths)

    # Check file names
    assert any("event_indexed_metrics.csv" in p for p in paths)
    assert any("elements.csv" in p for p in paths)
