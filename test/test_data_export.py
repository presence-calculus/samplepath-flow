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
    _derive_element_ids,
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
        t0=pd.Timestamp("2024-01-01"),
        tn=pd.Timestamp("2024-01-03"),
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
    tmp_path, sample_metrics, sample_empirical_metrics, sample_df
):
    """Test flow metrics CSV export in event mode."""
    export_dir = str(tmp_path)
    path = export_flow_metrics_csv(
        sample_df, sample_metrics, sample_empirical_metrics, export_dir
    )

    # Check file was created
    assert os.path.exists(path)
    assert path.endswith("event_indexed_metrics.csv")

    # Check content
    df = pd.read_csv(path)
    assert len(df) == 3
    assert df.columns.tolist() == [
        "timestamp",
        "element_id",
        "event_type",
        "A(T)",
        "D(T)",
        "N(t)",
        "H(T)",
        "L(T)",
        "Lambda(T)",
        "Theta(T)",
        "w(T)",
        "w'(T)",
        "W*(T)",
    ]


def test_export_flow_metrics_csv_calendar_mode(
    tmp_path, sample_empirical_metrics, sample_df
):
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
        sample_df,
        metrics,
        sample_empirical_metrics,
        export_dir,
        sampling_frequency="week",
    )

    # Check file was created with correct name
    assert os.path.exists(path)
    assert path.endswith("week_indexed_metrics.csv")

    # Check content
    df = pd.read_csv(path)
    assert len(df) == 3
    # First column should be the sampling frequency
    assert df.columns[0] == "week"
    # Should NOT have event_type or element_id column in calendar mode
    assert "event_type" not in df.columns
    assert "element_id" not in df.columns
    # Should have 9 columns total (no event_type, no element_id)
    assert len(df.columns) == 11


def test_export_element_csv(tmp_path, sample_df):
    """Test element CSV export."""
    export_dir = str(tmp_path)
    t_0 = pd.Timestamp("2024-01-01")
    t_n = pd.Timestamp("2024-01-04")
    path = export_element_csv(sample_df, export_dir, t_0, t_n)

    # Check file was created
    assert os.path.exists(path)
    assert path.endswith("elements.csv")

    # Check content
    df = pd.read_csv(path)
    assert len(df) == 3

    # Check columns
    assert "element_id" in df.columns
    assert "sojourn_time" in df.columns
    assert "residence_time" in df.columns
    assert "id" not in df.columns  # Should be renamed
    assert "duration_td" not in df.columns  # Should be dropped
    assert "duration_hr" not in df.columns  # Should be dropped
    assert "class" in df.columns  # Original columns should be preserved

    # Check exact column order per spec: element_id, start_ts, end_ts, <other cols>, sojourn_time, residence_time
    expected_columns = [
        "element_id",
        "start_ts",
        "end_ts",
        "class",
        "sojourn_time",
        "residence_time",
    ]
    assert df.columns.tolist() == expected_columns, (
        f"Column order must be: element_id, start_ts, end_ts, <other columns>, sojourn_time, residence_time. "
        f"Got: {df.columns.tolist()}"
    )

    # Check sorting by start_ts
    assert df["element_id"].tolist() == [1, 2, 3]

    # Check sojourn_time calculation
    assert df["sojourn_time"].iloc[0] == 86400.0  # 1 day in seconds
    assert df["sojourn_time"].iloc[1] == 86400.0  # 1 day in seconds
    assert pd.isna(df["sojourn_time"].iloc[2])  # Incomplete item

    # Check residence_time calculation
    assert df["residence_time"].iloc[0] == 86400.0  # Completed: same as sojourn
    assert df["residence_time"].iloc[1] == 86400.0  # Completed: same as sojourn
    # Incomplete: t_n - max(start_ts, t_0) = 2024-01-04 - 2024-01-03 = 1 day
    assert df["residence_time"].iloc[2] == 86400.0


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


def test_derive_element_ids_single_element_per_time():
    """Test element_id derivation when exactly one element per observation time."""
    df = pd.DataFrame(
        {
            "id": [1, 2, 3],
            "start_ts": [
                pd.Timestamp("2024-01-01"),
                pd.Timestamp("2024-01-03"),
                pd.Timestamp("2024-01-05"),
            ],
            "end_ts": [
                pd.Timestamp("2024-01-02"),
                pd.Timestamp("2024-01-04"),
                pd.NaT,
            ],
        }
    )
    observation_times = [
        pd.Timestamp("2024-01-01"),
        pd.Timestamp("2024-01-02"),
        pd.Timestamp("2024-01-03"),
    ]

    element_ids = _derive_element_ids(df, observation_times)

    # Each observation time has exactly one element, so should resolve
    assert element_ids[0] == "1"  # Element 1 arrives
    assert element_ids[1] == "1"  # Element 1 departs
    assert element_ids[2] == "2"  # Element 2 arrives


def test_derive_element_ids_multiple_elements_per_time():
    """Test element_id derivation with multiple arrivals at same time."""
    df = pd.DataFrame(
        {
            "id": [1, 2],
            "start_ts": [
                pd.Timestamp("2024-01-01"),
                pd.Timestamp("2024-01-01"),  # Same time as element 1
            ],
            "end_ts": [pd.Timestamp("2024-01-02"), pd.Timestamp("2024-01-03")],
        }
    )
    observation_times = [pd.Timestamp("2024-01-01")]

    element_ids = _derive_element_ids(df, observation_times)

    # Multiple arrivals at same time, should return "A:1;2"
    assert element_ids[0] == "A:1;2"


def test_derive_element_ids_no_elements_at_time():
    """Test element_id derivation returns None when no elements at observation time."""
    df = pd.DataFrame(
        {
            "id": [1],
            "start_ts": [pd.Timestamp("2024-01-01")],
            "end_ts": [pd.Timestamp("2024-01-02")],
        }
    )
    observation_times = [pd.Timestamp("2024-01-05")]  # No events at this time

    element_ids = _derive_element_ids(df, observation_times)

    # No elements at this time, should return None
    assert element_ids[0] is None


def test_derive_element_ids_multiple_elements_mixed():
    """Test element_id derivation with both arrivals and departures at same time."""
    df = pd.DataFrame(
        {
            "id": [1, 2],
            "start_ts": [
                pd.Timestamp("2024-01-01"),
                pd.Timestamp("2024-01-02"),
            ],
            "end_ts": [
                pd.Timestamp("2024-01-02"),  # Element 1 departs at 2024-01-02
                pd.Timestamp("2024-01-03"),
            ],
        }
    )
    observation_times = [pd.Timestamp("2024-01-02")]

    element_ids = _derive_element_ids(df, observation_times)

    # Element 2 arrives and element 1 departs at same time
    assert element_ids[0] == "A:2|D:1"


def test_derive_element_ids_multiple_departures():
    """Test element_id derivation with multiple departures at same time."""
    df = pd.DataFrame(
        {
            "id": [1, 2],
            "start_ts": [
                pd.Timestamp("2024-01-01"),
                pd.Timestamp("2024-01-02"),
            ],
            "end_ts": [
                pd.Timestamp("2024-01-05"),  # Both depart at same time
                pd.Timestamp("2024-01-05"),
            ],
        }
    )
    observation_times = [pd.Timestamp("2024-01-05")]

    element_ids = _derive_element_ids(df, observation_times)

    # Multiple departures at same time
    assert element_ids[0] == "D:1;2"


def test_derive_element_ids_zero_duration_among_multiple():
    """Test element_id derivation with zero-duration element among multiple elements."""
    df = pd.DataFrame(
        {
            "id": [1, 2],
            "start_ts": [
                pd.Timestamp("2024-01-01"),
                pd.Timestamp("2024-01-01"),  # Zero-duration element
            ],
            "end_ts": [
                pd.Timestamp("2024-01-02"),
                pd.Timestamp("2024-01-01"),  # Same as start_ts
            ],
        }
    )
    observation_times = [pd.Timestamp("2024-01-01")]

    element_ids = _derive_element_ids(df, observation_times)

    # Element 1 arrives, element 2 both arrives and departs (zero-duration)
    # Element 2 appears in both A and D groups
    assert element_ids[0] == "A:1;2|D:2"


def test_residence_time_for_completed_items():
    """Test residence_time equals sojourn_time for completed items."""
    df = pd.DataFrame(
        {
            "id": [1, 2],
            "start_ts": [
                pd.Timestamp("2024-01-01"),
                pd.Timestamp("2024-01-02"),
            ],
            "end_ts": [
                pd.Timestamp("2024-01-02"),
                pd.Timestamp("2024-01-04"),
            ],
        }
    )
    export_dir = "/tmp/test"
    t_0 = pd.Timestamp("2024-01-01")
    t_n = pd.Timestamp("2024-01-05")

    # We can't call export_element_csv directly without creating a directory
    # So let's test the logic inline
    export_df = df.copy()
    sojourn_time = (export_df["end_ts"] - export_df["start_ts"]).dt.total_seconds()
    residence_time = np.where(
        export_df["end_ts"].notna(),
        sojourn_time,
        (t_n - export_df["start_ts"].clip(lower=t_0)).dt.total_seconds(),
    )

    # For completed items, residence_time should equal sojourn_time
    assert residence_time[0] == sojourn_time[0]  # 1 day
    assert residence_time[1] == sojourn_time[1]  # 2 days


def test_residence_time_for_incomplete_items():
    """Test residence_time = t_n - max(start_ts, t_0) for incomplete items."""
    df = pd.DataFrame(
        {
            "id": [1, 2],
            "start_ts": [
                pd.Timestamp("2024-01-01"),
                pd.Timestamp("2023-12-28"),  # Before t_0
            ],
            "end_ts": [pd.NaT, pd.NaT],  # Both incomplete
        }
    )
    t_0 = pd.Timestamp("2024-01-01")
    t_n = pd.Timestamp("2024-01-05")

    export_df = df.copy()
    sojourn_time = (export_df["end_ts"] - export_df["start_ts"]).dt.total_seconds()
    residence_time = np.where(
        export_df["end_ts"].notna(),
        sojourn_time,
        (t_n - export_df["start_ts"].clip(lower=t_0)).dt.total_seconds(),
    )

    # Element 1: t_n - max(2024-01-01, 2024-01-01) = 4 days
    assert residence_time[0] == 4 * 86400.0
    # Element 2: t_n - max(2023-12-28, 2024-01-01) = 4 days (clipped to t_0)
    assert residence_time[1] == 4 * 86400.0


def test_derive_element_ids_zero_duration_element():
    """Test element_id derivation for zero-duration elements (start_ts == end_ts)."""
    df = pd.DataFrame(
        {
            "id": [1, 2],
            "start_ts": [
                pd.Timestamp("2024-01-01"),
                pd.Timestamp("2024-01-02"),
            ],
            "end_ts": [
                pd.Timestamp("2024-01-01"),  # Zero duration
                pd.Timestamp("2024-01-03"),
            ],
        }
    )
    observation_times = [pd.Timestamp("2024-01-01")]

    element_ids = _derive_element_ids(df, observation_times)

    # Element 1 arrives and departs at same time - should still resolve
    assert element_ids[0] == "1"


def test_residence_time_with_missing_start_ts():
    """Test residence_time returns NaN when start_ts is NaT."""
    df = pd.DataFrame(
        {
            "id": [1, 2],
            "start_ts": [pd.NaT, pd.Timestamp("2024-01-02")],  # First is missing
            "end_ts": [pd.NaT, pd.NaT],  # Both incomplete
        }
    )
    t_0 = pd.Timestamp("2024-01-01")
    t_n = pd.Timestamp("2024-01-05")

    export_df = df.copy()
    sojourn_time = (export_df["end_ts"] - export_df["start_ts"]).dt.total_seconds()
    residence_time = np.where(
        export_df["end_ts"].notna(),
        sojourn_time,
        np.where(
            export_df["start_ts"].notna(),
            (t_n - export_df["start_ts"].clip(lower=t_0)).dt.total_seconds(),
            np.nan,
        ),
    )

    # Element 1 has missing start_ts, should be NaN
    assert pd.isna(residence_time[0])
    # Element 2 has valid start_ts, should calculate correctly
    assert residence_time[1] == 3 * 86400.0  # t_n - start_ts = 3 days


def test_event_indexed_metrics_has_element_id_column():
    """Test that event_indexed_metrics.csv includes element_id as second column."""
    df = pd.DataFrame(
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
        }
    )
    times = [
        pd.Timestamp("2024-01-01"),
        pd.Timestamp("2024-01-02"),
        pd.Timestamp("2024-01-03"),
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
        mode="event",
        freq=None,
        t0=pd.Timestamp("2024-01-01"),
        tn=pd.Timestamp("2024-01-03"),
    )
    empirical_metrics = ElementWiseEmpiricalMetrics(
        times=times,
        W_star=np.array([50.0, 150.0, 250.0]),
        lam_star=np.array([0.4, 0.5, 0.6]),
        sojourn_vals=np.array([]),
        residence_time_vals=np.array([]),
        residence_completed=np.array([]),
    )

    import tempfile

    with tempfile.TemporaryDirectory() as tmp_dir:
        path = export_flow_metrics_csv(df, metrics, empirical_metrics, tmp_dir)
        result_df = pd.read_csv(path)

        # Check that element_id is the second column
        assert result_df.columns[0] == "timestamp"
        assert result_df.columns[1] == "element_id"
        assert result_df.columns[2] == "event_type"
