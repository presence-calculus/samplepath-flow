# -*- coding: utf-8 -*-
# Copyright (c) 2025 Krishna Kumar
# SPDX-License-Identifier: MIT

"""
Data export utilities for flow metrics and element data.
"""
from __future__ import annotations

import os
from typing import List

import numpy as np
import pandas as pd

from .metrics import ElementWiseEmpiricalMetrics, FlowMetricsResult


def _derive_element_ids(
    df: pd.DataFrame, observation_times: List[pd.Timestamp]
) -> List[str | None]:
    """
    Attempt to derive element IDs for each observation time.

    For each observation time, checks if exactly one element arrived or departed
    at that time. If so, returns that element's ID. If multiple elements are
    present, returns a formatted string showing all arrivals and departures.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing element data with id/element_id, start_ts, end_ts.
    observation_times : List[pd.Timestamp]
        Observation times to map to element IDs.

    Returns
    -------
    List[str | None]
        Element ID for each observation time, formatted as:
        - Single element: "1"
        - Multiple elements: "A:1;2|D:3;4" (arrivals before departures)
        - No elements: None
    """
    element_ids = []

    # Determine the ID column name
    id_col = "element_id" if "element_id" in df.columns else "id"

    for obs_time in observation_times:
        # Find elements that arrived at this time
        arrivals = df[df["start_ts"] == obs_time]
        # Find elements that departed at this time
        departures = df[df["end_ts"] == obs_time]

        # Combine arrivals and departures
        events_at_time = pd.concat([arrivals, departures])

        # Deduplicate by element ID (handles zero-duration elements)
        unique_elements = events_at_time[id_col].drop_duplicates()

        # If exactly one unique element, use its ID
        if len(unique_elements) == 1:
            element_ids.append(str(unique_elements.iloc[0]))
        elif len(unique_elements) > 1:
            # Multiple elements - format as A:id;id|D:id;id
            arrival_ids = sorted(arrivals[id_col].astype(str).tolist())
            departure_ids = sorted(departures[id_col].astype(str).tolist())

            parts = []
            if arrival_ids:
                parts.append(f"A:{';'.join(arrival_ids)}")
            if departure_ids:
                parts.append(f"D:{';'.join(departure_ids)}")

            element_ids.append("|".join(parts))
        else:
            # No elements at this time
            element_ids.append(None)

    return element_ids


def _derive_event_types(metrics: FlowMetricsResult) -> List[str]:
    """
    Derive event types from differences in cumulative arrivals/departures.

    Parameters
    ----------
    metrics : FlowMetricsResult
        Flow metrics result containing cumulative Arrivals and Departures arrays.

    Returns
    -------
    List[str]
        Event type for each observation: "A" (arrival), "D" (departure),
        "A/D" (both), or "-" (neither).
    """
    arrivals = metrics.Arrivals
    departures = metrics.Departures

    event_types = []
    for i in range(len(arrivals)):
        if i == 0:
            # First observation - check if there are events at t0
            arr_delta = arrivals[i]
            dep_delta = departures[i]
            if arr_delta > 0 and dep_delta > 0:
                event_types.append("A/D")
            elif arr_delta > 0:
                event_types.append("A")
            elif dep_delta > 0:
                event_types.append("D")
            else:
                event_types.append("-")
        else:
            arr_delta = arrivals[i] - arrivals[i - 1]
            dep_delta = departures[i] - departures[i - 1]

            if arr_delta > 0 and dep_delta > 0:
                event_types.append("A/D")
            elif arr_delta > 0:
                event_types.append("A")
            elif dep_delta > 0:
                event_types.append("D")
            else:
                event_types.append("-")

    return event_types


def export_flow_metrics_csv(
    element_df: pd.DataFrame,
    metrics: FlowMetricsResult,
    empirical_metrics: ElementWiseEmpiricalMetrics,
    export_dir: str,
    sampling_frequency: str | None = None,
) -> str:
    """
    Export flow metrics to CSV.

    Parameters
    ----------
    element_df : pd.DataFrame
        DataFrame containing element data (used for element_id derivation).
    metrics : FlowMetricsResult
        Flow metrics result containing times and metric arrays.
    empirical_metrics : ElementWiseEmpiricalMetrics
        Empirical metrics containing W_star values.
    export_dir : str
        Directory where the CSV will be written.
    sampling_frequency : str | None
        Sampling frequency string for calendar mode, or None for event mode.

    Returns
    -------
    str
        Path to the created CSV file.
    """
    # Build DataFrame from metrics
    df = pd.DataFrame(
        {
            "time": metrics.times,
            "Arrivals": metrics.Arrivals,
            "Departures": metrics.Departures,
            "N": metrics.N,
            "H": metrics.H,
            "L": metrics.L,
            "Lambda": metrics.Lambda,
            "Theta": metrics.Theta,
            "w": metrics.w,
            "w_prime": metrics.w_prime,
            "W_star": empirical_metrics.W_star,
        }
    )

    # Determine mode and filename
    if sampling_frequency is None:
        # Event mode
        filename = "event_indexed_metrics.csv"
        # Add element_id as second column
        element_ids = _derive_element_ids(element_df, metrics.times)
        df.insert(1, "element_id", element_ids)
        # Add event_type as third column
        event_types = _derive_event_types(metrics)
        df.insert(2, "event_type", event_types)
        df = df.rename(
            columns={
                "time": "timestamp",
                "Arrivals": "A(T)",
                "Departures": "D(T)",
                "N": "N(t)",
                "H": "H(T)",
                "L": "L(T)",
                "Lambda": "Lambda(T)",
                "Theta": "Theta(T)",
                "w": "w(T)",
                "w_prime": "w'(T)",
                "W_star": "W*(T)",
            }
        )
    else:
        # Calendar mode
        filename = f"{sampling_frequency}_indexed_metrics.csv"
        df = df.rename(
            columns={
                "time": sampling_frequency,
                "Arrivals": "A(T)",
                "Departures": "D(T)",
                "N": "N(T)",
                "H": "H(T)",
                "L": "L(T)",
                "Lambda": "Lambda(T)",
                "Theta": "Theta(T)",
                "w": "w(T)",
                "w_prime": "w'(T)",
                "W_star": "W*(T)",
            }
        )

    # Write CSV
    path = os.path.join(export_dir, filename)
    df.to_csv(path, index=False)
    return path


def export_element_csv(
    df: pd.DataFrame, export_dir: str, t_0: pd.Timestamp, t_n: pd.Timestamp
) -> str:
    """
    Export element data to CSV.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing element data with id, start_ts, end_ts columns.
    export_dir : str
        Directory where the CSV will be written.
    t_0 : pd.Timestamp
        Start of observation window.
    t_n : pd.Timestamp
        End of observation window.

    Returns
    -------
    str
        Path to the created CSV file.
    """
    # Make a copy to avoid modifying the original
    export_df = df.copy()

    # Rename id to element_id
    if "id" in export_df.columns:
        export_df = export_df.rename(columns={"id": "element_id"})

    # Add sojourn_time column
    if "end_ts" in export_df.columns and "start_ts" in export_df.columns:
        sojourn_time = (export_df["end_ts"] - export_df["start_ts"]).dt.total_seconds()
        export_df["sojourn_time"] = sojourn_time
    else:
        export_df["sojourn_time"] = np.nan

    # Add residence_time column
    if "end_ts" in export_df.columns and "start_ts" in export_df.columns:
        # For completed items: residence_time = sojourn_time
        # For incomplete items: residence_time = t_n - max(start_ts, t_0)
        # Guard against NaT in start_ts
        residence_time = np.where(
            export_df["end_ts"].notna(),
            export_df["sojourn_time"],  # Completed: use sojourn_time
            np.where(
                export_df["start_ts"].notna(),
                (
                    t_n - export_df["start_ts"].clip(lower=t_0)
                ).dt.total_seconds(),  # Incomplete: t_n - max(start_ts, t_0)
                np.nan,  # Missing start_ts: NaN
            ),
        )
        export_df["residence_time"] = residence_time
    else:
        export_df["residence_time"] = np.nan

    # Drop computed columns
    columns_to_drop = ["duration_td", "duration_hr"]
    export_df = export_df.drop(
        columns=[col for col in columns_to_drop if col in export_df.columns]
    )

    # Sort by start_ts
    if "start_ts" in export_df.columns:
        export_df = export_df.sort_values("start_ts")

    # Reorder columns: element_id, start_ts, end_ts, <other columns>, sojourn_time, residence_time
    fixed_cols = ["element_id", "start_ts", "end_ts"]
    trailing_cols = ["sojourn_time", "residence_time"]
    other_cols = [
        col
        for col in export_df.columns
        if col not in fixed_cols and col not in trailing_cols
    ]
    ordered_cols = (
        [col for col in fixed_cols if col in export_df.columns]
        + other_cols
        + [col for col in trailing_cols if col in export_df.columns]
    )
    export_df = export_df[ordered_cols]

    # Write CSV
    path = os.path.join(export_dir, "elements.csv")
    export_df.to_csv(path, index=False)
    return path


def export_data(
    df: pd.DataFrame,
    metrics: FlowMetricsResult,
    empirical_metrics: ElementWiseEmpiricalMetrics,
    export_dir: str,
    sampling_frequency: str | None = None,
) -> List[str]:
    """
    Export both flow metrics and element data to CSV files.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing element data.
    metrics : FlowMetricsResult
        Flow metrics result.
    empirical_metrics : ElementWiseEmpiricalMetrics
        Empirical metrics.
    export_dir : str
        Directory where CSVs will be written.
    sampling_frequency : str | None
        Sampling frequency string for calendar mode, or None for event mode.

    Returns
    -------
    List[str]
        List of paths to the created CSV files.
    """
    paths = []

    # Export flow metrics
    flow_path = export_flow_metrics_csv(
        df, metrics, empirical_metrics, export_dir, sampling_frequency
    )
    paths.append(flow_path)

    # Export element data
    element_path = export_element_csv(df, export_dir, metrics.t0, metrics.tn)
    paths.append(element_path)

    return paths
