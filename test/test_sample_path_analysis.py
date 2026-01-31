# -*- coding: utf-8 -*-
# Copyright (c) 2025 Krishna Kumar
# SPDX-License-Identifier: MIT
from types import SimpleNamespace
from unittest.mock import patch

from samplepath import sample_path_analysis
from samplepath.plots import core, plot_core_flow_metrics_charts


def _stub_inputs():
    df = None
    args = SimpleNamespace()
    filter_result = SimpleNamespace()
    metrics = SimpleNamespace(w=[0.0])
    empirical_metrics = SimpleNamespace()
    out_dir = "/tmp/out"
    return df, args, filter_result, metrics, empirical_metrics, out_dir


def test_produce_all_charts_calls_core_driver():
    df, args, filter_result, metrics, empirical_metrics, out_dir = _stub_inputs()
    with (
        patch(
            "samplepath.sample_path_analysis.plot_core_flow_metrics_charts"
        ) as mock_core,
        patch(
            "samplepath.sample_path_analysis.plot_convergence_charts", return_value=[]
        ),
        patch("samplepath.sample_path_analysis.plot_stability_charts", return_value=[]),
        patch("samplepath.sample_path_analysis.plot_advanced_charts", return_value=[]),
    ):
        sample_path_analysis.produce_all_charts(
            df, args, filter_result, metrics, empirical_metrics, out_dir
        )
    assert mock_core.call_count == 1


def test_produce_all_charts_returns_concatenated_list():
    df, args, filter_result, metrics, empirical_metrics, out_dir = _stub_inputs()
    with (
        patch(
            "samplepath.sample_path_analysis.plot_core_flow_metrics_charts",
            return_value=["core.png"],
        ),
        patch(
            "samplepath.sample_path_analysis.plot_convergence_charts",
            return_value=["conv.png"],
        ),
        patch(
            "samplepath.sample_path_analysis.plot_stability_charts",
            return_value=["stab.png"],
        ),
        patch(
            "samplepath.sample_path_analysis.plot_advanced_charts",
            return_value=["adv.png"],
        ),
    ):
        written = sample_path_analysis.produce_all_charts(
            df, args, filter_result, metrics, empirical_metrics, out_dir
        )
    assert written == ["core.png", "conv.png", "stab.png", "adv.png"]


def test_run_analysis_passes_sampling_frequency_to_metrics():
    args = SimpleNamespace(
        sampling_frequency="week",
        delimiter=None,
        start_column="start_ts",
        end_column="end_ts",
        date_format=None,
        dayfirst=False,
        completed=False,
        incomplete=False,
        classes=None,
        outlier_hours=None,
        outlier_pctl=None,
        outlier_iqr=None,
        outlier_iqr_two_sided=False,
    )
    import pandas as pd

    fake_df = pd.DataFrame(
        {
            "id": [1],
            "start_ts": [pd.Timestamp("2024-01-01")],
            "end_ts": [pd.Timestamp("2024-01-02")],
        }
    )
    fake_filter = SimpleNamespace(df=fake_df, display="", label="")
    with (
        patch("samplepath.sample_path_analysis.csv_to_dataframe", return_value=fake_df),
        patch(
            "samplepath.sample_path_analysis.apply_filters", return_value=fake_filter
        ),
        patch(
            "samplepath.sample_path_analysis.to_arrival_departure_process",
            return_value=[
                (pd.Timestamp("2024-01-01"), 1, 1),
                (pd.Timestamp("2024-01-02"), -1, 0),
            ],
        ),
        patch(
            "samplepath.sample_path_analysis.compute_finite_window_flow_metrics"
        ) as mock_metrics,
        patch("samplepath.sample_path_analysis.compute_elementwise_empirical_metrics"),
        patch("samplepath.sample_path_analysis.write_limits"),
        patch("samplepath.sample_path_analysis.produce_all_charts", return_value=[]),
    ):
        mock_metrics.return_value = SimpleNamespace(times=[])
        sample_path_analysis.run_analysis("dummy.csv", args, "/tmp/out")
    assert mock_metrics.call_args.kwargs["freq"] == "week"


def test_run_analysis_passes_anchor_to_metrics():
    args = SimpleNamespace(
        sampling_frequency="week",
        anchor="WED",
        delimiter=None,
        start_column="start_ts",
        end_column="end_ts",
        date_format=None,
        dayfirst=False,
        completed=False,
        incomplete=False,
        classes=None,
        outlier_hours=None,
        outlier_pctl=None,
        outlier_iqr=None,
        outlier_iqr_two_sided=False,
    )
    import pandas as pd

    fake_df = pd.DataFrame(
        {
            "id": [1],
            "start_ts": [pd.Timestamp("2024-01-01")],
            "end_ts": [pd.Timestamp("2024-01-02")],
        }
    )
    fake_filter = SimpleNamespace(df=fake_df, display="", label="")
    with (
        patch("samplepath.sample_path_analysis.csv_to_dataframe", return_value=fake_df),
        patch(
            "samplepath.sample_path_analysis.apply_filters", return_value=fake_filter
        ),
        patch(
            "samplepath.sample_path_analysis.to_arrival_departure_process",
            return_value=[
                (pd.Timestamp("2024-01-01"), 1, 1),
                (pd.Timestamp("2024-01-02"), -1, 0),
            ],
        ),
        patch(
            "samplepath.sample_path_analysis.compute_finite_window_flow_metrics"
        ) as mock_metrics,
        patch("samplepath.sample_path_analysis.compute_elementwise_empirical_metrics"),
        patch("samplepath.sample_path_analysis.write_limits"),
        patch("samplepath.sample_path_analysis.produce_all_charts", return_value=[]),
    ):
        mock_metrics.return_value = SimpleNamespace(times=[])
        sample_path_analysis.run_analysis("dummy.csv", args, "/tmp/out")
    kwargs = mock_metrics.call_args.kwargs
    assert kwargs["week_anchor"] == "WED"
    assert kwargs["quarter_anchor"] == "WED"
    assert kwargs["year_anchor"] == "WED"


def test_plots_export_uses_core_driver():
    assert plot_core_flow_metrics_charts is core.plot_core_flow_metrics_charts
