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
    metrics = SimpleNamespace()
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


def test_plots_export_uses_core_driver():
    assert plot_core_flow_metrics_charts is core.plot_core_flow_metrics_charts
