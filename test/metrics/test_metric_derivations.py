# -*- coding: utf-8 -*-
# Copyright (c) 2025 Krishna Kumar
# SPDX-License-Identifier: MIT
from samplepath.metrics import MetricDerivations


def test_metric_derivations_get_returns_expected_string():
    assert MetricDerivations.get("N") == "N(t) = A(t) − D(t)"


def test_metric_derivations_get_unknown_key_returns_none():
    assert MetricDerivations.get("unknown") is None
