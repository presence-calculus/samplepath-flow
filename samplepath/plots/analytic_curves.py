# -*- coding: utf-8 -*-
# Copyright (c) 2026 Krishna Kumar
# SPDX-License-Identifier: MIT
"""Analytical between-event curve builders for event-indexed metrics."""

from __future__ import annotations

from typing import List, Sequence, Tuple

import numpy as np
import pandas as pd


def _elapsed_seconds(times: Sequence[pd.Timestamp]) -> np.ndarray:
    """Return elapsed seconds from the first timestamp.

    Parameters
    ----------
    times
        Monotone timestamp sequence.

    Returns
    -------
    np.ndarray
        Float array ``T_i = (times[i] - times[0]).total_seconds()``.
        Returns an empty array for empty input.
    """
    if len(times) == 0:
        return np.array([], dtype=float)
    t0 = times[0]
    return np.array([(t - t0).total_seconds() for t in times], dtype=float)


def _samples_for_interval(delta_t: float, samples_per_interval: int) -> int:
    """Compute fixed per-interval sample count.

    Parameters
    ----------
    delta_t
        Interval width in seconds.
    samples_per_interval
        Requested number of samples per interval.

    Returns
    -------
    int
        ``0`` for non-positive intervals, else at least ``2`` samples.
    """
    if delta_t <= 0.0:
        return 0
    return max(2, int(samples_per_interval))


def _build_rate_curve(
    times: Sequence[pd.Timestamp],
    cumulative_vals: Sequence[float],
    *,
    samples_per_interval: int = 80,
    t_min_seconds: float = 1e-9,
) -> Tuple[List[pd.Timestamp], np.ndarray, List[Tuple[pd.Timestamp, float, float]]]:
    """Build analytical ``C(T)/T`` curve samples and jump metadata.

    Generic helper for rate metrics like ``Lambda(T) = A(T)/T`` and
    ``Theta(T) = D(T)/T``. Between events, the cumulative count ``C(T)``
    is constant under the right-continuous event convention, so for
    interval ``[T_i, T_{i+1})``:

    ``rate(T) = C_i / T``.

    Parameters
    ----------
    times
        Event-indexed timestamps in ascending order.
    cumulative_vals
        Event-indexed cumulative counts aligned to ``times``.
    samples_per_interval
        Number of within-interval samples (fixed for every interval).
    t_min_seconds
        Positive cutoff to avoid the ``T=0`` singularity.

    Returns
    -------
    tuple
        ``(curve_times, curve_values, jumps)`` where:
        - ``curve_times`` are sampled timestamps for plotting,
        - ``curve_values`` are sampled rate values in native units,
        - ``jumps`` is a list of ``(time, left, right)`` values.
    """
    n = len(times)
    if n == 0:
        return [], np.array([], dtype=float), []
    if len(cumulative_vals) != n:
        raise ValueError("times and cumulative_vals must have equal lengths")
    if n == 1:
        return [], np.array([], dtype=float), []

    elapsed = _elapsed_seconds(times)
    t0 = times[0]
    all_sample_t: List[np.ndarray] = []
    all_values: List[np.ndarray] = []
    jumps: List[Tuple[pd.Timestamp, float, float]] = []

    for i in range(n - 1):
        start_t = float(elapsed[i])
        end_t = float(elapsed[i + 1])
        delta_t = end_t - start_t
        k_i = _samples_for_interval(delta_t, samples_per_interval)
        if k_i > 0:
            sample_t = np.linspace(start_t, end_t, num=k_i, endpoint=False, dtype=float)
            valid_mask = sample_t >= max(0.0, t_min_seconds)
            if np.any(valid_mask):
                sample_t = sample_t[valid_mask]
                rate_t = float(cumulative_vals[i]) / sample_t
                all_sample_t.append(sample_t)
                all_values.append(rate_t)

        event_t = float(elapsed[i + 1])
        if event_t >= max(0.0, t_min_seconds):
            c_before = float(cumulative_vals[i])
            c_after = float(cumulative_vals[i + 1])
            if not np.isclose(c_before, c_after):
                jumps.append(
                    (
                        times[i + 1],
                        c_before / event_t,
                        c_after / event_t,
                    )
                )

    final_t = float(elapsed[-1])
    if final_t >= max(0.0, t_min_seconds):
        all_sample_t.append(np.array([final_t]))
        all_values.append(np.array([float(cumulative_vals[-1]) / final_t]))

    if not all_sample_t:
        return [], np.array([], dtype=float), jumps

    combined_t = np.concatenate(all_sample_t)
    out_times = list(t0 + pd.to_timedelta(combined_t, unit="s"))
    out_values = np.concatenate(all_values)

    return out_times, out_values, jumps


def build_lambda_curve(
    times: Sequence[pd.Timestamp],
    arrivals: Sequence[float],
    *,
    samples_per_interval: int = 80,
    t_min_seconds: float = 1e-9,
) -> Tuple[List[pd.Timestamp], np.ndarray, List[Tuple[pd.Timestamp, float, float]]]:
    """Build analytical ``Lambda(T)`` curve samples and jump metadata.

    Between events, the cumulative arrivals ``A(T)`` are constant under the
    right-continuous event convention, so for interval ``[T_i, T_{i+1})``:

    ``Lambda(T) = A_i / T``.

    The function samples this formula within each interval using a fixed sample
    count, then appends the final event point when defined. It also emits
    vertical-jump metadata at event times where arrivals change:
    ``(event_time, left_limit, right_limit)``.

    Parameters
    ----------
    times
        Event-indexed timestamps in ascending order.
    arrivals
        Event-indexed cumulative arrivals ``A_i`` aligned to ``times``.
    samples_per_interval
        Number of within-interval samples (fixed for every interval).
    t_min_seconds
        Positive cutoff to avoid the ``T=0`` singularity in ``A(T)/T``.

    Returns
    -------
    tuple
        ``(curve_times, curve_values, jumps)`` where:
        - ``curve_times`` are sampled timestamps for plotting,
        - ``curve_values`` are sampled ``Lambda(T)`` values in native units,
        - ``jumps`` is a list of ``(time, left, right)`` values.

    Notes
    -----
    - Empty or single-point inputs return empty outputs.
    - Mismatched ``times``/``arrivals`` lengths raise ``ValueError``.
    """
    return _build_rate_curve(
        times,
        arrivals,
        samples_per_interval=samples_per_interval,
        t_min_seconds=t_min_seconds,
    )


def build_theta_curve(
    times: Sequence[pd.Timestamp],
    departures: Sequence[float],
    *,
    samples_per_interval: int = 80,
    t_min_seconds: float = 1e-9,
) -> Tuple[List[pd.Timestamp], np.ndarray, List[Tuple[pd.Timestamp, float, float]]]:
    """Build analytical ``Theta(T)`` curve samples and jump metadata.

    Between events, cumulative departures ``D(T)`` are constant under the
    right-continuous event convention, so for interval ``[T_i, T_{i+1})``:

    ``Theta(T) = D_i / T``.

    Parameters and return values mirror :func:`build_lambda_curve`, with
    ``departures`` replacing ``arrivals`` and jump points emitted only when
    cumulative departures change.
    """
    return _build_rate_curve(
        times,
        departures,
        samples_per_interval=samples_per_interval,
        t_min_seconds=t_min_seconds,
    )


def build_L_curve(
    times: Sequence[pd.Timestamp],
    H_vals: Sequence[float],
    N_vals: Sequence[float],
    *,
    samples_per_interval: int = 80,
    t_min_seconds: float = 1e-9,
) -> Tuple[List[pd.Timestamp], np.ndarray]:
    """Build analytical ``L(T)`` curve samples.

    Under right-continuous event indexing, ``N(T)=N_i`` on
    ``[T_i, T_{i+1})`` and

    ``H(T) = H_i + N_i * (T - T_i)``,

    so this function samples:

    ``L(T) = (H_i + N_i * (T - T_i)) / T``

    on each interval and appends the final event point when defined.

    Parameters
    ----------
    times
        Event-indexed timestamps in ascending order.
    H_vals
        Event-indexed cumulative presence ``H_i`` aligned to ``times``.
    N_vals
        Event-indexed in-system count ``N_i`` aligned to ``times``.
    samples_per_interval
        Number of within-interval samples (fixed for every interval).
    t_min_seconds
        Positive cutoff to avoid the ``T=0`` singularity in ``H(T)/T``.

    Returns
    -------
    tuple
        ``(curve_times, curve_values)`` sampled for plotting.

    Notes
    -----
    - Empty or single-point inputs return empty outputs.
    - Mismatched input lengths raise ``ValueError``.
    """

    n = len(times)
    if n == 0:
        return [], np.array([], dtype=float)
    if len(H_vals) != n or len(N_vals) != n:
        raise ValueError("times, H_vals and N_vals must have equal lengths")
    if n == 1:
        return [], np.array([], dtype=float)

    elapsed = _elapsed_seconds(times)
    t0 = times[0]
    all_sample_t: List[np.ndarray] = []
    all_values: List[np.ndarray] = []

    for i in range(n - 1):
        start_t = float(elapsed[i])
        end_t = float(elapsed[i + 1])
        delta_t = end_t - start_t
        k_i = _samples_for_interval(delta_t, samples_per_interval)
        if k_i == 0:
            continue

        sample_t = np.linspace(start_t, end_t, num=k_i, endpoint=False, dtype=float)
        valid_mask = sample_t >= max(0.0, t_min_seconds)
        if not np.any(valid_mask):
            continue

        sample_t = sample_t[valid_mask]
        H_i = float(H_vals[i])
        N_i = float(N_vals[i])
        H_t = H_i + N_i * (sample_t - start_t)
        L_t = H_t / sample_t

        all_sample_t.append(sample_t)
        all_values.append(L_t)

    final_t = float(elapsed[-1])
    if final_t >= max(0.0, t_min_seconds):
        all_sample_t.append(np.array([final_t]))
        all_values.append(np.array([float(H_vals[-1]) / final_t]))

    if not all_sample_t:
        return [], np.array([], dtype=float)

    combined_t = np.concatenate(all_sample_t)
    out_times = list(t0 + pd.to_timedelta(combined_t, unit="s"))
    out_values = np.concatenate(all_values)

    return out_times, out_values


def _build_residence_curve(
    times: Sequence[pd.Timestamp],
    H_vals: Sequence[float],
    N_vals: Sequence[float],
    counts: Sequence[float],
    *,
    samples_per_interval: int = 80,
) -> Tuple[List[pd.Timestamp], np.ndarray, List[Tuple[pd.Timestamp, float, float]]]:
    """Build analytical curve samples for ``H(T)/count(T)`` metrics.

    This helper supports metrics whose denominator is step-wise constant between
    events, such as ``w(T)=H(T)/A(T)`` and ``w'(T)=H(T)/D(T)``. Over interval
    ``[T_i, T_{i+1})``:

    ``H(T) = H_i + N_i * (T - T_i)`` and ``count(T) = count_i``.

    Samples are emitted only where ``count_i > 0`` to avoid undefined values.
    Jump metadata is emitted at event times where ``count`` changes and both
    left/right denominators are positive.
    """

    n = len(times)
    if n == 0:
        return [], np.array([], dtype=float), []
    if len(H_vals) != n or len(N_vals) != n or len(counts) != n:
        raise ValueError("times, H_vals, N_vals and counts must have equal lengths")
    if n == 1:
        return [], np.array([], dtype=float), []

    elapsed = _elapsed_seconds(times)
    t0 = times[0]
    all_sample_t: List[np.ndarray] = []
    all_values: List[np.ndarray] = []
    jumps: List[Tuple[pd.Timestamp, float, float]] = []

    for i in range(n - 1):
        start_t = float(elapsed[i])
        end_t = float(elapsed[i + 1])
        delta_t = end_t - start_t
        k_i = _samples_for_interval(delta_t, samples_per_interval)
        denom_before = float(counts[i])

        if k_i > 0 and denom_before > 0.0:
            sample_t = np.linspace(start_t, end_t, num=k_i, endpoint=False, dtype=float)
            H_i = float(H_vals[i])
            N_i = float(N_vals[i])
            H_t = H_i + N_i * (sample_t - start_t)
            m_t = H_t / denom_before
            all_sample_t.append(sample_t)
            all_values.append(m_t)

        denom_after = float(counts[i + 1])
        if not np.isclose(denom_before, denom_after):
            if denom_before > 0.0 and denom_after > 0.0:
                H_event = float(H_vals[i]) + float(N_vals[i]) * delta_t
                jumps.append(
                    (
                        times[i + 1],
                        H_event / denom_before,
                        H_event / denom_after,
                    )
                )

    denom_final = float(counts[-1])
    if denom_final > 0.0:
        all_sample_t.append(np.array([0.0]))
        all_values.append(np.array([float(H_vals[-1]) / denom_final]))

    if not all_sample_t:
        return [], np.array([], dtype=float), jumps

    # Batch timestamp conversion: replace final dummy 0.0 with actual time
    combined_t = np.concatenate(all_sample_t)
    if denom_final > 0.0:
        combined_t[-1] = float(elapsed[-1])
        out_times = list(t0 + pd.to_timedelta(combined_t[:-1], unit="s"))
        out_times.append(times[-1])
    else:
        out_times = list(t0 + pd.to_timedelta(combined_t, unit="s"))
    out_values = np.concatenate(all_values)

    return out_times, out_values, jumps


def build_w_curve(
    times: Sequence[pd.Timestamp],
    H_vals: Sequence[float],
    N_vals: Sequence[float],
    arrivals: Sequence[float],
    *,
    samples_per_interval: int = 80,
) -> Tuple[List[pd.Timestamp], np.ndarray, List[Tuple[pd.Timestamp, float, float]]]:
    """Build analytical ``w(T) = H(T)/A(T)`` samples and jump metadata.

    Parameters mirror :func:`build_L_curve`, with ``arrivals`` providing the
    denominator ``A_i``. Samples are emitted only where ``A_i > 0``.
    Jump metadata captures vertical discontinuities at event times where
    arrivals change and both sides are defined.
    """

    return _build_residence_curve(
        times,
        H_vals,
        N_vals,
        arrivals,
        samples_per_interval=samples_per_interval,
    )


def build_w_prime_curve(
    times: Sequence[pd.Timestamp],
    H_vals: Sequence[float],
    N_vals: Sequence[float],
    departures: Sequence[float],
    *,
    samples_per_interval: int = 80,
) -> Tuple[List[pd.Timestamp], np.ndarray, List[Tuple[pd.Timestamp, float, float]]]:
    """Build analytical ``w'(T) = H(T)/D(T)`` samples and jump metadata.

    Parameters mirror :func:`build_w_curve`, with ``departures`` as
    denominator ``D_i``.
    """

    return _build_residence_curve(
        times,
        H_vals,
        N_vals,
        departures,
        samples_per_interval=samples_per_interval,
    )
