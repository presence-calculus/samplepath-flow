# -*- coding: utf-8 -*-
# Copyright (c) 2025 Krishna Kumar
# SPDX-License-Identifier: MIT
from __future__ import annotations

import os

import numpy as np
from typing import Dict, Optional, List, Literal, Tuple
from dataclasses import dataclass
import pandas as pd
from pathlib import Path

from spath.metrics import FlowMetricsResult, ElementWiseEmpiricalMetrics

MetricKind = Literal["time_average", "cumulative_rate", "empirical_rate", "bounded_mean"]


@dataclass
class ConvergenceCriteria:
    tail_frac: float = 0.35
    eps_abs: float = 0.25
    eps_rel: Optional[float] = 0.02
    slope_tol: Optional[float] = None  # auto if None
    band_coverage: float = 0.95
    min_tail_points: int = 12
    max_rate_rmse: float = 0.25  # for cumulative fits, in units of metric


DEFAULTS_BY_KIND: Dict[MetricKind, ConvergenceCriteria] = {
    "time_average": ConvergenceCriteria(eps_abs=0.25, eps_rel=0.02),
    "empirical_rate": ConvergenceCriteria(eps_abs=0.05, eps_rel=0.02),
    "bounded_mean": ConvergenceCriteria(eps_abs=0.25, eps_rel=0.03),
    "cumulative_rate": ConvergenceCriteria(eps_abs=1.0, eps_rel=None, max_rate_rmse=0.5),
}


@dataclass
class LimitResult:
    converged: bool
    limit: Optional[float]
    uncertainty: Optional[float]
    method: str
    tests: Dict[str, float]
    tail_idx: int


def estimate_limit(
        T: np.ndarray,
        Y: np.ndarray,
        tail_frac: float = 0.35,  # use last 35% as the "tail"
        eps_abs: float = 0.25,  # absolute tolerance for band/range tests (units of Y)
        eps_rel: Optional[float] = 0.02,  # relative tolerance; final eps = max(eps_abs, eps_rel*max(limit,1))
        slope_tol: Optional[float] = None,  # if None, set adaptively from tail scale
        min_tail_points: int = 12
) -> LimitResult:
    """
    Estimate the limit of Y(T) as T→∞ if it exists (numerically).
    Returns a dict with keys:
      converged (bool), limit (float or None), uncertainty (float or None),
      method (str), tests (dict), tail_idx (slice).
    Requirements:
      - T strictly increasing, same length as Y, no NaNs in the tail.
    """
    T = np.asarray(T, dtype=float)
    Y = np.asarray(Y, dtype=float)

    assert T.ndim == 1 and Y.ndim == 1 and len(T) == len(Y) and len(T) >= 5
    n = len(T)
    k0 = int(np.floor((1 - tail_frac) * n))
    k0 = max(0, min(n - min_tail_points, k0))
    tail = slice(k0, n)

    Tt, Yt = T[tail], Y[tail]
    mask = np.isfinite(Tt) & np.isfinite(Yt)
    Tt, Yt = Tt[mask], Yt[mask]
    if len(Tt) < max(min_tail_points, 5):
        return LimitResult(
            converged=False,
            limit=None,
            uncertainty=None,
            method="insufficient_tail",
            tests={},
            tail_idx=k0
        )
    # --- Test 3: slope-to-zero on tail (finite diff)
    dT = np.diff(Tt)
    dY = np.diff(Yt)
    slopes = dY / dT
    max_abs_slope = np.max(np.abs(slopes))
    # If user didn't set slope_tol, tie it to tail scale and time span
    if slope_tol is None:
        scale = np.maximum(1.0, np.median(np.abs(Yt)))
        horizon = Tt[-1] - Tt[0]
        slope_tol = 0.05 * scale / max(horizon, 1.0)  # ~5% of scale over tail span

    slope_ok = (max_abs_slope <= slope_tol)

    # --- 1/T regression on tail: Y ≈ β0 + β1*(1/T)
    x = 1.0 / Tt
    X = np.c_[np.ones_like(x), x]
    # least squares
    beta, *_ = np.linalg.lstsq(X, Yt, rcond=None)
    beta0, beta1 = beta
    yhat = X @ beta
    resid = Yt - yhat
    rmse = float(np.sqrt(np.mean(resid ** 2)))
    limit_hat = float(beta0)

    # --- Tail spread tests (Cauchy & epsilon-band)
    tail_range = float(np.max(Yt) - np.min(Yt))
    # initial absolute band around limit_hat
    eps = eps_abs
    if eps_rel is not None:
        eps = max(eps, eps_rel * max(abs(limit_hat), 1.0))

    inside = np.abs(Yt - limit_hat) <= eps
    band_ok = (inside.mean() >= 0.95)  # at least 95% of tail in the band
    cauchy_ok = (tail_range <= 2 * eps)  # whole tail fits within a 2*eps slab

    # --- Uncertainty estimate: robust tail MAD blended with regression RMSE
    mad = np.median(np.abs(Yt - np.median(Yt))) * 1.4826
    unc = float(max(mad, rmse))

    tests = {
        "tail_points": int(len(Tt)),
        "tail_range": tail_range,
        "eps_used": eps,
        "band_coverage": float(inside.mean()),
        "cauchy_ok": bool(cauchy_ok),
        "slope_tol": float(slope_tol),
        "max_abs_slope": float(max_abs_slope),
        "rmse_1_over_T_fit": rmse,
        "beta1_magnitude": float(abs(beta1)),
    }

    converged = bool(band_ok and cauchy_ok and slope_ok)

    return LimitResult(
        converged=converged,
        limit=limit_hat if converged else None,
        uncertainty=unc if converged else None,
        method="1/T_intercept_with_tail_checks",
        tests=tests,
        tail_idx=k0
    )


def _ols_two_param(x: np.ndarray, y: np.ndarray) -> Tuple[float, float, float, float]:
    """
    OLS for y = a + b*x.
    Returns (a_hat, b_hat, rmse, se_b).
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    X = np.c_[np.ones_like(x), x]
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    yhat = X @ beta
    resid = y - yhat
    n = len(x)
    rmse = float(np.sqrt((resid @ resid) / max(n - 2, 1)))
    # standard error of slope
    s2 = (resid @ resid) / max(n - 2, 1)
    Sxx = float(np.sum((x - x.mean()) ** 2))
    se_b = float(np.sqrt(s2 / max(Sxx, 1e-12)))
    return float(beta[0]), float(beta[1]), rmse, se_b


def estimate_linear_rate(
        T: np.ndarray,
        C: np.ndarray,
        tail_frac: float = 0.35,
        min_tail_points: int = 12,
        max_rate_rmse: float = 0.25,  # units of C
        rate_rel_tol: float = 0.03,  # 3% relative stability tolerance on rho across the tail
        curvature_improvement_tol: float = 0.05,  # adding 1/T must not improve RMSE by >5%
        eps_abs: float = 1.0,  # not used directly; kept for uniform interface
        eps_rel: Optional[float] = None,  # not used directly; kept for uniform interface
        **_
) -> LimitResult:
    """
    Estimate the asymptotic rate ρ from a cumulative series C(T) via tail OLS:
        C(T) ≈ α + ρ T.
    Convergence is declared if:
      - OLS RMSE on the tail ≤ max_rate_rmse
      - The slope ρ is stable across tail splits within rate_rel_tol
      - Adding a curvature term (1/T) does not materially reduce RMSE

    Returns:
      dict(converged, limit=ρ̂, uncertainty=se(ρ̂), method, tests, tail_idx)
    """
    T = np.asarray(T, dtype=float)
    C = np.asarray(C, dtype=float)
    assert T.ndim == 1 and C.ndim == 1 and len(T) == len(C) and len(T) >= 5

    n = len(T)
    k0 = int(np.floor((1 - tail_frac) * n))
    k0 = max(0, min(n - min_tail_points, k0))
    tail = slice(k0, n)

    Tt, Ct = T[tail], C[tail]
    mask = np.isfinite(Tt) & np.isfinite(Ct)
    Tt, Ct = Tt[mask], Ct[mask]
    if len(Tt) < max(min_tail_points, 5):
        return LimitResult(
            converged=False,
            limit=None,
            uncertainty=None,
            method="rate_ols_insufficient_tail",
            tests={},
            tail_idx=k0
        )

    # Main OLS: C ~ a + rho*T
    a_hat, rho_hat, rmse, se_rho = _ols_two_param(Tt, Ct)

    # Stability across the tail: compare first vs last 60% windows (overlapping to have enough points)
    m = len(Tt)
    kA = max(int(np.ceil(0.60 * m)), 3)
    A = slice(0, kA)
    B = slice(m - kA, m)
    _, rho_A, rmse_A, _ = _ols_two_param(Tt[A], Ct[A])
    _, rho_B, rmse_B, _ = _ols_two_param(Tt[B], Ct[B])
    rho_diff_abs = float(abs(rho_A - rho_B))
    rho_diff_rel = float(rho_diff_abs / max(abs(rho_hat), 1e-12))

    # Curvature check: add 1/T term
    x2 = 1.0 / Tt
    X_curv = np.c_[np.ones_like(Tt), Tt, x2]
    beta_c, *_ = np.linalg.lstsq(X_curv, Ct, rcond=None)
    yhat_c = X_curv @ beta_c
    resid_c = Ct - yhat_c
    rmse_curv = float(np.sqrt((resid_c @ resid_c) / max(len(Tt) - 3, 1)))
    rmse_improve = float((rmse - rmse_curv) / max(rmse, 1e-12))  # fractional improvement

    converged = (
            rmse <= max_rate_rmse
            and rho_diff_rel <= rate_rel_tol
            and rmse_improve <= curvature_improvement_tol
    )

    tests = {
        "tail_points": int(len(Tt)),
        "rho_hat": float(rho_hat),
        "rmse": rmse,
        "se_rho": se_rho,
        "rho_A": float(rho_A),
        "rho_B": float(rho_B),
        "rho_diff_abs": rho_diff_abs,
        "rho_diff_rel": rho_diff_rel,
        "rmse_curv": rmse_curv,
        "rmse_improvement_frac_with_1_over_T": rmse_improve,
        "max_rate_rmse": float(max_rate_rmse),
        "rate_rel_tol": float(rate_rel_tol),
    }

    return LimitResult(
        converged=bool(converged),
        limit=float(rho_hat) if converged else None,
        uncertainty=float(se_rho) if converged else None,
        method="tail_OLS_rate (C ~ a + ρT)",
        tests=tests,
        tail_idx=k0
    )


def compare_series_tail(
        T: np.ndarray,
        Y1: np.ndarray,
        Y2: np.ndarray,
        tail_frac: float = 0.35,
        min_tail_points: int = 12,
        eps_abs: float = 0.25,
        eps_rel: Optional[float] = 0.02,
        band_coverage: float = 0.95,
        **_
) -> LimitResult:
    """
    Tail agreement test between two series (e.g., L(T) vs λ*(T)·W*(T)).
    Declares 'converged' (coherent) if the absolute/relative difference in the tail
    is within an ε-band for at least `band_coverage` fraction of tail points.

    Returns:
      dict(converged, limit=None, uncertainty=MAD_diff, method, tests, tail_idx)
    """
    T = np.asarray(T, dtype=float)
    Y1 = np.asarray(Y1, dtype=float)
    Y2 = np.asarray(Y2, dtype=float)
    assert T.ndim == 1 and len(T) == len(Y1) == len(Y2)

    n = len(T)
    k0 = int(np.floor((1 - tail_frac) * n))
    k0 = max(0, min(n - min_tail_points, k0))
    tail = slice(k0, n)

    Tt = T[tail]
    D = (Y1 - Y2)[tail]
    mask = np.isfinite(Tt) & np.isfinite(D)
    Dt = D[mask]
    if len(Dt) < max(min_tail_points, 5):
        return LimitResult(
            converged=False,
            limit=None,
            uncertainty=None,
            method="tail_agreement_insufficient_tail",
            tests={},
            tail_idx=k0
        )

    # Scale for relative tolerance
    # Use the median magnitude of the two series on tail to avoid being skewed by outliers.
    Y1t = Y1[tail][mask]
    Y2t = Y2[tail][mask]
    scale = float(max(np.median(np.abs(np.r_[Y1t, Y2t])), 1.0))

    eps = float(max(eps_abs, (eps_rel or 0.0) * scale))
    inside = np.abs(Dt) <= eps
    coverage = float(np.mean(inside))

    # Robust uncertainty as MAD of differences
    mad = float(np.median(np.abs(Dt - np.median(Dt))) * 1.4826)
    tail_range = float(np.max(Dt) - np.min(Dt))
    mean_abs = float(np.mean(np.abs(Dt)))
    median_abs = float(np.median(np.abs(Dt)))

    converged = bool(coverage >= band_coverage and tail_range <= 2 * eps)

    tests = {
        "tail_points": int(len(Dt)),
        "eps_used": eps,
        "coverage": coverage,
        "tail_range": tail_range,
        "mean_abs_diff": mean_abs,
        "median_abs_diff": median_abs,
        "mad_abs_diff": mad,
        "band_coverage_required": float(band_coverage),
    }

    return LimitResult(
        converged=converged,
        limit=None,  # this test checks agreement, not a limit value
        uncertainty=mad,
        method="tail_epsilon_band_agreement",
        tests=tests,
        tail_idx=k0
    )


def measure_process_limits(
    metrics: FlowMetricsResult,
    element: ElementWiseEmpiricalMetrics,
    criteria_overrides: Optional[Dict[str, ConvergenceCriteria]] = None
) -> Dict[str, LimitResult]:
    """
    Top Level Driver:

    Returns limit estimates for:
      L(T), w(T), Λ(T) [rate], λ*(T), W*(T), and a coherence check for L vs λ*W.
    """
    times: List[pd.Timestamp] = metrics.times
    T = np.array(pd.to_datetime(times).astype("int64") / 1e9 / 86400.0)  # days since epoch
    out: Dict[str, LimitResult] = {}

    def crit(name: str, kind: MetricKind) -> ConvergenceCriteria:
        co = criteria_overrides.get(name) if criteria_overrides else None
        return co or DEFAULTS_BY_KIND[kind]

    # helper to extract only fields relevant to estimate_limit
    def limit_kwargs(c: ConvergenceCriteria) -> dict:
        return dict(
            tail_frac=c.tail_frac,
            eps_abs=c.eps_abs,
            eps_rel=c.eps_rel,
            slope_tol=c.slope_tol,
            min_tail_points=c.min_tail_points
        )

    # helper to extract only fields relevant to estimate_linear_rate
    def rate_kwargs(c: ConvergenceCriteria) -> dict:
        return dict(
            tail_frac=c.tail_frac,
            min_tail_points=c.min_tail_points,
            max_rate_rmse=c.max_rate_rmse,
        )

    # helper to extract fields relevant to compare_series_tail
    def coherence_kwargs(c: ConvergenceCriteria) -> dict:
        return dict(
            tail_frac=c.tail_frac,
            min_tail_points=c.min_tail_points,
            eps_abs=c.eps_abs,
            eps_rel=c.eps_rel,
            band_coverage=c.band_coverage,
        )

    # 1) L(T): time-average
    c = crit("L", "time_average")
    out["L"] = estimate_limit(T, metrics.L, **limit_kwargs(c))

    # 2) w(T): bounded mean
    c = crit("w", "bounded_mean")
    out["w"] = estimate_limit(T, metrics.w, **limit_kwargs(c))

    # 3) Λ(T): estimate *rate* via linear fit C(T) ≈ α + ρ T
    c = crit("Lambda_rate", "cumulative_rate")
    out["Lambda_rate"] = estimate_linear_rate(T, metrics.Lambda, **rate_kwargs(c))

    # 4) λ*(T), W*(T): time-averages of empirical series
    c = crit("lambda_star", "empirical_rate")
    out["lambda_star"] = estimate_limit(T, element.lam_star, **limit_kwargs(c))
    c = crit("W_star", "bounded_mean")
    out["W_star"] = estimate_limit(T, element.W_star, **limit_kwargs(c))

    # 5) Coherence: L(T) vs λ*(T)·W*(T)
    lamW = element.lam_star * element.W_star
    c = crit("coherence", "time_average")
    out["coherence_L_vs_lamW"] = compare_series_tail(T, metrics.L, lamW, **coherence_kwargs(c))

    return out



def write_limit_results(results: Dict[str, LimitResult], filepath: str | Path) -> None:
    """
    Pretty-print a dictionary of metric -> LimitResult to a text file.

    Parameters
    ----------
    results : Dict[str, LimitResult]
        Mapping from metric names (e.g., 'L', 'Lambda_rate') to LimitResult objects.
    filepath : str | Path
        Output file path (will be overwritten).
    """
    path = Path(filepath)
    lines: list[str] = []
    sep = "━" * 72

    lines.append("PROCESS LIMIT ESTIMATION RESULTS")
    lines.append(sep)

    for name, res in results.items():
        lines.append(f"{name.upper():<20} {'✅' if res.converged else '❌'}  ({res.method})")

        if res.limit is not None:
            lines.append(f"    Limit:        {res.limit:,.6f}")
        if res.uncertainty is not None:
            lines.append(f"    Uncertainty:  ±{res.uncertainty:,.6f}")

        # Tail range and coverage if available in tests
        t = res.tests
        if "tail_range" in t:
            lines.append(f"    Tail range:   {t['tail_range']:.6f}")
        if "band_coverage" in t or "coverage" in t:
            cov = t.get("band_coverage", t.get("coverage"))
            if cov is not None:
                lines.append(f"    Coverage:     {cov * 100:.1f}%")
        if "rmse" in t:
            lines.append(f"    RMSE:         {t['rmse']:.6f}")

        lines.append(f"    Tail points:  {t.get('tail_points', '—')}")
        lines.append(sep)

    path.write_text("\n".join(lines))
    print(f"Wrote {len(results)} limit results → {path}")


def write_limits(metrics, empirical_metrics, out_dir):
    limits = measure_process_limits(metrics, empirical_metrics)
    write_limit_results(limits, os.path.join(out_dir, 'advanced/limits.txt'))
