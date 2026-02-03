# -*- coding: utf-8 -*-
# Copyright (c) 2025 Krishna Kumar
# SPDX-License-Identifier: MIT
from __future__ import annotations

import os
from typing import List, Optional, Tuple

from matplotlib import pyplot as plt
import numpy as np
import pandas as pd

from samplepath.filter import FilterResult
from samplepath.metrics import (
    FlowMetricsResult,
    compute_elementwise_empirical_metrics,
)
from samplepath.plots.chart_config import ChartConfig
from samplepath.plots.helpers import (
    _clip_axis_to_percentile,
    apply_gridlines,
    format_date_axis,
)
from samplepath.utils.duration_scale import HOURS, DurationScale


def _resolve_duration_scale(chart_config: ChartConfig) -> DurationScale:
    return chart_config.duration_scale or HOURS


def plot_llaw_manifold_3d(
    df,
    metrics,  # FlowMetricsResult
    out_dir: str,
    title: str = "Manifold view: L = Λ · w (log-space plane z = x + y)",
    caption: Optional[str] = None,
    figsize: Tuple[int, int] = (9, 7),
    elev: float = 28.0,
    azim: float = -135.0,
    alpha_surface: float = 0.22,  # kept in signature; not used explicitly
    wireframe_stride: int = 6,
    point_size: int = 16,
) -> List[str]:
    """Log-linear manifold: z = x + y with x=log Λ, y=log w, z=log L.
    Plots only the finite-time trajectory on a filled plane (no empirical series).
    """
    import os

    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
    import numpy as np

    # ---- helpers ------------------------------------------------------------
    def _safe_log(a: np.ndarray) -> np.ndarray:
        """Natural log with nan for non-positive entries."""
        out = np.full_like(a, np.nan, dtype=float)
        m = np.isfinite(a) & (a > 0)
        out[m] = np.log(a[m])
        return out

    def _finite(*arrs):
        m = np.ones_like(arrs[0], dtype=bool)
        for a in arrs:
            m &= np.isfinite(a)
        return m

    def _pad_range(v: np.ndarray, pad: float = 0.05) -> Tuple[float, float]:
        v = v[np.isfinite(v)]
        if v.size == 0:
            return (-1.0, 1.0)
        lo, hi = float(np.nanmin(v)), float(np.nanmax(v))
        span = hi - lo
        if not np.isfinite(span) or span <= 0:
            span = 1.0
        return lo - pad * span, hi + pad * span

    # ---- finite-time series (on-plane in log space) -------------------------
    T = metrics.times  # not used here but kept for signature compatibility
    L_vals = np.asarray(metrics.L, dtype=float)
    Lam_vals = np.asarray(metrics.Lambda, dtype=float)
    w_vals = np.asarray(metrics.w, dtype=float)

    # Logs for the finite trio
    x_fin = _safe_log(Lam_vals)  # log Λ
    y_fin = _safe_log(w_vals)  # log w
    z_fin = _safe_log(L_vals)  # log L
    mask_fin = _finite(x_fin, y_fin, z_fin)

    # ---- figure / axes ------------------------------------------------------
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection="3d")
    ax.grid(False)
    try:
        ax.set_proj_type("ortho")  # orthographic for honest distances
    except Exception:
        pass

    # Plane extents from finite trajectory only
    x_lo, x_hi = _pad_range(x_fin[mask_fin] if mask_fin.any() else np.array([0.0]))
    y_lo, y_hi = _pad_range(y_fin[mask_fin] if mask_fin.any() else np.array([0.0]))

    # Build grid for plane z = x + y
    nx = max(10, 2 * wireframe_stride)
    ny = max(10, 2 * wireframe_stride)
    Xg = np.linspace(x_lo, x_hi, nx)
    Yg = np.linspace(y_lo, y_hi, ny)
    X, Y = np.meshgrid(Xg, Yg)
    Z = X + Y

    # Filled plane: darker grey, semi-opaque; no mesh lines
    ax.plot_surface(X, Y, Z, color="dimgray", alpha=0.5, linewidth=0, antialiased=True)

    # ---- finite-time trajectory (lies on the plane) -------------------------
    if mask_fin.any():
        ax.plot(
            x_fin[mask_fin],
            y_fin[mask_fin],
            z_fin[mask_fin],
            lw=1.6,
            label="(log Λ, log w, log L)",
        )

    # ---- labels / view / legend / caption ----------------------------------
    ax.set_xlabel("log Λ")
    ax.set_ylabel("log w")
    ax.set_zlabel("log L")
    ax.view_init(elev=elev, azim=azim)
    ax.set_title(title)
    try:
        ax.legend(loc="upper left")
    except Exception:
        pass

    if caption:
        fig.text(0.01, 0.01, caption, ha="left", va="bottom", fontsize=9)

    # z-limits: include plane and line
    if np.isfinite(Z).any():
        z_hi = np.nanmax(
            [
                np.nanmax(Z),
                np.nanmax(z_fin[mask_fin]) if mask_fin.any() else np.nanmin(Z),
            ]
        )
        z_lo = np.nanmin(
            [
                np.nanmin(Z),
                np.nanmin(z_fin[mask_fin]) if mask_fin.any() else np.nanmax(Z),
            ]
        )
        ax.set_zlim(z_lo, z_hi)

    plt.tight_layout()
    out_path = os.path.join(out_dir, "advanced/invariant_manifold3D_log.png")
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)

    return [out_path]


# -----
# Tracking errors and end-effects
# -----


def compute_end_effect_series(
    df: pd.DataFrame, times: List[pd.Timestamp], H_vals: np.ndarray, W_star: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute end-effect diagnostics over [t0, t]:

    Returns arrays aligned to `times`:
      - rH(T) = E(T) / H(T), where E(T) = H(T) - sum(full durations of items fully contained)
      - rB(T) = B(T) / total_items_started_by_t, boundary share
      - rho(T) = T / W*(t), window/typical-duration ratio
    """
    n = len(times)
    rH = np.full(n, np.nan, dtype=float)
    rB = np.full(n, np.nan, dtype=float)
    rho = np.full(n, np.nan, dtype=float)
    if n == 0:
        return rH, rB, rho

    df = df.copy()
    df["duration_s"] = (df["end_ts"] - df["start_ts"]).dt.total_seconds()
    df_sorted_by_end = df.sort_values("end_ts")
    df_sorted_by_start = df.sort_values("start_ts")

    t0 = times[0]

    for i, t in enumerate(times):
        elapsed_s = (t - t0).total_seconds()
        if elapsed_s <= 0:
            continue

        H_T = float(H_vals[i]) if i < len(H_vals) and np.isfinite(H_vals[i]) else np.nan
        if not np.isfinite(H_T) or H_T <= 0:
            continue

        mask_full = df_sorted_by_end["end_ts"].notna() & (
            df_sorted_by_end["end_ts"] <= t
        )
        H_full = (
            float(df_sorted_by_end.loc[mask_full, "duration_s"].sum())
            if mask_full.any()
            else 0.0
        )

        E_T = max(H_T - H_full, 0.0)
        rH[i] = E_T / H_T if H_T > 0 else np.nan

        mask_started = df_sorted_by_start["start_ts"] <= t
        total_started = int(mask_started.sum())
        mask_incomplete_by_t = mask_started & (
            (df_sorted_by_start["end_ts"].isna()) | (df_sorted_by_start["end_ts"] > t)
        )
        B_T = int(mask_incomplete_by_t.sum())
        rB[i] = (B_T / total_started) if total_started > 0 else np.nan

        Wstar_t = float(W_star[i]) if i < len(W_star) else float("nan")
        rho[i] = (
            (elapsed_s / Wstar_t) if np.isfinite(Wstar_t) and Wstar_t > 0 else np.nan
        )

    return rH, rB, rho


def compute_tracking_errors(
    times: List[pd.Timestamp],
    w_vals: np.ndarray,
    lam_vals: np.ndarray,
    W_star: np.ndarray,
    lam_star: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = len(times)
    if n == 0:
        return np.array([]), np.array([]), np.array([])
    t0 = times[0]
    elapsed_seconds = np.array([(t - t0).total_seconds() for t in times], dtype=float)

    eW = np.full(n, np.nan, dtype=float)
    eLam = np.full(n, np.nan, dtype=float)

    valid_W = np.isfinite(w_vals) & np.isfinite(W_star) & (W_star > 0)
    valid_L = np.isfinite(lam_vals) & np.isfinite(lam_star) & (lam_star > 0)

    eW[valid_W] = np.abs(w_vals[valid_W] - W_star[valid_W]) / W_star[valid_W]
    eLam[valid_L] = np.abs(lam_vals[valid_L] - lam_star[valid_L]) / lam_star[valid_L]

    return eW, eLam, elapsed_seconds


def compute_coherence_score(
    eW: np.ndarray,
    eLam: np.ndarray,
    elapsed_seconds: np.ndarray,
    epsilon: float,
    horizon_seconds: float,
) -> Tuple[float, int, int]:
    ok_idx = np.isfinite(eW) & np.isfinite(eLam) & (elapsed_seconds >= horizon_seconds)
    total = int(np.sum(ok_idx))
    if total == 0:
        return float("nan"), 0, 0
    coherent = int(np.sum(np.maximum(eW[ok_idx], eLam[ok_idx]) <= epsilon))
    return coherent / total, coherent, total


def draw_dynamic_convergence_panel_with_errors(
    times: List[pd.Timestamp],
    w_vals: np.ndarray,
    lam_vals: np.ndarray,
    W_star: np.ndarray,
    lam_star: np.ndarray,
    eW: np.ndarray,
    eLam: np.ndarray,
    epsilon: Optional[float],
    title: str,
    out_path: str,
    lambda_pctl_upper: Optional[float] = None,
    lambda_pctl_lower: Optional[float] = None,
    lambda_warmup_seconds: Optional[float] = None,
    scale: Optional[DurationScale] = None,
    grid_lines: bool = True,
) -> None:
    fig, axes = plt.subplots(3, 1, figsize=(12, 9.2), sharex=True)
    duration_scale = scale or HOURS
    w_scaled = np.asarray(w_vals, dtype=float) / duration_scale.divisor
    W_star_scaled = np.asarray(W_star, dtype=float) / duration_scale.divisor
    lam_scaled = np.asarray(lam_vals, dtype=float) * duration_scale.divisor
    lam_star_scaled = np.asarray(lam_star, dtype=float) * duration_scale.divisor

    axes[0].plot(times, w_scaled, label=f"w(T) [{duration_scale.label}]")
    axes[0].plot(
        times,
        W_star_scaled,
        linestyle="--",
        label=f"W*(t) [{duration_scale.label}] (completed ≤ t)",
    )
    axes[0].set_title("w(T) vs W*(t) — dynamic")
    axes[0].set_ylabel(duration_scale.label)
    axes[0].legend()

    axes[1].plot(times, lam_scaled, label=f"Λ(T) [{duration_scale.rate_label}]")
    axes[1].plot(
        times,
        lam_star_scaled,
        linestyle="--",
        label=f"λ*(t) [{duration_scale.rate_label}] (arrivals ≤ t)",
    )
    axes[1].set_title("Λ(T) vs λ*(t) — dynamic")
    axes[1].set_ylabel(duration_scale.rate_label)
    axes[1].legend()
    _clip_axis_to_percentile(
        axes[1],
        times,
        lam_scaled,
        upper_p=lambda_pctl_upper,
        lower_p=lambda_pctl_lower,
        warmup_seconds=lambda_warmup_seconds,
    )

    axes[2].plot(times, eW, label="rel. error e_W")
    axes[2].plot(times, eLam, label="rel. error e_λ")
    if epsilon is not None:
        axes[2].axhline(epsilon, linestyle="--", label=f"ε = {epsilon:g}")
    axes[2].set_title("Relative tracking errors")
    axes[2].set_ylabel("relative error")
    axes[2].set_xlabel("Date")
    axes[2].legend()

    err = np.concatenate([eW[np.isfinite(eW)], eLam[np.isfinite(eLam)]])
    if err.size > 0:
        ub = float(np.nanpercentile(err, 99.5))
        axes[2].set_ylim(
            0.0, max(ub, (epsilon if epsilon is not None else 0.0) * 1.5 + 1e-6)
        )

    for ax in axes:
        format_date_axis(ax)
        apply_gridlines(ax, enabled=grid_lines)

    fig.suptitle(title)
    plt.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out_path)
    plt.close(fig)


def draw_dynamic_convergence_panel_with_errors_and_endeffects(
    times: List[pd.Timestamp],
    w_vals: np.ndarray,
    lam_vals: np.ndarray,
    W_star: np.ndarray,
    lam_star: np.ndarray,
    eW: np.ndarray,
    eLam: np.ndarray,
    rH: np.ndarray,
    rB: np.ndarray,
    rho: np.ndarray,
    epsilon: Optional[float],
    title: str,
    out_path: str,
    lambda_pctl_upper: Optional[float] = None,
    lambda_pctl_lower: Optional[float] = None,
    lambda_warmup_seconds: Optional[float] = None,
    scale: Optional[DurationScale] = None,
    grid_lines: bool = True,
) -> None:
    """Four-row dynamic convergence view with end-effect metrics."""
    fig, axes = plt.subplots(4, 1, figsize=(12, 12), sharex=True)
    duration_scale = scale or HOURS
    w_scaled = np.asarray(w_vals, dtype=float) / duration_scale.divisor
    W_star_scaled = np.asarray(W_star, dtype=float) / duration_scale.divisor
    lam_scaled = np.asarray(lam_vals, dtype=float) * duration_scale.divisor
    lam_star_scaled = np.asarray(lam_star, dtype=float) * duration_scale.divisor

    axes[0].plot(times, w_scaled, label=f"w(T) [{duration_scale.label}]")
    axes[0].plot(
        times,
        W_star_scaled,
        linestyle="--",
        label=f"W*(t) [{duration_scale.label}] (completed ≤ t)",
    )
    axes[0].set_title("w(T) vs W*(t) — dynamic")
    axes[0].set_ylabel(duration_scale.label)
    axes[0].legend()

    axes[1].plot(times, lam_scaled, label=f"Λ(T) [{duration_scale.rate_label}]")
    axes[1].plot(
        times,
        lam_star_scaled,
        linestyle="--",
        label=f"λ*(t) [{duration_scale.rate_label}] (arrivals ≤ t)",
    )
    axes[1].set_title("Λ(T) vs λ*(t) — dynamic")
    axes[1].set_ylabel(duration_scale.rate_label)
    axes[1].legend()
    _clip_axis_to_percentile(
        axes[1],
        times,
        lam_scaled,
        upper_p=lambda_pctl_upper,
        lower_p=lambda_pctl_lower,
        warmup_seconds=lambda_warmup_seconds,
    )

    axes[2].plot(times, eW, label="rel. error e_W")
    axes[2].plot(times, eLam, label="rel. error e_λ")
    if epsilon is not None:
        axes[2].axhline(epsilon, linestyle="--", label=f"ε = {epsilon:g}")
    axes[2].set_title("Relative tracking errors")
    axes[2].set_ylabel("relative error")
    axes[2].legend()

    err = np.concatenate([eW[np.isfinite(eW)], eLam[np.isfinite(eLam)]])
    if err.size > 0:
        ub = float(np.nanpercentile(err, 99.5))
        axes[2].set_ylim(
            0.0, max(ub, (epsilon if epsilon is not None else 0.0) * 1.5 + 1e-6)
        )

    axes[3].plot(times, rH, label="r_H(T) = E/H", alpha=0.9)
    axes[3].plot(times, rB, label="r_B(T) = B/starts", alpha=0.9)
    axes[3].set_title("End-effects: mass share and boundary share")
    axes[3].set_ylabel("share [0–1]")
    axes[3].set_ylim(0.0, 1.0)
    ax2 = axes[3].twinx()
    ax2.plot(times, rho, linestyle="--", label="ρ(T)=T/W*(t)", alpha=0.7)
    ax2.set_ylabel("ρ (window / duration)")
    lines1, labels1 = axes[3].get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    axes[3].legend(lines1 + lines2, labels1 + labels2, loc="upper right")

    for ax in axes:
        format_date_axis(ax)
        apply_gridlines(ax, enabled=grid_lines)

    fig.suptitle(title)
    plt.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out_path)
    plt.close(fig)


def plot_residence_time_convergence_error_charts(
    df: pd.DataFrame,
    args,
    chart_config: ChartConfig,
    filter_result: Optional[FilterResult],
    metrics: FlowMetricsResult,
    out_dir: str,
) -> List[str]:
    written: List[str] = []
    if len(metrics.times) == 0:
        return written

    W_star_ts, lam_star_ts = compute_elementwise_empirical_metrics(
        df, metrics.times
    ).as_tuple()
    eW_ts, eLam_ts, _ = compute_tracking_errors(
        metrics.times, metrics.w, metrics.Lambda, W_star_ts, lam_star_ts
    )
    epsilon = chart_config.epsilon
    mode_label = "complete" if not getattr(args, "incomplete", False) else "incomplete"
    lambda_pctl_upper = chart_config.lambda_pctl_upper
    lambda_pctl_lower = chart_config.lambda_pctl_lower
    lambda_warmup_seconds = chart_config.lambda_warmup_seconds
    scale = _resolve_duration_scale(chart_config)

    ts_conv_dyn3 = os.path.join(out_dir, "advanced/residence_convergence_errors.png")
    draw_dynamic_convergence_panel_with_errors(
        metrics.times,
        metrics.w,
        metrics.Lambda,
        W_star_ts,
        lam_star_ts,
        eW_ts,
        eLam_ts,
        epsilon,
        f"Residence time convergence + errors (timestamp, {mode_label})",
        ts_conv_dyn3,
        lambda_pctl_upper=lambda_pctl_upper,
        lambda_pctl_lower=lambda_pctl_lower,
        lambda_warmup_seconds=lambda_warmup_seconds,
        scale=scale,
        grid_lines=chart_config.grid_lines,
    )
    written.append(ts_conv_dyn3)

    rH_ts, rB_ts, rho_ts = compute_end_effect_series(
        df, metrics.times, metrics.H, W_star_ts
    )
    ts_conv_dyn4 = os.path.join(
        out_dir, "advanced/residence_time_convergence_errors_endeffects.png"
    )
    draw_dynamic_convergence_panel_with_errors_and_endeffects(
        metrics.times,
        metrics.w,
        metrics.Lambda,
        W_star_ts,
        lam_star_ts,
        eW_ts,
        eLam_ts,
        rH_ts,
        rB_ts,
        rho_ts,
        epsilon,
        f"Residence time convergence + errors + end-effects (timestamp, {mode_label})",
        ts_conv_dyn4,
        lambda_pctl_upper=lambda_pctl_upper,
        lambda_pctl_lower=lambda_pctl_lower,
        lambda_warmup_seconds=lambda_warmup_seconds,
        scale=scale,
        grid_lines=chart_config.grid_lines,
    )
    written.append(ts_conv_dyn4)
    return written


def plot_advanced_charts(
    df: pd.DataFrame,
    args,
    chart_config: ChartConfig,
    filter_result: Optional[FilterResult],
    metrics: FlowMetricsResult,
    out_dir: str,
) -> List[str]:
    written = []
    written += plot_llaw_manifold_3d(df, metrics, out_dir)
    written += plot_residence_time_convergence_error_charts(
        df, args, chart_config, filter_result, metrics, out_dir
    )
    return written
