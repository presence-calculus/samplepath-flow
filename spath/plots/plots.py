# -*- coding: utf-8 -*-
# Copyright (c) 2025 Krishna Kumar
# SPDX-License-Identifier: MIT
import os
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from spath.file_utils import ensure_output_dirs
from spath.filter import FilterResult
from spath.metrics import FlowMetricsResult, compute_elementwise_empirical_metrics, compute_total_active_age_series
from spath.plots.convergence import plot_convergence_charts
from spath.plots.core import draw_five_panel_column, draw_five_panel_column_with_scatter, plot_core_flow_metrics_charts
from spath.plots.helpers import add_caption, format_date_axis, _clip_axis_to_percentile


# ── Arrival departure convergence: two-panel stack ───────────────────


# ── Residence vs Sojourn: two-panel stack ────────────────────────────────────


# ------- STABILITY CHARTS -------

def plot_rate_stability_charts(
    df: pd.DataFrame,
    args,                 # kept for signature consistency
    filter_result,        # may provide .title_prefix and .display
    metrics,              # FlowMetricsResult with .times, .N, .t0, .w
    out_dir: str,
) -> List[str]:
    """
    Produce:
      - timestamp_rate_stability_n.png          (N(T)/T)
      - timestamp_rate_stability_r.png          (R(T)/T)
      - timestamp_rate_stability_stack.png      (4-row stack: N/T, R/T, λ*(T), W-coherence)

    The stacked figure has suptitle "Equilibrium and Coherence" and a caption with the filter display.
    """
    written: List[str] = []

    # Observation grid
    times = [pd.Timestamp(t) for t in metrics.times]
    if not times:
        return written

    # Elapsed hours since t0
    t0 = metrics.t0 if hasattr(metrics, "t0") and pd.notna(metrics.t0) else times[0]
    elapsed_h = np.array([(t - t0).total_seconds() / 3600.0 for t in times], dtype=float)
    denom = np.where(elapsed_h > 0.0, elapsed_h, np.nan)

    # Core rate series
    N_raw = np.asarray(metrics.N, dtype=float)
    R_raw = compute_total_active_age_series(df, times)  # hours

    with np.errstate(divide="ignore", invalid="ignore"):
        N_over_T = N_raw / denom
        R_over_T = R_raw / denom

    # Dynamic empirical series (for λ* and W*)
    W_star_ts, lam_star_ts = compute_elementwise_empirical_metrics(df, times).as_tuple()
    w_ts = np.asarray(metrics.w, dtype=float)

    # Optional display bits
    title_prefix = getattr(filter_result, "title_prefix", None)
    caption_text = getattr(filter_result, "display", None)

    # --------------------- Chart 1: N(t) sample path + N(T)/T ---------------------
    out_path_N = os.path.join(out_dir, "stability/panels/wip_growth_rate.png")
    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(10, 7.8), sharex=True)

    # Top: N(t) sample path (step plot)
    ax_top = axes[0]
    ax_top.step(times, N_raw, where='post', label='N(t)', linewidth=1.5)
    ax_top.set_ylabel('count')
    ax_top.set_title('N(t) — Sample Path')
    ax_top.legend(loc='best')
    format_date_axis(ax_top)

    # Bottom: WIP Growth Rate N(T)/T
    ax = axes[1]
    ax.plot(times, N_over_T, label='N(t)/T', linewidth=1.9, zorder=3)
    ax.axhline(0.0, linewidth=0.8, alpha=0.6, zorder=1)
    ax.axhline(1.0, linewidth=1.0, alpha=1.0, linestyle=':', zorder=1)
    format_date_axis(ax)
    ax.set_xlabel('time')
    ax.set_ylabel('rate')
    ax.set_title(f"{title_prefix}: WIP Growth Rate - N(t)/T" if title_prefix else 'WIP Growth Rate - N(t)/T')
    ax.legend(loc='best')

    finite_vals_N = N_over_T[np.isfinite(N_over_T)]
    if finite_vals_N.size:
        top = float(np.nanpercentile(finite_vals_N, 99.5))
        bot = float(np.nanmin(finite_vals_N))
        ax.set_ylim(bottom=min(0.0, bot * 1.05), top=top * 1.05)

    fig.tight_layout()
    fig.savefig(out_path_N, dpi=200)
    plt.close(fig)
    written.append(out_path_N)

    # --------------------- Chart 2: R(t) sample path + R(T)/T ---------------------
    out_path_R = os.path.join(out_dir, "stability/panels/total_age_growth_rate.png")
    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(10, 7.8), sharex=True)

    # Top: R(t) — total age of WIP at time t
    ax_top = axes[0]
    ax_top.plot(times, R_raw, label='R(t) [hours]', linewidth=1.5, zorder=3)
    ax_top.set_ylabel('hours')
    ax_top.set_title('R(t) — Total age of WIP')
    ax_top.legend(loc='best')
    format_date_axis(ax_top)

    # Bottom: Total Age Growth Rate R(T)/T
    ax = axes[1]
    ax.plot(times, R_over_T, label="R(T)/T", linewidth=1.9, zorder=3)
    ax.axhline(0.0, linewidth=0.8, alpha=0.6, zorder=1)
    ax.axhline(1.0, linewidth=1.0, alpha=1.0, linestyle=":", zorder=1)  # reference guide

    format_date_axis(ax)
    ax.set_xlabel("time")
    ax.set_ylabel("rate")
    ax.set_title(
        f"{title_prefix}: Total Age Growth Rate - R(T)/T" if title_prefix else "Total Age Growth Rate - R(T)/T")
    ax.legend(loc="best")

    finite_vals_R = R_over_T[np.isfinite(R_over_T)]
    if finite_vals_R.size:
        top = float(np.nanpercentile(finite_vals_R, 99.5))
        bot = float(np.nanmin(finite_vals_R))
        ax.set_ylim(bottom=min(0.0, bot * 1.05), top=top * 1.05)

    fig.tight_layout()
    fig.savefig(out_path_R, dpi=200)
    plt.close(fig)
    written.append(out_path_R)

    # --------------------- 4-row stack: Equilibrium and Coherence --------------
    out_path_stack = os.path.join(out_dir, "stability/rate_stability.png")
    fig, axes = plt.subplots(nrows=4, ncols=1, figsize=(12, 13.5), sharex=True)

    # Panel 1: N(T)/T
    axN = axes[0]
    axN.plot(times, N_over_T, label="N(T)/T", linewidth=1.9, zorder=3)
    axN.axhline(0.0, linewidth=0.8, alpha=0.6, zorder=1)
    axN.axhline(1.0, linewidth=1.0, alpha=1.0, linestyle=":", zorder=1)
    format_date_axis(axN)
    axN.set_ylabel("rate")
    axN.set_title("WIP Growth Rate: N(T)/T")
    axN.legend(loc="best")
    if finite_vals_N.size:
        topN = float(np.nanpercentile(finite_vals_N, 99.5))
        botN = float(np.nanmin(finite_vals_N))
        axN.set_ylim(bottom=min(0.0, botN * 1.05), top=topN * 1.05)

    # Panel 2: R(T)/T
    axR = axes[1]
    axR.plot(times, R_over_T, label="R(T)/T", linewidth=1.9, zorder=3)
    axR.axhline(0.0, linewidth=0.8, alpha=0.6, zorder=1)
    axR.axhline(1.0, linewidth=1.0, alpha=1.0, linestyle=":", zorder=1)
    format_date_axis(axR)
    axR.set_ylabel("rate")
    axR.set_title("Total Age Growth Rate: R(T)/T")
    axR.legend(loc="best")
    if finite_vals_R.size:
        topR = float(np.nanpercentile(finite_vals_R, 99.5))
        botR = float(np.nanmin(finite_vals_R))
        axR.set_ylim(bottom=min(0.0, botR * 1.05), top=topR * 1.05)

    # Panel 3: λ*(T)
    axLam = axes[2]
    axLam.plot(times, lam_star_ts, label="λ*(T) [1/hr]", linewidth=1.9, zorder=3)
    axLam.axhline(0.0, linewidth=0.8, alpha=0.6, zorder=1)
    format_date_axis(axLam)
    axLam.set_ylabel("[1/hr]")
    axLam.set_title("λ*(T) — running arrival rate")
    axLam.legend(loc="best")
    # Clip like other charts
    try:
        _clip_axis_to_percentile(
            axLam, times, lam_star_ts,
            upper_p=getattr(args, "lambda_pctl", None),
            lower_p=getattr(args, "lambda_lower_pctl", None),
            warmup_hours=float(getattr(args, "lambda_warmup", 0.0) or 0.0),
        )
    except Exception:
        pass

    # Panel 4: W-coherence overlay
    axW = axes[3]
    axW.plot(times, w_ts,        label="w(T) [hrs] (finite-window)", linewidth=1.9, zorder=3)
    axW.plot(times, W_star_ts,   label="W*(T) [hrs] (completed mean)", linewidth=1.9, linestyle="--", zorder=3)
    axW.axhline(0.0, linewidth=0.8, alpha=0.6, zorder=1)
    format_date_axis(axW)
    axW.set_xlabel("time")
    axW.set_ylabel("hours")
    axW.set_title("w(T) vs W*(T) — coherence")
    axW.legend(loc="best")

    # Subtitle + caption
    fig.suptitle("Equilibrium and Coherence", fontsize=14, y=0.98)
    try:
        if caption_text:
            add_caption(fig, caption_text)
    except Exception:
        pass

    fig.tight_layout(rect=(0, 0.06, 1, 0.96))
    fig.savefig(out_path_stack, dpi=200)
    plt.close(fig)
    written.append(out_path_stack)

    return written


def plot_stability_charts(
    df: pd.DataFrame,
    args,
    filter_result: Optional[FilterResult],
    metrics: FlowMetricsResult,
    out_dir: str,
) -> List[str]:
    written = []
    written += plot_rate_stability_charts(df, args, filter_result, metrics, out_dir)
    return written

# ---- ADVANCED CHARTS -----
def plot_llaw_manifold_3d(
    df,
    metrics,                           # FlowMetricsResult
    out_dir: str,
    title: str = "Manifold view: L = Λ · w (log-space plane z = x + y)",
    caption: Optional[str] = None,
    figsize: Tuple[int, int] = (9, 7),
    elev: float = 28.0,
    azim: float = -135.0,
    alpha_surface: float = 0.22,       # kept in signature; not used explicitly
    wireframe_stride: int = 6,
    point_size: int = 16,
) -> List[str]:
    """Log-linear manifold: z = x + y with x=log Λ, y=log w, z=log L.
    Plots only the finite-time trajectory on a filled plane (no empirical series).
    """
    import os
    import numpy as np
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

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
    x_fin = _safe_log(Lam_vals)   # log Λ
    y_fin = _safe_log(w_vals)     # log w
    z_fin = _safe_log(L_vals)     # log L
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
    ax.plot_surface(
        X, Y, Z,
        color="dimgray",
        alpha=0.5,
        linewidth=0,
        antialiased=True
    )

    # ---- finite-time trajectory (lies on the plane) -------------------------
    if mask_fin.any():
        ax.plot(x_fin[mask_fin], y_fin[mask_fin], z_fin[mask_fin],
                lw=1.6, label="(log Λ, log w, log L)")

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
        z_hi = np.nanmax([np.nanmax(Z), np.nanmax(z_fin[mask_fin]) if mask_fin.any() else np.nanmin(Z)])
        z_lo = np.nanmin([np.nanmin(Z), np.nanmin(z_fin[mask_fin]) if mask_fin.any() else np.nanmax(Z)])
        ax.set_zlim(z_lo, z_hi)

    plt.tight_layout()
    out_path = os.path.join(out_dir, "advanced/invariant_manifold3D_log.png")
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)

    return [out_path]

def plot_advanced_charts(
    df: pd.DataFrame,
    args,
    filter_result: Optional[FilterResult],
    metrics: FlowMetricsResult,
    out_dir: str,
) -> List[str]:
    written = []
    written += plot_llaw_manifold_3d(df, metrics, out_dir)
    return written

# ---- MISC CHARTS
def plot_five_column_stacks(df, args, filter_result, metrics, out_dir):
    t_scatter_times = df["start_ts"].tolist()
    t_scatter_vals = df["duration_hr"].to_numpy()
    written = []

    col_ts5 = os.path.join(out_dir, 'misc/timestamp_stack_with_A.png')
    draw_five_panel_column(metrics.times, metrics.N, metrics.Lambda, metrics.Lambda, metrics.w, metrics.A,
                           f'Finite-window metrics incl. A(T) (timestamp, {filter_result.label})', col_ts5,
                           scatter_times=t_scatter_times, scatter_values=t_scatter_vals,
                           lambda_pctl_upper=args.lambda_pctl, lambda_pctl_lower=args.lambda_lower_pctl,
                           lambda_warmup_hours=args.lambda_warmup)
    written.append(col_ts5)


    col_ts5s = os.path.join(out_dir, 'misc/timestamp_stack_with_scatter.png')
    draw_five_panel_column_with_scatter(metrics.times, metrics.N, metrics.L, metrics.Lambda, metrics.w,
                                        f'Finite-window metrics with w(T) plain + w(T)+scatter (timestamp, {filter_result.label})',
                                        col_ts5s,
                                        scatter_times=t_scatter_times, scatter_values=t_scatter_vals,
                                        lambda_pctl_upper=args.lambda_pctl,
                                        lambda_pctl_lower=args.lambda_lower_pctl,
                                        lambda_warmup_hours=args.lambda_warmup)
    written.append(col_ts5s)

    return written

def plot_misc_charts(df: pd.DataFrame,
    args,
    filter_result: Optional[FilterResult],
    metrics: FlowMetricsResult,
    out_dir: str,
) -> List[str]:
    # 5-panel stacks including scatter
    return  plot_five_column_stacks(df, args, filter_result, metrics, out_dir)

# ---- MAIN Driver -----

def produce_all_charts(df, csv_path, args, filter_result, metrics, empirical_metrics):
    out_dir = ensure_output_dirs(csv_path, output_dir=args.output_dir, clean=args.clean)
    written: List[str] = []
    # create plots
    written += plot_core_flow_metrics_charts(df, args, filter_result, metrics, out_dir)
    written += plot_convergence_charts(df, args, filter_result, metrics, empirical_metrics, out_dir)
    written += plot_stability_charts(df, args, filter_result, metrics, out_dir)
    written += plot_advanced_charts(df, args, filter_result, metrics, out_dir)
    written += plot_misc_charts(df, args, filter_result, metrics, out_dir)
    return written
