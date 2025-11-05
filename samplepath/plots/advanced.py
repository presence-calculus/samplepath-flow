# -*- coding: utf-8 -*-
# Copyright (c) 2025 Krishna Kumar
# SPDX-License-Identifier: MIT
from __future__ import annotations

from typing import Optional, Tuple, List

import pandas as pd

from samplepath.filter import FilterResult
from samplepath.metrics import FlowMetricsResult


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
