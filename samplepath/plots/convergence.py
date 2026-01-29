# -*- coding: utf-8 -*-
# Copyright (c) 2025 Krishna Kumar
# SPDX-License-Identifier: MIT
from dataclasses import dataclass
import os
from typing import List, Optional, Tuple

from matplotlib import pyplot as plt
import numpy as np
import pandas as pd

from samplepath.filter import FilterResult
from samplepath.metrics import (
    ElementWiseEmpiricalMetrics,
    FlowMetricsResult,
    compute_coherence_score,
    compute_elementwise_empirical_metrics,
    compute_end_effect_series,
    compute_tracking_errors,
)
from samplepath.plots.chart_config import ChartConfig
from samplepath.plots.core import CFDPanel
from samplepath.plots.figure_context import (
    FigureDecorSpec,
    LayoutSpec,
    _first_axis,
    figure_context,
    layout_context,
)
from samplepath.plots.helpers import (
    _clip_axis_to_percentile,
    add_caption,
    build_event_overlays,
    format_and_save,
    format_date_axis,
    init_fig_ax,
    render_line_chart,
    render_scatter_chart,
    resolve_caption,
)


@dataclass
class ProcessTimeConvergencePanel:
    show_title: bool = True
    title: str = "Process Time Convergence"
    with_event_marks: bool = False

    def render(
        self,
        ax,
        times: List[pd.Timestamp],
        w_vals: np.ndarray,
        w_prime_vals: np.ndarray,
        w_star_vals: np.ndarray,
        *,
        arrival_times: Optional[List[pd.Timestamp]] = None,
        departure_times: Optional[List[pd.Timestamp]] = None,
    ) -> None:
        arrivals = arrival_times or []
        departures = departure_times or []
        w_overlays = (
            build_event_overlays(
                times,
                w_vals,
                arrivals,
                [],
                drop_lines_for_arrivals=True,
                drop_lines_for_departures=False,
            )
            if self.with_event_marks
            else None
        )
        w_prime_overlays = (
            build_event_overlays(
                times,
                w_prime_vals,
                [],
                departures,
                drop_lines_for_arrivals=False,
                drop_lines_for_departures=True,
            )
            if self.with_event_marks
            else None
        )
        w_star_overlays = (
            build_event_overlays(
                times,
                w_star_vals,
                [],
                departures,
                drop_lines_for_arrivals=False,
                drop_lines_for_departures=True,
            )
            if self.with_event_marks
            else None
        )
        render_line_chart(
            ax,
            times,
            w_vals,
            label="w(T) [hrs]",
            color="tab:blue",
            overlays=w_overlays,
        )
        render_line_chart(
            ax,
            times,
            w_prime_vals,
            label="w'(T) [hrs]",
            color="tab:orange",
            overlays=w_prime_overlays,
        )
        render_line_chart(
            ax,
            times,
            w_star_vals,
            label="W*(t) [hrs]",
            color="tab:green",
            overlays=w_star_overlays,
        )
        if self.show_title:
            ax.set_title(self.title)
        ax.set_ylabel("hours")
        ax.legend()

    def plot(
        self,
        metrics: FlowMetricsResult,
        empirical_metrics: ElementWiseEmpiricalMetrics,
        filter_result: Optional[FilterResult],
        chart_config: ChartConfig,
        out_dir: str,
    ) -> str:
        unit = metrics.freq if metrics.freq else "timestamp"
        caption = resolve_caption(filter_result)
        with figure_context(
            chart_config=chart_config,
            nrows=1,
            ncols=1,
            caption=caption,
            unit=unit,
            out_dir=out_dir,
            subdir="convergence/panels",
            base_name="process_time_convergence",
        ) as (
            _,
            axes,
            resolved_out_path,
        ):
            ax = _first_axis(axes)
            self.render(
                ax,
                metrics.times,
                metrics.w,
                metrics.w_prime,
                empirical_metrics.W_star,
                arrival_times=metrics.arrival_times,
                departure_times=metrics.departure_times,
            )
        return resolved_out_path


@dataclass
class SojournTimeScatterPanel:
    show_title: bool = True
    title: str = "Sojourn Time vs Residence Times"
    with_event_marks: bool = False

    def render(
        self,
        ax,
        times: List[pd.Timestamp],
        w_vals: np.ndarray,
        w_prime_vals: np.ndarray,
        departure_times: List[pd.Timestamp],
        sojourn_vals: np.ndarray,
    ) -> None:
        departures = departure_times or []
        w_overlays = (
            build_event_overlays(
                times,
                w_vals,
                [],
                departures,
                drop_lines_for_arrivals=False,
                drop_lines_for_departures=True,
            )
            if self.with_event_marks
            else None
        )
        w_prime_overlays = (
            build_event_overlays(
                times,
                w_prime_vals,
                [],
                departures,
                drop_lines_for_arrivals=False,
                drop_lines_for_departures=True,
            )
            if self.with_event_marks
            else None
        )
        render_scatter_chart(
            ax,
            departures,
            sojourn_vals,
            label="Sojourn time (departures)",
            color="tab:purple",
        )
        render_line_chart(
            ax,
            times,
            w_vals,
            label="w(T) [hrs]",
            color="tab:blue",
            overlays=w_overlays,
        )
        render_line_chart(
            ax,
            times,
            w_prime_vals,
            label="w'(T) [hrs]",
            color="tab:orange",
            overlays=w_prime_overlays,
        )
        if self.show_title:
            ax.set_title(self.title)
        ax.set_ylabel("Time [hrs]")
        ax.legend()

    def plot(
        self,
        metrics: FlowMetricsResult,
        empirical_metrics: ElementWiseEmpiricalMetrics,
        filter_result: Optional[FilterResult],
        chart_config: ChartConfig,
        out_dir: str,
    ) -> str:
        unit = metrics.freq if metrics.freq else "timestamp"
        caption = resolve_caption(filter_result)
        with figure_context(
            chart_config=chart_config,
            nrows=1,
            ncols=1,
            caption=caption,
            unit=unit,
            out_dir=out_dir,
            subdir="convergence/panels",
            base_name="residence_time_sojourn_time_scatter_plot",
        ) as (
            _,
            axes,
            resolved_out_path,
        ):
            ax = _first_axis(axes)
            self.render(
                ax,
                metrics.times,
                metrics.w,
                metrics.w_prime,
                metrics.departure_times,
                empirical_metrics.sojourn_vals,
            )
        return resolved_out_path


def draw_cumulative_arrival_rate_convergence_panel(
    times: List[pd.Timestamp],
    w_vals: np.ndarray,
    lam_vals: np.ndarray,
    W_star: np.ndarray,
    lam_star: np.ndarray,
    title: str,
    out_path: str,
    lambda_pctl_upper: Optional[float] = None,
    lambda_pctl_lower: Optional[float] = None,
    lambda_warmup_hours: Optional[float] = None,
    caption: Optional[str] = None,
) -> None:
    fig, ax = plt.subplots(figsize=(12, 6.5))

    ax.plot(times, lam_vals, label="Λ(T) [1/hr]")
    ax.plot(times, lam_star, linestyle="--", label="λ*(t) [1/hr] (arrivals ≤ t)")
    ax.set_title("Λ(T) vs λ*(t)  — arrival rate")
    ax.set_ylabel("1/hr")
    ax.set_xlabel("Date")
    ax.legend()

    _clip_axis_to_percentile(
        ax,
        times,
        lam_vals,
        upper_p=lambda_pctl_upper,
        lower_p=lambda_pctl_lower,
        warmup_hours=lambda_warmup_hours,
    )

    format_date_axis(ax)

    fig.suptitle(title)
    if caption:
        add_caption(fig, caption)  # uses the helper you already have
    fig.tight_layout(rect=(0, 0, 1, 1))

    fig.savefig(out_path)
    plt.close(fig)


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
    lambda_warmup_hours: Optional[float] = None,
) -> None:
    fig, axes = plt.subplots(3, 1, figsize=(12, 9.2), sharex=True)

    axes[0].plot(times, w_vals, label="w(T) [hrs]")
    axes[0].plot(times, W_star, linestyle="--", label="W*(t) [hrs] (completed ≤ t)")
    axes[0].set_title("w(T) vs W*(t) — dynamic")
    axes[0].set_ylabel("hours")
    axes[0].legend()

    axes[1].plot(times, lam_vals, label="Λ(T) [1/hr]")
    axes[1].plot(times, lam_star, linestyle="--", label="λ*(t) [1/hr] (arrivals ≤ t)")
    axes[1].set_title("Λ(T) vs λ*(t) — dynamic")
    axes[1].set_ylabel("1/hr")
    axes[1].legend()
    _clip_axis_to_percentile(
        axes[1],
        times,
        lam_vals,
        upper_p=lambda_pctl_upper,
        lower_p=lambda_pctl_lower,
        warmup_hours=lambda_warmup_hours,
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
    lambda_warmup_hours: Optional[float] = None,
) -> None:
    """Four-row dynamic convergence view with end-effect metrics."""
    fig, axes = plt.subplots(4, 1, figsize=(12, 12), sharex=True)

    axes[0].plot(times, w_vals, label="w(T) [hrs]")
    axes[0].plot(times, W_star, linestyle="--", label="W*(t) [hrs] (completed ≤ t)")
    axes[0].set_title("w(T) vs W*(t) — dynamic")
    axes[0].set_ylabel("hours")
    axes[0].legend()

    axes[1].plot(times, lam_vals, label="Λ(T) [1/hr]")
    axes[1].plot(times, lam_star, linestyle="--", label="λ*(t) [1/hr] (arrivals ≤ t)")
    axes[1].set_title("Λ(T) vs λ*(t) — dynamic")
    axes[1].set_ylabel("1/hr")
    axes[1].legend()
    _clip_axis_to_percentile(
        axes[1],
        times,
        lam_vals,
        upper_p=lambda_pctl_upper,
        lower_p=lambda_pctl_lower,
        warmup_hours=lambda_warmup_hours,
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

    fig.suptitle(title)
    plt.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out_path)
    plt.close(fig)


def draw_arrival_departure_convergence_stack(
    times: List[pd.Timestamp],
    arrivals_cum: np.ndarray,  # A(t)  = metrics.Arrivals
    departures_cum: np.ndarray,  # D(t)  = metrics.Departures
    lambda_cum_rate: np.ndarray,  # Λ(T)  = metrics.Lambda [1/hr]
    title: str,
    out_path: str,
    lambda_pctl_upper: Optional[float] = None,
    lambda_pctl_lower: Optional[float] = None,
    lambda_warmup_hours: Optional[float] = None,
    caption: Optional[str] = None,
) -> None:
    """
    Two stacked charts sharing the x-axis:

      (1) Cumulative Arrivals vs Cumulative Departures
          Stability: A(t) and D(t) grow together; gap A-D = N(t) stays bounded.

      (2) Λ(T) vs θ(T)  (arrival rate vs throughput rate)
          θ(T) := D(T) / (T - t0) [1/hr], masked after the last departure to avoid the
          idle-tail artifact where the ratio would decay toward 0.
    """
    # ---- Compute elapsed hours and throughput rate θ(T) ----------------------
    n = len(times)
    if n > 0:
        t0 = times[0]
        elapsed_h = np.array(
            [(t - t0).total_seconds() / 3600.0 for t in times], dtype=float
        )
    else:
        elapsed_h = np.array([], dtype=float)

    with np.errstate(divide="ignore", invalid="ignore"):
        theta_rate = np.where(elapsed_h > 0.0, departures_cum / elapsed_h, np.nan)

    # Find last departure index (last time D(t) increases)
    last_dep_idx = -1
    if len(departures_cum) > 0:
        d = np.asarray(departures_cum, dtype=float)
        inc = np.flatnonzero(np.diff(d, prepend=d[0]) > 0)
        if inc.size > 0:
            last_dep_idx = int(inc.max())

    # Mask θ(T) after the last departure (prevents idle tail from misleading viewers)
    if last_dep_idx >= 0 and last_dep_idx + 1 < n:
        theta_rate[last_dep_idx + 1 :] = np.nan

    # ---- Figure ----------------------------------------------------------------
    fig, axes = plt.subplots(2, 1, figsize=(12, 6.5), sharex=True)

    # Panel 1: cumulative counts (step plots)
    CFDPanel().render(axes[0], times, arrivals_cum, departures_cum)
    axes[0].legend(loc="best")
    format_date_axis(axes[0])

    # Panel 2: rates (Λ vs θ), with house percentile clipping on Λ
    axes[1].plot(times, lambda_cum_rate, label="Λ(T) [1/hr]")
    axes[1].plot(times, theta_rate, label="θ(T) = D(T)/elapsed [1/hr]")
    axes[1].set_title("Arrival Rate Λ(T) vs Departure Rate θ(T)")
    axes[1].set_ylabel("1/hr")
    axes[1].legend(loc="best")
    _clip_axis_to_percentile(
        axes[1],
        times,
        lambda_cum_rate,
        upper_p=lambda_pctl_upper,
        lower_p=lambda_pctl_lower,
        warmup_hours=(lambda_warmup_hours or 0.0),
    )
    format_date_axis(axes[1])

    fig.suptitle(title)
    if caption:
        fig.text(0.5, 0.01, caption, ha="center", va="bottom", fontsize=9)
        plt.tight_layout(rect=(0, 0.03, 1, 0.96))
    else:
        plt.tight_layout(rect=(0, 0, 1, 0.96))

    fig.savefig(out_path)
    plt.close(fig)


def plot_arrival_rate_convergence(
    args,
    filter_result: Optional[FilterResult],
    metrics: FlowMetricsResult,
    empirical_metrics: ElementWiseEmpiricalMetrics,
    out_dir: str,
) -> List[str]:
    """
    Inputs expected from FlowMetricsResult:
      - metrics.times                : List[pd.Timestamp]
      - metrics.Arrivals             : cumulative arrivals A(t)
      - metrics.Departures           : cumulative departures D(t)
      - metrics.Lambda               : Cumulative Arrival Rate Λ(T) [1/hr]
    Returns list of written image paths.
    """

    caption = filter_result.display if filter_result else None

    pctl_upper = getattr(args, "lambda_pctl", None)
    pctl_lower = getattr(args, "lambda_lower_pctl", None)
    warmup_hrs = getattr(args, "lambda_warmup", None)

    eq_path = os.path.join(out_dir, "convergence/arrival_departure_equilibrium.png")
    draw_arrival_departure_convergence_stack(
        metrics.times,
        metrics.Arrivals,
        metrics.Departures,
        metrics.Lambda,
        title="Flow Equilibrium: Arrival/Departure Convergence",
        out_path=eq_path,
        lambda_pctl_upper=pctl_upper,
        lambda_pctl_lower=pctl_lower,
        lambda_warmup_hours=warmup_hrs,
        caption=caption,
    )
    lambda_path = os.path.join(
        out_dir, "convergence/panels/arrival_rate_convergence.png"
    )
    draw_cumulative_arrival_rate_convergence_panel(
        metrics.times,
        metrics.w,
        metrics.Lambda,
        empirical_metrics.W_star,
        empirical_metrics.lam_star,
        title="Flow Equilibrium: Arrival/Departure Convergence",
        out_path=lambda_path,
        lambda_pctl_upper=pctl_upper,
        lambda_pctl_lower=pctl_lower,
        lambda_warmup_hours=warmup_hrs,
        caption=caption,
    )
    return [eq_path, lambda_path]


def plot_residence_time_sojourn_time_coherence_charts(
    df, args, filter_result, metrics, out_dir
):
    # Empirical targets & dynamic baselines
    horizon_days = args.horizon_days
    epsilon = args.epsilon
    lambda_pctl_upper = args.lambda_pctl
    lambda_pctl_lower = args.lambda_lower_pctl
    lambda_warmup_hours = args.lambda_warmup
    mode_label = filter_result.label

    written: List[str] = []

    if len(metrics.times) > 0:
        W_star_ts, lam_star_ts = compute_elementwise_empirical_metrics(
            df, metrics.times
        ).as_tuple()
    else:
        W_star_ts = np.array([])
        lam_star_ts = np.array([])
    # Relative errors & coherence
    eW_ts, eLam_ts, elapsed_ts = compute_tracking_errors(
        metrics.times, metrics.w, metrics.Lambda, W_star_ts, lam_star_ts
    )
    coh_summary_lines: List[str] = []
    if epsilon is not None and horizon_days is not None:
        h_hrs = float(horizon_days) * 24.0
        sc_ts, ok_ts, tot_ts = compute_coherence_score(
            eW_ts, eLam_ts, elapsed_ts, float(epsilon), h_hrs
        )
        coh_summary_lines.append(
            f"Coherence (timestamp): eps={epsilon:g}, H={horizon_days:g}d -> {ok_ts}/{tot_ts} ({(sc_ts * 100 if sc_ts == sc_ts else 0):.1f}%)"
        )
    # Convergence diagnostics (timestamp)
    if len(metrics.times) > 0:

        ts_conv_dyn3 = os.path.join(
            out_dir, "advanced/residence_convergence_errors.png"
        )
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
            lambda_warmup_hours=lambda_warmup_hours,
        )
        written.append(ts_conv_dyn3)
    # --- End-effect diagnostics ---
    rH_ts, rB_ts, rho_ts = (
        compute_end_effect_series(df, metrics.times, metrics.H, W_star_ts)
        if len(metrics.times) > 0
        else (np.array([]), np.array([]), np.array([]))
    )
    if len(metrics.times) > 0:
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
            lambda_warmup_hours=lambda_warmup_hours,
        )
        written.append(ts_conv_dyn4)

    return written


def plot_residence_vs_sojourn_stack(
    filter_result: Optional[FilterResult],
    metrics: FlowMetricsResult,
    empirical_metrics: ElementWiseEmpiricalMetrics,
    chart_config: ChartConfig,
    out_dir: str,
) -> List[str]:
    caption = resolve_caption(filter_result)

    layout = LayoutSpec(nrows=2, ncols=1, figsize=(12.0, 6.5), sharex=True)
    decor = FigureDecorSpec(
        suptitle="Process Time Convergence Stack",
        suptitle_y=0.97,
        caption=caption,
        caption_position="top",
        caption_y=0.945,
        tight_layout=True,
        tight_layout_rect=(0, 0, 1, 0.96),
    )
    unit = metrics.freq if metrics.freq else "timestamp"
    with layout_context(
        chart_config=chart_config,
        layout=layout,
        decor=decor,
        unit=unit,
        format_targets="bottom_row",
        format_axis_fn=format_date_axis,
        out_dir=out_dir,
        subdir="convergence",
        base_name="process_time_convergence_stack",
    ) as (_, axes, resolved_out_path):
        flat_axes = axes if not isinstance(axes, np.ndarray) else axes.ravel()
        ProcessTimeConvergencePanel(
            with_event_marks=chart_config.with_event_marks
        ).render(
            flat_axes[0],
            metrics.times,
            metrics.w,
            metrics.w_prime,
            empirical_metrics.W_star,
            arrival_times=metrics.arrival_times,
            departure_times=metrics.departure_times,
        )
        SojournTimeScatterPanel(with_event_marks=chart_config.with_event_marks).render(
            flat_axes[1],
            metrics.times,
            metrics.w,
            metrics.w_prime,
            metrics.departure_times,
            empirical_metrics.sojourn_vals,
        )
    return [resolved_out_path]


def draw_ll_scatter_coherence(
    L_vals: np.ndarray,  # metrics.L (avg number-in-system)
    lam_star: np.ndarray,  # λ*(t) [1/hr] aligned to metrics.times
    W_star_hours: np.ndarray,  # W*(t) [hrs] aligned to metrics.times
    times: List[pd.Timestamp],  # metrics.times
    epsilon: float,  # tolerance for "within band"
    out_png: str,
    title: str = "Sample Path Coherence",
    caption: Optional[str] = None,
    horizon_days: float = 0.0,  # require elapsed >= horizon_days
) -> Tuple[float, int, int]:
    """
    Scatter points x=L(T) vs y=λ*(t)·W*(t), draw x=y and an ε relative band:
        (1-ε)·x  <=  y  <=  (1+ε)·x
    Return (score, ok_count, total_count) using only points with elapsed >= horizon_hours.

    Notes:
      • Relative band makes the tolerance scale with magnitude.
      • Also robust to zeros by ignoring points where x <= 0 or y is NaN/inf.
    """
    # Build y = λ*(t)·W*(t). W* is in HOURS, λ* in 1/hr so product is dimensionless (matches L).
    y_vals = lam_star * W_star_hours
    x_vals = np.asarray(L_vals, dtype=float)

    n = len(times)
    if n > 0:
        t0 = times[0]
        elapsed_h = np.array(
            [(t - t0).total_seconds() / 3600.0 for t in times], dtype=float
        )
    else:
        elapsed_h = np.array([], dtype=float)

    # Mask: finite and past horizon
    finite_mask = np.isfinite(x_vals) & np.isfinite(y_vals) & (x_vals > 0.0)
    horizon_hours = float(horizon_days) * 24.0
    if horizon_hours and horizon_hours > 0.0:
        finite_mask &= elapsed_h >= float(horizon_hours)

    X = x_vals[finite_mask]
    Y = y_vals[finite_mask]

    # Coherence test: inside relative epsilon tube around x=y
    # Equivalent to |Y/X - 1| <= ε  (avoid division-by-zero via x>0 mask above)
    rel_err = np.abs(Y / X - 1.0)
    ok_mask = rel_err <= float(epsilon)
    ok_count = int(np.count_nonzero(ok_mask))
    total_count = int(X.size)
    score = (ok_count / total_count) if total_count > 0 else float("nan")

    # ---- Plot
    fig, ax = init_fig_ax(figsize=(7.5, 6.0))

    # Scatter (only evaluated points)
    ax.scatter(X, Y, s=16, alpha=0.7, label="Points: (L(T), λ*(t)·W*(t))")

    # Diagonal x=y
    if X.size > 0:
        x_min, x_max = float(np.nanmin(X)), float(np.nanmax(X))
        pad = 0.05 * (x_max - x_min if x_max > x_min else 1.0)
        x_line = np.linspace(max(0.0, x_min - pad), x_max + pad, 200)
        ax.plot(x_line, x_line, linewidth=1.2, label="x = y")

        # ε-band: (1-ε)x .. (1+ε)x
        ax.plot(
            x_line,
            (1.0 + float(epsilon)) * x_line,
            linewidth=0.8,
            linestyle="--",
            label=f"y=(1+ε)x",
        )
        ax.plot(
            x_line,
            (1.0 - float(epsilon)) * x_line,
            linewidth=0.8,
            linestyle="--",
            label=f"y=(1−ε)x",
        )
        ax.fill_between(
            x_line,
            (1.0 - float(epsilon)) * x_line,
            (1.0 + float(epsilon)) * x_line,
            alpha=0.08,
        )

    ax.set_xlabel("L(T)  (average number in system)")
    ax.set_ylabel("λ*(t)·W*(t)")
    ax.set_title(title)
    ax.legend(loc="best")

    # annotate score
    if total_count > 0:
        ax.text(
            0.02,
            0.98,
            f"ε={epsilon:.3g}, horizon={horizon_days:.1f}d:  {ok_count}/{total_count}  ({score*100:.1f}%)",
            ha="left",
            va="top",
            transform=ax.transAxes,
            fontsize=9,
        )

    if caption:
        fig.text(0.5, 0.01, caption, ha="center", va="bottom", fontsize=9)
        plt.tight_layout(rect=(0, 0.03, 1, 0.96))
    else:
        plt.tight_layout()

    fig.savefig(out_png)
    plt.close(fig)

    return score, ok_count, total_count


def plot_sample_path_convergence(
    df: pd.DataFrame,
    args,
    filter_result: Optional[FilterResult],
    metrics: FlowMetricsResult,
    out_dir: str,
) -> List[str]:
    """

    Uses:
      • metrics.L                : L(T) series (dimensionless)
      • metrics.times            : timestamps
      • compute_dynamic_empirical_series(df, metrics.times) -> (W*, λ*)
        - W* in HOURS, λ* in 1/hr

    CLI-style knobs (optional in args):
      • args.epsilon   : default 0.05
      • args.horizon_day : horizon in days (default 28)

    Outputs:
      • PNG: timestamp_sample_path_coherence.png

    """

    caption = filter_result.display if filter_result else None

    # derive W*(t), λ*(t) aligned to times
    if len(metrics.times) > 0:
        W_star_hours, lam_star = compute_elementwise_empirical_metrics(
            df, metrics.times
        ).as_tuple()
    else:
        W_star_hours = np.array([])
        lam_star = np.array([])

    epsilon = getattr(args, "epsilon", 0.05)
    horizon_days = getattr(args, "horizon_days", 28)

    png_path = os.path.join(out_dir, "sample_path_convergence.png")
    score, ok_count, total_count = draw_ll_scatter_coherence(
        metrics.L,
        lam_star,
        W_star_hours,
        metrics.times,
        epsilon=epsilon,
        out_png=png_path,
        title="Sample Path Convergence: L(T) vs λ*(t)·W*(t)",
        caption=caption,
        horizon_days=horizon_days,
    )

    # wri
    if np.isnan(score):
        print(
            f"Sample Path Convergence: ε={epsilon}, H={horizon_days}d -> n/a (no valid points)\n"
        )
    else:
        print(
            f"Sample Path Convergence: ε={epsilon}, H={horizon_days}d -> "
            f"{ok_count}/{total_count} ({score*100:.1f}%)\n"
        )
    return [png_path]


def plot_convergence_charts(
    df: pd.DataFrame,
    args,
    filter_result: Optional[FilterResult],
    metrics: FlowMetricsResult,
    empirical_metrics: ElementWiseEmpiricalMetrics,
    out_dir: str,
) -> List[str]:
    written = []
    chart_config = ChartConfig.init_from_args(args)

    written += plot_arrival_rate_convergence(
        args, filter_result, metrics, empirical_metrics, out_dir
    )

    written.append(
        ProcessTimeConvergencePanel(
            with_event_marks=chart_config.with_event_marks
        ).plot(metrics, empirical_metrics, filter_result, chart_config, out_dir)
    )

    written.append(
        SojournTimeScatterPanel(with_event_marks=chart_config.with_event_marks).plot(
            metrics, empirical_metrics, filter_result, chart_config, out_dir
        )
    )

    written += plot_residence_time_sojourn_time_coherence_charts(
        df, args, filter_result, metrics, out_dir
    )

    written += plot_residence_vs_sojourn_stack(
        filter_result, metrics, empirical_metrics, chart_config, out_dir
    )

    written += plot_sample_path_convergence(df, args, filter_result, metrics, out_dir)

    return written
