# -*- coding: utf-8 -*-
# Copyright (c) 2025 Krishna Kumar
# SPDX-License-Identifier: MIT
from dataclasses import dataclass
from typing import List, Optional, Tuple

from matplotlib import pyplot as plt
import numpy as np
import pandas as pd

from samplepath.filter import FilterResult
from samplepath.metrics import (
    ElementWiseEmpiricalMetrics,
    FlowMetricsResult,
    MetricDerivations,
)
from samplepath.plots.chart_config import ChartConfig, ColorConfig
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
    build_event_overlays,
    format_date_axis,
    render_line_chart,
    render_scatter_chart,
    resolve_caption,
)
from samplepath.utils.duration_scale import HOURS, DurationScale


def _resolve_duration_scale(chart_config: ChartConfig) -> DurationScale:
    return chart_config.duration_scale or HOURS


@dataclass
class ProcessTimeConvergencePanel:
    show_title: bool = True
    title: str = "Process Time Convergence"
    with_event_marks: bool = False
    sampling_frequency: Optional[str] = None

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
        scale: Optional[DurationScale] = None,
    ) -> None:
        duration_scale = scale or HOURS
        w_scaled = np.asarray(w_vals, dtype=float) / duration_scale.divisor
        w_prime_scaled = np.asarray(w_prime_vals, dtype=float) / duration_scale.divisor
        w_star_scaled = np.asarray(w_star_vals, dtype=float) / duration_scale.divisor
        arrivals = arrival_times or []
        departures = departure_times or []
        w_overlays = (
            build_event_overlays(
                times,
                w_scaled,
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
                w_prime_scaled,
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
                w_star_scaled,
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
            w_scaled,
            label=f"w(T) [{duration_scale.label}]",
            color="tab:blue",
            overlays=w_overlays,
            sampling_frequency=self.sampling_frequency,
        )
        render_line_chart(
            ax,
            times,
            w_prime_scaled,
            label=f"w'(T) [{duration_scale.label}]",
            color="tab:orange",
            overlays=w_prime_overlays,
            sampling_frequency=self.sampling_frequency,
        )
        render_line_chart(
            ax,
            times,
            w_star_scaled,
            label=f"W*(t) [{duration_scale.label}]",
            color=ColorConfig.departure_color,
            overlays=w_star_overlays,
            sampling_frequency=self.sampling_frequency,
        )
        if self.show_title:
            ax.set_title(self.title)
        ax.set_ylabel(f"Duration [{duration_scale.label}]")
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
            scale = _resolve_duration_scale(chart_config)
            self.render(
                ax,
                metrics.times,
                metrics.w,
                metrics.w_prime,
                empirical_metrics.W_star,
                arrival_times=metrics.arrival_times,
                departure_times=metrics.departure_times,
                scale=scale,
            )
        return resolved_out_path


@dataclass
class SojournVsResidenceTimeScatterPanel:
    show_title: bool = True
    title: str = "Sojourn Time vs Residence Times"
    with_event_marks: bool = False
    sampling_frequency: Optional[str] = None

    def render(
        self,
        ax,
        times: List[pd.Timestamp],
        w_vals: np.ndarray,
        w_prime_vals: np.ndarray,
        departure_times: List[pd.Timestamp],
        sojourn_vals: np.ndarray,
        *,
        scale: Optional[DurationScale] = None,
    ) -> None:
        duration_scale = scale or HOURS
        w_scaled = np.asarray(w_vals, dtype=float) / duration_scale.divisor
        w_prime_scaled = np.asarray(w_prime_vals, dtype=float) / duration_scale.divisor
        sojourn_scaled = np.asarray(sojourn_vals, dtype=float) / duration_scale.divisor
        departures = departure_times or []
        w_overlays = (
            build_event_overlays(
                times,
                w_scaled,
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
                w_prime_scaled,
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
            sojourn_scaled,
            label="Sojourn time (departures)",
            color=ColorConfig.departure_color,
        )
        render_line_chart(
            ax,
            times,
            w_scaled,
            label=f"w(T) [{duration_scale.label}]",
            color="tab:blue",
            overlays=w_overlays,
            sampling_frequency=self.sampling_frequency,
        )
        render_line_chart(
            ax,
            times,
            w_prime_scaled,
            label=f"w'(T) [{duration_scale.label}]",
            color="tab:orange",
            overlays=w_prime_overlays,
            sampling_frequency=self.sampling_frequency,
        )
        if self.show_title:
            ax.set_title(self.title)
        ax.set_ylabel(f"Duration [{duration_scale.label}]")
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
            scale = _resolve_duration_scale(chart_config)
            self.render(
                ax,
                metrics.times,
                metrics.w,
                metrics.w_prime,
                metrics.departure_times,
                empirical_metrics.sojourn_vals,
                scale=scale,
            )
        return resolved_out_path


@dataclass
class ArrivalDepartureRateConvergencePanel:
    show_title: bool = True
    title: str = "Arrival Rate Λ(T) vs Departure Rate θ(T)"
    show_derivations: bool = False
    with_event_marks: bool = False
    sampling_frequency: Optional[str] = None

    def render(
        self,
        ax,
        times: List[pd.Timestamp],
        departures_cum: np.ndarray,
        lambda_cum_rate: np.ndarray,
        *,
        arrival_times: Optional[List[pd.Timestamp]] = None,
        departure_times: Optional[List[pd.Timestamp]] = None,
        lambda_pctl_upper: Optional[float] = None,
        lambda_pctl_lower: Optional[float] = None,
        lambda_warmup_seconds: Optional[float] = None,
        scale: Optional[DurationScale] = None,
    ) -> None:
        duration_scale = scale or HOURS
        n = len(times)
        if n > 0:
            t0 = times[0]
            elapsed_seconds = np.array(
                [(t - t0).total_seconds() for t in times], dtype=float
            )
        else:
            elapsed_seconds = np.array([], dtype=float)

        with np.errstate(divide="ignore", invalid="ignore"):
            theta_rate = np.where(
                elapsed_seconds > 0.0, departures_cum / elapsed_seconds, np.nan
            )
        lambda_rate_scaled = (
            np.asarray(lambda_cum_rate, dtype=float) * duration_scale.divisor
        )
        theta_rate_scaled = np.asarray(theta_rate, dtype=float) * duration_scale.divisor

        last_dep_idx = -1
        if len(departures_cum) > 0:
            dep_vals = np.asarray(departures_cum, dtype=float)
            inc = np.flatnonzero(np.diff(dep_vals, prepend=dep_vals[0]) > 0)
            if inc.size > 0:
                last_dep_idx = int(inc.max())

        if last_dep_idx >= 0 and last_dep_idx + 1 < n:
            theta_rate[last_dep_idx + 1 :] = np.nan

        lambda_label = "Λ(T)- Cumulative Arrival Rate"
        theta_label = "θ(T) - Cumulative Departure Rate "
        if self.show_derivations:
            deriv_lambda = MetricDerivations.get("Lambda")
            deriv_theta = MetricDerivations.get("Theta")
            if deriv_lambda:
                lambda_label = f"{lambda_label} — {deriv_lambda}"
            if deriv_theta:
                theta_label = f"{theta_label} — {deriv_theta}"

        overlays_lambda = (
            build_event_overlays(
                times,
                lambda_rate_scaled,
                arrival_times or [],
                [],
                drop_lines_for_arrivals=True,
                drop_lines_for_departures=False,
            )
            if self.with_event_marks
            else None
        )
        overlays_theta = (
            build_event_overlays(
                times,
                theta_rate_scaled,
                [],
                departure_times or [],
                drop_lines_for_arrivals=False,
                drop_lines_for_departures=True,
            )
            if self.with_event_marks
            else None
        )
        render_line_chart(
            ax,
            times,
            lambda_rate_scaled,
            label=f"{lambda_label} [{duration_scale.rate_label}]",
            color="tab:blue",
            overlays=overlays_lambda,
            sampling_frequency=self.sampling_frequency,
        )
        render_line_chart(
            ax,
            times,
            theta_rate_scaled,
            label=f"{theta_label} [{duration_scale.rate_label}]",
            color="tab:orange",
            overlays=overlays_theta,
            sampling_frequency=self.sampling_frequency,
        )
        if self.show_title:
            ax.set_title(self.title)
        ax.set_ylabel(f"Rate [{duration_scale.rate_label}]")
        ax.legend(loc="best")
        _clip_axis_to_percentile(
            ax,
            times,
            lambda_rate_scaled,
            upper_p=lambda_pctl_upper,
            lower_p=lambda_pctl_lower,
            warmup_seconds=(lambda_warmup_seconds or 0.0),
        )

    def plot(
        self,
        metrics: FlowMetricsResult,
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
            base_name="arrival_departure_rate_convergence",
        ) as (
            _,
            axes,
            resolved_out_path,
        ):
            ax = _first_axis(axes)
            scale = _resolve_duration_scale(chart_config)
            self.render(
                ax,
                metrics.times,
                metrics.Departures,
                metrics.Lambda,
                arrival_times=metrics.arrival_times,
                departure_times=metrics.departure_times,
                lambda_pctl_upper=chart_config.lambda_pctl_upper,
                lambda_pctl_lower=chart_config.lambda_pctl_lower,
                lambda_warmup_seconds=chart_config.lambda_warmup_seconds,
                scale=scale,
            )
        return resolved_out_path


@dataclass
class CumulativeArrivalRateConvergencePanel:
    show_title: bool = True
    title: str = "Λ(T) vs λ*(t) — arrival rate"
    show_derivations: bool = False
    with_event_marks: bool = False
    sampling_frequency: Optional[str] = None

    def render(
        self,
        ax,
        times: List[pd.Timestamp],
        lam_vals: np.ndarray,
        lam_star: np.ndarray,
        *,
        arrival_times: Optional[List[pd.Timestamp]] = None,
        lambda_pctl_upper: Optional[float] = None,
        lambda_pctl_lower: Optional[float] = None,
        lambda_warmup_seconds: Optional[float] = None,
        scale: Optional[DurationScale] = None,
    ) -> None:
        duration_scale = scale or HOURS
        lam_vals_scaled = np.asarray(lam_vals, dtype=float) * duration_scale.divisor
        lam_star_scaled = np.asarray(lam_star, dtype=float) * duration_scale.divisor
        lambda_label = "Λ(T) - Cumulative Arrival Rate"
        if self.show_derivations:
            deriv_lambda = MetricDerivations.get("Lambda")
            if deriv_lambda:
                lambda_label = f"{lambda_label} — {deriv_lambda}"
        overlays = (
            build_event_overlays(
                times,
                lam_vals_scaled,
                arrival_times or [],
                [],
                drop_lines_for_arrivals=True,
                drop_lines_for_departures=False,
            )
            if self.with_event_marks
            else None
        )
        render_line_chart(
            ax,
            times,
            lam_vals_scaled,
            label=f"{lambda_label} [{duration_scale.rate_label}]",
            color="tab:blue",
            overlays=overlays,
            sampling_frequency=self.sampling_frequency,
        )
        render_line_chart(
            ax,
            times,
            lam_star_scaled,
            label=f"λ*(T) (Arrivals ≤ T) [{duration_scale.rate_label}]",
            color="tab:orange",
            sampling_frequency=self.sampling_frequency,
        )
        if self.show_title:
            ax.set_title(self.title)
        ax.set_ylabel(f"Arrival Rate [{duration_scale.rate_label}]")
        ax.legend()
        _clip_axis_to_percentile(
            ax,
            times,
            lam_vals_scaled,
            upper_p=lambda_pctl_upper,
            lower_p=lambda_pctl_lower,
            warmup_seconds=lambda_warmup_seconds,
        )

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
            base_name="arrival_rate_convergence",
        ) as (
            _,
            axes,
            resolved_out_path,
        ):
            ax = _first_axis(axes)
            scale = _resolve_duration_scale(chart_config)
            self.render(
                ax,
                metrics.times,
                metrics.Lambda,
                empirical_metrics.lam_star,
                arrival_times=metrics.arrival_times,
                lambda_pctl_upper=chart_config.lambda_pctl_upper,
                lambda_pctl_lower=chart_config.lambda_pctl_lower,
                lambda_warmup_seconds=chart_config.lambda_warmup_seconds,
                scale=scale,
            )
        return resolved_out_path


@dataclass
class SamplePathConvergencePanel:
    show_title: bool = True
    title: str = "Sample Path Convergence: L(T) vs λ*(t)·W*(t)"

    def render(
        self,
        ax: plt.Axes,
        L_vals: np.ndarray,
        lam_star: np.ndarray,
        W_star_seconds: np.ndarray,
        times: List[pd.Timestamp],
        *,
        epsilon: float,
        horizon_seconds: float,
    ) -> Tuple[float, int, int]:
        """
        Scatter points x=L(T) vs y=λ*(t)·W*(t), draw x=y and an ε relative band.
        Return (score, ok_count, total_count) using only points with elapsed >= horizon.
        """
        y_vals = lam_star * W_star_seconds
        x_vals = np.asarray(L_vals, dtype=float)

        n = len(times)
        if n > 0:
            t0 = times[0]
            elapsed_seconds = np.array(
                [(t - t0).total_seconds() for t in times], dtype=float
            )
        else:
            elapsed_seconds = np.array([], dtype=float)

        finite_mask = np.isfinite(x_vals) & np.isfinite(y_vals) & (x_vals > 0.0)
        if horizon_seconds and horizon_seconds > 0.0:
            finite_mask &= elapsed_seconds >= float(horizon_seconds)

        X = x_vals[finite_mask]
        Y = y_vals[finite_mask]

        rel_err = np.abs(Y / X - 1.0)
        ok_mask = rel_err <= float(epsilon)
        ok_count = int(np.count_nonzero(ok_mask))
        total_count = int(X.size)
        score = (ok_count / total_count) if total_count > 0 else float("nan")

        ax.scatter(X, Y, s=16, alpha=0.7, label="Points: (L(T), λ*(t)·W*(t))")

        if X.size > 0:
            x_min, x_max = float(np.nanmin(X)), float(np.nanmax(X))
            pad = 0.05 * (x_max - x_min if x_max > x_min else 1.0)
            x_line = np.linspace(max(0.0, x_min - pad), x_max + pad, 200)
            ax.plot(x_line, x_line, linewidth=1.2, label="x = y")
            ax.plot(
                x_line,
                (1.0 + float(epsilon)) * x_line,
                linewidth=0.8,
                linestyle="--",
                label="y=(1+ε)x",
            )
            ax.plot(
                x_line,
                (1.0 - float(epsilon)) * x_line,
                linewidth=0.8,
                linestyle="--",
                label="y=(1−ε)x",
            )
            ax.fill_between(
                x_line,
                (1.0 - float(epsilon)) * x_line,
                (1.0 + float(epsilon)) * x_line,
                alpha=0.08,
            )

        ax.set_xlabel("L(T)")
        ax.set_ylabel("λ*(t)·W*(t)")
        ax.legend(loc="best")

        if total_count > 0:
            ax.text(
                0.02,
                0.98,
                f"ε={epsilon:.3g}, horizon={horizon_seconds/86400.0:.1f}d:  {ok_count}/{total_count}  ({score*100:.1f}%)",
                ha="left",
                va="top",
                transform=ax.transAxes,
                fontsize=9,
            )
        return score, ok_count, total_count

    def plot(
        self,
        metrics: FlowMetricsResult,
        empirical_metrics: ElementWiseEmpiricalMetrics,
        filter_result: Optional[FilterResult],
        chart_config: ChartConfig,
        out_dir: str,
    ) -> str:
        caption = resolve_caption(filter_result)
        layout = LayoutSpec(nrows=1, ncols=1, figsize=(7.5, 6.0), sharex=False)
        decor = FigureDecorSpec(
            suptitle=self.title if self.show_title else None,
            suptitle_y=0.98,
            caption=caption,
            caption_position="top",
            caption_y=0.945,
            tight_layout=True,
            tight_layout_rect=(0, 0, 1, 0.93),
        )
        with layout_context(
            chart_config=chart_config,
            layout=layout,
            decor=decor,
            unit=None,
            out_dir=out_dir,
            subdir=None,
            base_name="sample_path_convergence",
        ) as (
            _,
            axes,
            resolved_out_path,
        ):
            ax = _first_axis(axes)
            score, ok_count, total_count = self.render(
                ax,
                metrics.L,
                empirical_metrics.lam_star,
                empirical_metrics.W_star,
                metrics.times,
                epsilon=chart_config.epsilon,
                horizon_seconds=chart_config.horizon_seconds,
            )

        if np.isnan(score):
            print(
                f"Sample Path Convergence: ε={chart_config.epsilon}, "
                f"H={chart_config.horizon_seconds/86400.0:.1f}d -> n/a (no valid points)\n"
            )
        else:
            print(
                f"Sample Path Convergence: ε={chart_config.epsilon}, "
                f"H={chart_config.horizon_seconds/86400.0:.1f}d -> "
                f"{ok_count}/{total_count} ({score*100:.1f}%)\n"
            )
        return resolved_out_path


# -- Stacks ----


def plot_arrival_departure_equilibrium_stack(
    filter_result: Optional[FilterResult],
    metrics: FlowMetricsResult,
    chart_config: ChartConfig,
    out_dir: str,
) -> str:
    caption = resolve_caption(filter_result)

    layout = LayoutSpec(nrows=2, ncols=1, figsize=(12.0, 6.5), sharex=True)
    decor = FigureDecorSpec(
        suptitle="Flow Equilibrium: Arrival/Departure Convergence",
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
        base_name="arrival_departure_equilibrium",
    ) as (_, axes, resolved_out_path):
        flat_axes = axes if not isinstance(axes, np.ndarray) else axes.ravel()
        scale = _resolve_duration_scale(chart_config)
        CFDPanel(
            with_event_marks=chart_config.with_event_marks,
            show_derivations=chart_config.show_derivations,
        ).render(
            flat_axes[0],
            metrics.times,
            metrics.Arrivals,
            metrics.Departures,
            arrival_times=metrics.arrival_times,
            departure_times=metrics.departure_times,
        )
        ArrivalDepartureRateConvergencePanel(
            with_event_marks=chart_config.with_event_marks,
            show_derivations=chart_config.show_derivations,
            sampling_frequency=chart_config.sampling_frequency,
        ).render(
            flat_axes[1],
            metrics.times,
            metrics.Departures,
            metrics.Lambda,
            arrival_times=metrics.arrival_times,
            departure_times=metrics.departure_times,
            lambda_pctl_upper=chart_config.lambda_pctl_upper,
            lambda_pctl_lower=chart_config.lambda_pctl_lower,
            lambda_warmup_seconds=chart_config.lambda_warmup_seconds,
            scale=scale,
        )
    return resolved_out_path


def plot_process_time_convergence_stack(
    filter_result: Optional[FilterResult],
    metrics: FlowMetricsResult,
    empirical_metrics: ElementWiseEmpiricalMetrics,
    chart_config: ChartConfig,
    out_dir: str,
) -> str:
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
        scale = _resolve_duration_scale(chart_config)
        ProcessTimeConvergencePanel(
            with_event_marks=chart_config.with_event_marks,
            sampling_frequency=chart_config.sampling_frequency,
        ).render(
            flat_axes[0],
            metrics.times,
            metrics.w,
            metrics.w_prime,
            empirical_metrics.W_star,
            arrival_times=metrics.arrival_times,
            departure_times=metrics.departure_times,
            scale=scale,
        )
        SojournVsResidenceTimeScatterPanel(
            with_event_marks=chart_config.with_event_marks,
            sampling_frequency=chart_config.sampling_frequency,
        ).render(
            flat_axes[1],
            metrics.times,
            metrics.w,
            metrics.w_prime,
            metrics.departure_times,
            empirical_metrics.sojourn_vals,
            scale=scale,
        )
    return resolved_out_path


def plot_convergence_charts(
    metrics: FlowMetricsResult,
    empirical_metrics: ElementWiseEmpiricalMetrics,
    filter_result: Optional[FilterResult],
    chart_config: ChartConfig,
    out_dir: str,
) -> List[str]:
    show_derivations = chart_config.show_derivations

    path_lambda = CumulativeArrivalRateConvergencePanel(
        with_event_marks=chart_config.with_event_marks,
        show_derivations=show_derivations,
        sampling_frequency=chart_config.sampling_frequency,
    ).plot(metrics, empirical_metrics, filter_result, chart_config, out_dir)

    path_rate = ArrivalDepartureRateConvergencePanel(
        with_event_marks=chart_config.with_event_marks,
        show_derivations=show_derivations,
        sampling_frequency=chart_config.sampling_frequency,
    ).plot(metrics, filter_result, chart_config, out_dir)

    path_process_time = ProcessTimeConvergencePanel(
        with_event_marks=chart_config.with_event_marks,
        sampling_frequency=chart_config.sampling_frequency,
    ).plot(metrics, empirical_metrics, filter_result, chart_config, out_dir)

    path_sojourn_scatter = SojournVsResidenceTimeScatterPanel(
        with_event_marks=chart_config.with_event_marks,
        sampling_frequency=chart_config.sampling_frequency,
    ).plot(metrics, empirical_metrics, filter_result, chart_config, out_dir)

    path_arrival_departure_stack = plot_arrival_departure_equilibrium_stack(
        filter_result, metrics, chart_config, out_dir
    )
    path_process_time_stack = plot_process_time_convergence_stack(
        filter_result, metrics, empirical_metrics, chart_config, out_dir
    )

    path_sample_convergence = SamplePathConvergencePanel().plot(
        metrics, empirical_metrics, filter_result, chart_config, out_dir
    )

    return [
        path_lambda,
        path_rate,
        path_process_time,
        path_sojourn_scatter,
        path_arrival_departure_stack,
        path_process_time_stack,
        path_sample_convergence,
    ]
