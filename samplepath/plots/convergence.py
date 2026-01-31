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
    build_event_overlays,
    format_date_axis,
    render_line_chart,
    render_scatter_chart,
    resolve_caption,
)


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
            sampling_frequency=self.sampling_frequency,
        )
        render_line_chart(
            ax,
            times,
            w_prime_vals,
            label="w'(T) [hrs]",
            color="tab:orange",
            overlays=w_prime_overlays,
            sampling_frequency=self.sampling_frequency,
        )
        render_line_chart(
            ax,
            times,
            w_star_vals,
            label="W*(t) [hrs]",
            color="tab:green",
            overlays=w_star_overlays,
            sampling_frequency=self.sampling_frequency,
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
    sampling_frequency: Optional[str] = None

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
            sampling_frequency=self.sampling_frequency,
        )
        render_line_chart(
            ax,
            times,
            w_prime_vals,
            label="w'(T) [hrs]",
            color="tab:orange",
            overlays=w_prime_overlays,
            sampling_frequency=self.sampling_frequency,
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
        lambda_warmup_hours: Optional[float] = None,
    ) -> None:
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
                lambda_cum_rate,
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
                theta_rate,
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
            lambda_cum_rate,
            label=lambda_label,
            color="tab:blue",
            overlays=overlays_lambda,
            sampling_frequency=self.sampling_frequency,
        )
        render_line_chart(
            ax,
            times,
            theta_rate,
            label=theta_label,
            color="tab:orange",
            overlays=overlays_theta,
            sampling_frequency=self.sampling_frequency,
        )
        if self.show_title:
            ax.set_title(self.title)
        ax.set_ylabel("Rate [per hr]")
        ax.legend(loc="best")
        _clip_axis_to_percentile(
            ax,
            times,
            lambda_cum_rate,
            upper_p=lambda_pctl_upper,
            lower_p=lambda_pctl_lower,
            warmup_hours=(lambda_warmup_hours or 0.0),
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
            self.render(
                ax,
                metrics.times,
                metrics.Departures,
                metrics.Lambda,
                arrival_times=metrics.arrival_times,
                departure_times=metrics.departure_times,
                lambda_pctl_upper=chart_config.lambda_pctl_upper,
                lambda_pctl_lower=chart_config.lambda_pctl_lower,
                lambda_warmup_hours=chart_config.lambda_warmup_hours,
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
        lambda_warmup_hours: Optional[float] = None,
    ) -> None:
        lambda_label = "Λ(T) - Cumulative Arrival Rate"
        if self.show_derivations:
            deriv_lambda = MetricDerivations.get("Lambda")
            if deriv_lambda:
                lambda_label = f"{lambda_label} — {deriv_lambda}"
        overlays = (
            build_event_overlays(
                times,
                lam_vals,
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
            lam_vals,
            label=lambda_label,
            color="tab:blue",
            overlays=overlays,
            sampling_frequency=self.sampling_frequency,
        )
        render_line_chart(
            ax,
            times,
            lam_star,
            label="λ*(T) (Arrivals ≤ T)",
            color="tab:orange",
            sampling_frequency=self.sampling_frequency,
        )
        if self.show_title:
            ax.set_title(self.title)
        ax.set_ylabel("Arrival Rate [per hr]")
        ax.legend()
        _clip_axis_to_percentile(
            ax,
            times,
            lam_vals,
            upper_p=lambda_pctl_upper,
            lower_p=lambda_pctl_lower,
            warmup_hours=lambda_warmup_hours,
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
            self.render(
                ax,
                metrics.times,
                metrics.Lambda,
                empirical_metrics.lam_star,
                arrival_times=metrics.arrival_times,
                lambda_pctl_upper=chart_config.lambda_pctl_upper,
                lambda_pctl_lower=chart_config.lambda_pctl_lower,
                lambda_warmup_hours=chart_config.lambda_warmup_hours,
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
        W_star_hours: np.ndarray,
        times: List[pd.Timestamp],
        *,
        epsilon: float,
        horizon_days: float,
    ) -> Tuple[float, int, int]:
        """
        Scatter points x=L(T) vs y=λ*(t)·W*(t), draw x=y and an ε relative band.
        Return (score, ok_count, total_count) using only points with elapsed >= horizon.
        """
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

        finite_mask = np.isfinite(x_vals) & np.isfinite(y_vals) & (x_vals > 0.0)
        horizon_hours = float(horizon_days) * 24.0
        if horizon_hours and horizon_hours > 0.0:
            finite_mask &= elapsed_h >= float(horizon_hours)

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
                f"ε={epsilon:.3g}, horizon={horizon_days:.1f}d:  {ok_count}/{total_count}  ({score*100:.1f}%)",
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
                horizon_days=chart_config.horizon_days,
            )

        if np.isnan(score):
            print(
                f"Sample Path Convergence: ε={chart_config.epsilon}, "
                f"H={chart_config.horizon_days}d -> n/a (no valid points)\n"
            )
        else:
            print(
                f"Sample Path Convergence: ε={chart_config.epsilon}, "
                f"H={chart_config.horizon_days}d -> "
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
            lambda_warmup_hours=chart_config.lambda_warmup_hours,
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
        )
        SojournTimeScatterPanel(
            with_event_marks=chart_config.with_event_marks,
            sampling_frequency=chart_config.sampling_frequency,
        ).render(
            flat_axes[1],
            metrics.times,
            metrics.w,
            metrics.w_prime,
            metrics.departure_times,
            empirical_metrics.sojourn_vals,
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

    path_sojourn_scatter = SojournTimeScatterPanel(
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
