# -*- coding: utf-8 -*-
# Copyright (c) 2025 Krishna Kumar
# SPDX-License-Identifier: MIT
"""
Experimental core plotting module using figure_context.

This module defines panel-level renderers (N, L, Λ, w, H) and a 4-panel stack
layout built on the new figure_context helper. It leaves existing plotting
entrypoints untouched; wiring to sample_path_analysis can happen once this
path is validated.
"""

from dataclasses import dataclass
from typing import List, Optional, Sequence

from matplotlib import colors as mcolors, pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

from samplepath.filter import FilterResult
from samplepath.metrics import FlowMetricsResult, MetricDerivations
from samplepath.plots.chart_config import ChartConfig
from samplepath.plots.figure_context import (
    FigureDecorSpec,
    LayoutSpec,
    _first_axis,
    figure_context,
    layout_context,
)
from samplepath.plots.helpers import (
    ScatterOverlay,
    _clip_axis_to_percentile,
    build_event_overlays,
    format_date_axis,
    render_line_chart,
    render_scatter_chart,
    render_step_chart,
    resolve_caption,
)


@dataclass
class ClipOptions:
    """Bundle for Lambda clipping parameters."""

    pctl_upper: Optional[float] = None
    pctl_lower: Optional[float] = None
    warmup_hours: float = 0.0


def construct_title(
    base_title: str, show_derivations: bool, derivation_key: Optional[str] = None
) -> str:
    if not show_derivations:
        return base_title
    if not derivation_key:
        return base_title
    derivation = MetricDerivations.get(derivation_key)
    return f"{base_title}: {derivation}" if derivation else base_title


@dataclass
class NPanel:
    show_title: bool = True
    title: str = "N(t) — Sample Path"
    show_derivations: bool = False
    with_event_marks: bool = False

    def render(
        self,
        ax,
        times: Sequence[pd.Timestamp],
        N_vals: Sequence[float],
        *,
        arrival_times: Optional[List[pd.Timestamp]] = None,
        departure_times: Optional[List[pd.Timestamp]] = None,
    ) -> None:
        overlays = (
            build_event_overlays(times, N_vals, arrival_times, departure_times)
            if self.with_event_marks
            else None
        )
        color = "grey" if overlays else "tab:blue"
        render_step_chart(
            ax,
            times,
            N_vals,
            label="N(t)",
            color=color,
            fill=True,
            overlays=overlays,
        )
        if self.show_title:
            ax.set_title(
                construct_title(self.title, self.show_derivations, derivation_key="N")
            )
        ax.set_ylabel("N(t)")
        ax.legend()

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
            subdir="core/panels",
            base_name="sample_path_N",
        ) as (
            _,
            axes,
            resolved_out_path,
        ):
            ax = _first_axis(axes)
            self.render(
                ax,
                metrics.times,
                metrics.N,
                arrival_times=metrics.arrival_times,
                departure_times=metrics.departure_times,
            )
        return resolved_out_path


@dataclass
class LPanel:
    show_title: bool = True
    title: str = "L(T) — Time-Average of N(t)"
    show_derivations: bool = False
    with_event_marks: bool = False

    def render(
        self,
        ax,
        times: Sequence[pd.Timestamp],
        L_vals: Sequence[float],
        *,
        arrival_times: Optional[List[pd.Timestamp]] = None,
        departure_times: Optional[List[pd.Timestamp]] = None,
    ) -> None:
        overlays = (
            build_event_overlays(times, L_vals, arrival_times, departure_times)
            if self.with_event_marks
            else None
        )
        color = "grey" if overlays else "tab:blue"
        render_line_chart(
            ax, times, L_vals, label="L(T)", color=color, overlays=overlays
        )
        if self.show_title:
            ax.set_title(
                construct_title(self.title, self.show_derivations, derivation_key="L")
            )
        ax.set_ylabel("L(T)")
        ax.legend()

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
            subdir="core/panels",
            base_name="time_average_N_L",
        ) as (
            _,
            axes,
            resolved_out_path,
        ):
            ax = _first_axis(axes)
            self.render(
                ax,
                metrics.times,
                metrics.L,
                arrival_times=metrics.arrival_times,
                departure_times=metrics.departure_times,
            )
        return resolved_out_path


@dataclass
class LambdaPanel:
    show_title: bool = True
    title: str = "Λ(T) — Cumulative Arrival Rate"
    show_derivations: bool = False
    with_event_marks: bool = False
    clip_opts: Optional[ClipOptions] = None

    def render(
        self,
        ax,
        times: Sequence[pd.Timestamp],
        Lam_vals: Sequence[float],
        *,
        arrival_times: Optional[List[pd.Timestamp]] = None,
    ) -> None:
        overlays = (
            build_event_overlays(
                times,
                Lam_vals,
                arrival_times,
                [],
                drop_lines_for_arrivals=True,
            )
            if self.with_event_marks
            else None
        )
        render_line_chart(
            ax,
            times,
            Lam_vals,
            label="Λ(T) [1/hr]",
            color="tab:blue",
            overlays=overlays,
        )
        opts = self.clip_opts or ClipOptions()
        if opts.pctl_upper is not None or opts.pctl_lower is not None:
            _clip_axis_to_percentile(
                ax,
                list(times),
                Lam_vals,
                opts.pctl_upper,
                opts.pctl_lower,
                opts.warmup_hours,
            )
        if self.show_title:
            ax.set_title(
                construct_title(
                    self.title, self.show_derivations, derivation_key="Lambda"
                )
            )
        ax.set_ylabel("Λ(T) [1/hr]")
        ax.legend()

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
            subdir="core/panels",
            base_name="cumulative_arrival_rate_Lambda",
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
                arrival_times=metrics.arrival_times,
            )
        return resolved_out_path


@dataclass
class ThetaPanel:
    show_title: bool = True
    title: str = "Θ(T) — Cumulative Departure Rate"
    show_derivations: bool = False
    with_event_marks: bool = False

    def render(
        self,
        ax,
        times: Sequence[pd.Timestamp],
        theta_vals: Sequence[float],
        *,
        departure_times: Optional[List[pd.Timestamp]] = None,
    ) -> None:
        label = "Θ(T) [1/hr]"
        overlays = (
            build_event_overlays(times, theta_vals, [], departure_times)
            if self.with_event_marks and departure_times is not None
            else None
        )
        color = "grey" if overlays else "tab:blue"
        render_line_chart(
            ax, times, theta_vals, label=label, color=color, overlays=overlays
        )
        if self.show_title:
            ax.set_title(
                construct_title(
                    self.title, self.show_derivations, derivation_key="Theta"
                )
            )
        ax.set_ylabel(label)
        ax.legend()

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
            subdir="core/panels",
            base_name="cumulative_departure_rate_Theta",
        ) as (
            _,
            axes,
            resolved_out_path,
        ):
            ax = _first_axis(axes)
            self.render(
                ax,
                metrics.times,
                metrics.Theta,
                departure_times=metrics.departure_times,
            )
        return resolved_out_path


@dataclass
class WPanel:
    show_title: bool = True
    title: str = "w(T) — Average Residence Time per Arrival"
    show_derivations: bool = False
    with_event_marks: bool = False

    def render(
        self,
        ax,
        times: Sequence[pd.Timestamp],
        w_vals: Sequence[float],
        *,
        arrival_times: Optional[List[pd.Timestamp]] = None,
        departure_times: Optional[List[pd.Timestamp]] = None,
    ) -> None:
        label = "w(T) [hrs]"
        overlays = (
            build_event_overlays(
                times,
                w_vals,
                arrival_times,
                departure_times,
                drop_lines_for_arrivals=True,
                drop_lines_for_departures=False,
            )
            if self.with_event_marks
            else None
        )
        render_line_chart(
            ax, times, w_vals, label=label, color="tab:blue", overlays=overlays
        )
        if self.show_title:
            ax.set_title(
                construct_title(self.title, self.show_derivations, derivation_key="w")
            )
        ax.set_ylabel(label)
        ax.legend()

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
            subdir="core/panels",
            base_name="average_residence_time_w",
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
                arrival_times=metrics.arrival_times,
                departure_times=metrics.departure_times,
            )
        return resolved_out_path


@dataclass
class HPanel:
    show_title: bool = True
    title: str = "H(T) — Cumulative Presence Mass"
    show_derivations: bool = False
    with_event_marks: bool = False

    def render(
        self,
        ax,
        times: Sequence[pd.Timestamp],
        H_vals: Sequence[float],
        *,
        arrival_times: Optional[List[pd.Timestamp]] = None,
        departure_times: Optional[List[pd.Timestamp]] = None,
    ) -> None:
        label = "H(T) [hrs·items]"
        overlays = (
            build_event_overlays(times, H_vals, arrival_times, departure_times)
            if self.with_event_marks
            else None
        )
        color = "grey" if overlays else "tab:blue"
        render_line_chart(
            ax, times, H_vals, label=label, color=color, overlays=overlays
        )
        if self.show_title:
            ax.set_title(
                construct_title(self.title, self.show_derivations, derivation_key="H")
            )
        ax.set_ylabel(label)
        ax.legend()

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
            subdir="core/panels",
            base_name="cumulative_presence_mass_H",
        ) as (
            _,
            axes,
            resolved_out_path,
        ):
            ax = _first_axis(axes)
            self.render(
                ax,
                metrics.times,
                metrics.H,
                arrival_times=metrics.arrival_times,
                departure_times=metrics.departure_times,
            )
        return resolved_out_path


@dataclass
class WPrimePanel:
    show_title: bool = True
    title: str = "w'(T) — Average Residence Time per Departure"
    show_derivations: bool = False
    with_event_marks: bool = False

    def render(
        self,
        ax,
        times: Sequence[pd.Timestamp],
        w_prime_vals: Sequence[float],
        *,
        arrival_times: Optional[List[pd.Timestamp]] = None,
        departure_times: Optional[List[pd.Timestamp]] = None,
    ) -> None:
        label = "w'(T) [hrs]"
        overlays = (
            build_event_overlays(
                times,
                w_prime_vals,
                arrival_times,
                departure_times,
                drop_lines_for_arrivals=False,
                drop_lines_for_departures=True,
            )
            if self.with_event_marks
            else None
        )
        color = "grey" if overlays else "tab:blue"
        render_line_chart(
            ax, times, w_prime_vals, label=label, color=color, overlays=overlays
        )
        if self.show_title:
            ax.set_title(
                construct_title(
                    self.title, self.show_derivations, derivation_key="w_prime"
                )
            )
        ax.set_ylabel(label)
        ax.legend()

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
            subdir="core/panels",
            base_name="average_residence_time_w_prime",
        ) as (
            _,
            axes,
            resolved_out_path,
        ):
            ax = _first_axis(axes)
            self.render(
                ax,
                metrics.times,
                metrics.w_prime,
                arrival_times=metrics.arrival_times,
                departure_times=metrics.departure_times,
            )
        return resolved_out_path


@dataclass
class CFDPanel:
    show_title: bool = True
    title: str = "Cumulative Flow Diagram"
    show_derivations: bool = False
    with_event_marks: bool = False

    def render(
        self,
        ax,
        times: Sequence[pd.Timestamp],
        arrivals_cum: Sequence[float],
        departures_cum: Sequence[float],
        *,
        arrival_times: Optional[List[pd.Timestamp]] = None,
        departure_times: Optional[List[pd.Timestamp]] = None,
    ) -> None:
        arrivals_overlay = None
        departures_overlay = None
        if self.with_event_marks:
            if arrival_times is not None:
                time_to_idx = {t: i for i, t in enumerate(times)}
                arrival_x = [t for t in arrival_times if t in time_to_idx]
                arrival_y = [float(arrivals_cum[time_to_idx[t]]) for t in arrival_x]
                arrivals_overlay = [
                    ScatterOverlay(
                        x=arrival_x,
                        y=arrival_y,
                        color="purple",
                        label="Arrival",
                        drop_lines=True,
                    )
                ]
            if departure_times is not None:
                time_to_idx = {t: i for i, t in enumerate(times)}
                departure_x = [t for t in departure_times if t in time_to_idx]
                departure_y = [
                    float(departures_cum[time_to_idx[t]]) for t in departure_x
                ]
                departures_overlay = [
                    ScatterOverlay(
                        x=departure_x,
                        y=departure_y,
                        color="green",
                        label="Departure",
                        drop_lines=True,
                    )
                ]
        arrivals_label = "A(T) - Cumulative arrivals"
        departures_label = "D(T) - Cumulative departures"
        if self.show_derivations:
            deriv_arrivals = MetricDerivations.get("A")
            deriv_departures = MetricDerivations.get("D")
            if deriv_arrivals:
                arrivals_label = f"{arrivals_label} — {deriv_arrivals}"
            if deriv_departures:
                departures_label = f"{departures_label} — {deriv_departures}"
        render_step_chart(
            ax,
            times,
            arrivals_cum,
            label=arrivals_label,
            color="purple",
            fill=False,
            overlays=arrivals_overlay,
        )
        render_step_chart(
            ax,
            times,
            departures_cum,
            label=departures_label,
            color="green",
            fill=False,
            overlays=departures_overlay,
        )
        arr = np.asarray(arrivals_cum, dtype=float)
        dep = np.asarray(departures_cum, dtype=float)
        mask = arr >= dep
        if mask.any():
            ax.fill_between(
                times,
                arr,
                dep,
                where=mask,
                color="grey",
                alpha=0.3,
                step="post",
                interpolate=True,
                zorder=1,
            )
        if self.show_title:
            ax.set_title(self.title)
        ax.set_ylabel("count")
        ax.legend()

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
            subdir="core/panels",
            base_name="cumulative_flow_diagram",
        ) as (
            _,
            axes,
            resolved_out_path,
        ):
            ax = _first_axis(axes)
            self.render(
                ax,
                metrics.times,
                metrics.Arrivals,
                metrics.Departures,
                arrival_times=metrics.arrival_times,
                departure_times=metrics.departure_times,
            )
        return resolved_out_path


@dataclass
class EventIndicatorPanel:
    show_title: bool = True
    title: str = "Arrival/Departure Indicator Process"
    with_event_marks: bool = False

    def render(
        self,
        ax,
        times: Sequence[pd.Timestamp],
        *,
        arrival_times: Optional[List[pd.Timestamp]] = None,
        departure_times: Optional[List[pd.Timestamp]] = None,
    ) -> None:
        overlays: Optional[List[ScatterOverlay]] = None
        if self.with_event_marks:
            overlays = []
            if arrival_times:
                overlays.append(
                    ScatterOverlay(
                        x=arrival_times,
                        y=[1] * len(arrival_times),
                        color="purple",
                        label="Arrival",
                        drop_lines=True,
                    )
                )
            if departure_times:
                overlays.append(
                    ScatterOverlay(
                        x=departure_times,
                        y=[1] * len(departure_times),
                        color="green",
                        label="Departure",
                        drop_lines=True,
                    )
                )
        render_scatter_chart(
            ax,
            [],
            [],
            color="tab:blue",
            overlays=overlays,
        )
        ax.set_ylim(0, 1.05)
        ax.set_yticks([])
        if self.show_title:
            ax.set_title(self.title)
        ax.set_ylabel("Indicator")
        ax.legend()

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
            subdir="core/panels",
            base_name="arrival_departure_indicator_process",
        ) as (
            _,
            axes,
            resolved_out_path,
        ):
            ax = _first_axis(axes)
            self.render(
                ax,
                metrics.times,
                arrival_times=metrics.arrival_times,
                departure_times=metrics.departure_times,
            )
        return resolved_out_path


@dataclass
class ArrivalsPanel:
    show_title: bool = True
    title: str = "A(T) — Cumulative Arrivals"
    show_derivations: bool = False
    with_event_marks: bool = False

    def render(
        self,
        ax,
        times: Sequence[pd.Timestamp],
        arrivals_cum: Sequence[float],
        *,
        arrival_times: Optional[List[pd.Timestamp]] = None,
    ) -> None:
        label = "A(T)"
        overlays = (
            build_event_overlays(
                times,
                arrivals_cum,
                arrival_times,
                [],
                drop_lines_for_arrivals=True,
                drop_lines_for_departures=False,
            )
            if self.with_event_marks and arrival_times is not None
            else None
        )
        color = "grey" if overlays else "purple"
        render_step_chart(
            ax,
            times,
            arrivals_cum,
            label=label,
            color=color,
            fill=False,
            overlays=overlays,
        )
        if self.show_title:
            ax.set_title(
                construct_title(self.title, self.show_derivations, derivation_key="A")
            )
        ax.set_ylabel("count")
        ax.legend()

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
            subdir="core/panels",
            base_name="cumulative_arrivals_A",
        ) as (
            _,
            axes,
            resolved_out_path,
        ):
            ax = _first_axis(axes)
            self.render(
                ax,
                metrics.times,
                metrics.Arrivals,
                arrival_times=metrics.arrival_times,
            )
        return resolved_out_path


@dataclass
class DeparturesPanel:
    show_title: bool = True
    title: str = "D(T) — Cumulative Departures"
    show_derivations: bool = False
    with_event_marks: bool = False

    def render(
        self,
        ax,
        times: Sequence[pd.Timestamp],
        departures_cum: Sequence[float],
        *,
        departure_times: Optional[List[pd.Timestamp]] = None,
    ) -> None:
        label = "D(T)"
        overlays = (
            build_event_overlays(
                times,
                departures_cum,
                [],
                departure_times,
                drop_lines_for_arrivals=False,
                drop_lines_for_departures=True,
            )
            if self.with_event_marks and departure_times is not None
            else None
        )
        color = "grey" if overlays else "green"
        render_step_chart(
            ax,
            times,
            departures_cum,
            label=label,
            color=color,
            fill=False,
            overlays=overlays,
        )
        if self.show_title:
            ax.set_title(
                construct_title(self.title, self.show_derivations, derivation_key="D")
            )
        ax.set_ylabel("count")
        ax.legend()

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
            subdir="core/panels",
            base_name="cumulative_departures_D",
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
                departure_times=metrics.departure_times,
            )
        return resolved_out_path


@dataclass(frozen=True)
class LLWPanel:
    show_title: bool = True
    title: str = "L(T) vs Λ(T).w(T)"
    with_event_marks: bool = False

    def render(
        self,
        ax: plt.Axes,
        times: List[pd.Timestamp],
        L_vals: np.ndarray,
        Lam_vals: np.ndarray,
        w_vals: np.ndarray,
        *,
        arrival_times: Optional[List[pd.Timestamp]] = None,
        departure_times: Optional[List[pd.Timestamp]] = None,
    ) -> None:
        x = np.asarray(L_vals, dtype=float)
        y = np.asarray(Lam_vals, dtype=float) * np.asarray(w_vals, dtype=float)
        t = np.asarray(times, dtype=object)
        mask = np.isfinite(x) & np.isfinite(y)
        x, y, t = x[mask], y[mask], t[mask]

        if x.size:
            arrival_set = set(arrival_times or [])
            departure_set = set(departure_times or [])
            alphas = np.linspace(0.2, 1.0, num=len(t))
            colors = []
            drop_colors = []
            for idx, time_val in enumerate(t):
                base = "tab:blue"
                if self.with_event_marks:
                    if time_val in departure_set:
                        base = "green"
                    elif time_val in arrival_set:
                        base = "purple"
                colors.append(mcolors.to_rgba(base, alpha=float(alphas[idx])))
                drop_colors.append(mcolors.to_rgba(base, alpha=0.25))
            ax.scatter(
                x,
                y,
                s=18,
                color=colors,
                label=None,
            )
            drop_mask = np.array(
                [time_val in arrival_set for time_val in t], dtype=bool
            )
            if drop_mask.any():
                ax.vlines(
                    x[drop_mask],
                    0,
                    y[drop_mask],
                    colors=np.array(drop_colors, dtype=object)[drop_mask],
                    linewidths=0.5,
                    alpha=0.25,
                )
                ax.hlines(
                    y[drop_mask],
                    0,
                    x[drop_mask],
                    colors=np.array(drop_colors, dtype=object)[drop_mask],
                    linewidths=0.5,
                    alpha=0.25,
                )
            if self.with_event_marks:
                ax.legend(
                    handles=[
                        Line2D(
                            [0],
                            [0],
                            marker="o",
                            color="purple",
                            linestyle="None",
                            label="Arrival",
                        ),
                        Line2D(
                            [0],
                            [0],
                            marker="o",
                            color="green",
                            linestyle="None",
                            label="Departure",
                        ),
                    ],
                    loc="best",
                )

        if x.size and y.size:
            mn = float(np.nanmin([x.min(), y.min()]))
            mx = float(np.nanmax([x.max(), y.max()]))
            pad = 0.03 * (mx - mn if mx > mn else 1.0)
            lo, hi = mn - pad, mx + pad
            ax.plot([lo, hi], [lo, hi], linestyle="--")
            ax.set_xlim(lo, hi)
            ax.set_ylim(lo, hi)

        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("L(T)")
        ax.set_ylabel("Λ(T)·w(T)")
        if self.show_title:
            ax.set_title(self.title)

    def plot(
        self,
        metrics: FlowMetricsResult,
        filter_result: Optional[FilterResult],
        chart_config: ChartConfig,
        out_dir: str,
    ) -> str:
        caption = resolve_caption(filter_result)
        with figure_context(
            chart_config=chart_config,
            nrows=1,
            ncols=1,
            figsize=(6.0, 6.0),
            caption=caption,
            unit=None,
            out_dir=out_dir,
            subdir="core/panels",
            base_name="littles_law_invariant",
        ) as (
            _,
            axes,
            resolved_out_path,
        ):
            ax = _first_axis(axes)
            self.render(
                ax,
                metrics.times,
                metrics.L,
                metrics.Lambda,
                metrics.w,
                arrival_times=metrics.arrival_times,
                departure_times=metrics.departure_times,
            )
        return resolved_out_path


@dataclass(frozen=True)
class LThetaWPrimePanel:
    show_title: bool = True
    title: str = "L(T) vs Θ(T).w'(T)"
    with_event_marks: bool = False

    def render(
        self,
        ax: plt.Axes,
        times: List[pd.Timestamp],
        L_vals: np.ndarray,
        Theta_vals: np.ndarray,
        w_prime_vals: np.ndarray,
        *,
        arrival_times: Optional[List[pd.Timestamp]] = None,
        departure_times: Optional[List[pd.Timestamp]] = None,
    ) -> None:
        x = np.asarray(L_vals, dtype=float)
        y = np.asarray(Theta_vals, dtype=float) * np.asarray(w_prime_vals, dtype=float)
        t = np.asarray(times, dtype=object)
        mask = np.isfinite(x) & np.isfinite(y)
        x, y, t = x[mask], y[mask], t[mask]

        if x.size:
            arrival_set = set(arrival_times or [])
            departure_set = set(departure_times or [])
            alphas = np.linspace(0.2, 1.0, num=len(t))
            colors = []
            drop_colors = []
            for idx, time_val in enumerate(t):
                base = "tab:blue"
                if self.with_event_marks:
                    if time_val in departure_set:
                        base = "green"
                    elif time_val in arrival_set:
                        base = "purple"
                colors.append(mcolors.to_rgba(base, alpha=float(alphas[idx])))
                drop_colors.append(mcolors.to_rgba(base, alpha=0.25))
            ax.scatter(
                x,
                y,
                s=18,
                color=colors,
                label=None,
            )
            drop_mask = np.array(
                [time_val in departure_set for time_val in t], dtype=bool
            )
            if drop_mask.any():
                ax.vlines(
                    x[drop_mask],
                    0,
                    y[drop_mask],
                    colors=np.array(drop_colors, dtype=object)[drop_mask],
                    linewidths=0.5,
                    alpha=0.25,
                )
                ax.hlines(
                    y[drop_mask],
                    0,
                    x[drop_mask],
                    colors=np.array(drop_colors, dtype=object)[drop_mask],
                    linewidths=0.5,
                    alpha=0.25,
                )
            if self.with_event_marks:
                ax.legend(
                    handles=[
                        Line2D(
                            [0],
                            [0],
                            marker="o",
                            color="purple",
                            linestyle="None",
                            label="Arrival",
                        ),
                        Line2D(
                            [0],
                            [0],
                            marker="o",
                            color="green",
                            linestyle="None",
                            label="Departure",
                        ),
                    ],
                    loc="best",
                )

        if x.size and y.size:
            mn = float(np.nanmin([x.min(), y.min()]))
            mx = float(np.nanmax([x.max(), y.max()]))
            pad = 0.03 * (mx - mn if mx > mn else 1.0)
            lo, hi = mn - pad, mx + pad
            ax.plot([lo, hi], [lo, hi], linestyle="--")
            ax.set_xlim(lo, hi)
            ax.set_ylim(lo, hi)

        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("L(T)")
        ax.set_ylabel("Θ(T)·w'(T)")
        if self.show_title:
            ax.set_title(self.title)

    def plot(
        self,
        metrics: FlowMetricsResult,
        filter_result: Optional[FilterResult],
        chart_config: ChartConfig,
        out_dir: str,
    ) -> str:
        caption = resolve_caption(filter_result)
        with figure_context(
            chart_config=chart_config,
            nrows=1,
            ncols=1,
            figsize=(6.0, 6.0),
            caption=caption,
            unit=None,
            out_dir=out_dir,
            subdir="core/panels",
            base_name="departure_littles_law_invariant",
        ) as (
            _,
            axes,
            resolved_out_path,
        ):
            ax = _first_axis(axes)
            self.render(
                ax,
                metrics.times,
                metrics.L,
                metrics.Theta,
                metrics.w_prime,
                arrival_times=metrics.arrival_times,
                departure_times=metrics.departure_times,
            )
        return resolved_out_path


# ---------------------------------------------------------------------------
# Composite stack (4-panel)
# ---------------------------------------------------------------------------


def plot_core_stack(
    metrics: FlowMetricsResult,
    filter_result: Optional[FilterResult],
    chart_config: ChartConfig,
    out_dir: str,
) -> str:
    layout = LayoutSpec(nrows=4, ncols=1, figsize=(12.0, 11.0), sharex=True)
    caption = resolve_caption(filter_result)
    decor = FigureDecorSpec(
        suptitle="Sample Path Flow Metrics",
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
        base_name="sample_path_flow_metrics",
    ) as (_, axes, resolved_out_path):
        flat_axes = axes if not isinstance(axes, np.ndarray) else axes.ravel()

        NPanel(
            show_derivations=chart_config.show_derivations,
            with_event_marks=chart_config.with_event_marks,
        ).render(
            flat_axes[0],
            metrics.times,
            metrics.N,
            arrival_times=metrics.arrival_times,
            departure_times=metrics.departure_times,
        )
        LPanel(
            show_derivations=chart_config.show_derivations,
            with_event_marks=chart_config.with_event_marks,
        ).render(
            flat_axes[1],
            metrics.times,
            metrics.L,
            arrival_times=metrics.arrival_times,
            departure_times=metrics.departure_times,
        )
        LambdaPanel(
            show_derivations=chart_config.show_derivations,
            with_event_marks=chart_config.with_event_marks,
            clip_opts=ClipOptions(
                pctl_upper=chart_config.lambda_pctl_upper,
                pctl_lower=chart_config.lambda_pctl_lower,
                warmup_hours=chart_config.lambda_warmup_hours,
            ),
        ).render(
            flat_axes[2],
            metrics.times,
            metrics.Lambda,
            arrival_times=metrics.arrival_times,
        )
        WPanel(
            show_derivations=chart_config.show_derivations,
            with_event_marks=chart_config.with_event_marks,
        ).render(
            flat_axes[3],
            metrics.times,
            metrics.w,
            arrival_times=metrics.arrival_times,
            departure_times=metrics.departure_times,
        )
    return resolved_out_path


def plot_LT_derivation_stack(
    metrics: FlowMetricsResult,
    filter_result: Optional[FilterResult],
    chart_config: ChartConfig,
    out_dir: str,
) -> str:
    layout = LayoutSpec(nrows=4, ncols=1, figsize=(12.0, 11.0), sharex=True)
    caption = resolve_caption(filter_result)
    decor = FigureDecorSpec(
        suptitle="L(T) Derivation from Cumulative Flow Diagram",
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
        subdir="core",
        base_name="lt_derivation_stack",
    ) as (_, axes, resolved_out_path):
        flat_axes = axes if not isinstance(axes, np.ndarray) else axes.ravel()

        CFDPanel(
            show_derivations=chart_config.show_derivations,
            with_event_marks=chart_config.with_event_marks,
        ).render(
            flat_axes[0],
            metrics.times,
            metrics.Arrivals,
            metrics.Departures,
            arrival_times=metrics.arrival_times,
            departure_times=metrics.departure_times,
        )
        NPanel(
            show_derivations=chart_config.show_derivations,
            with_event_marks=chart_config.with_event_marks,
        ).render(
            flat_axes[1],
            metrics.times,
            metrics.N,
            arrival_times=metrics.arrival_times,
            departure_times=metrics.departure_times,
        )
        HPanel(
            show_derivations=chart_config.show_derivations,
            with_event_marks=chart_config.with_event_marks,
        ).render(
            flat_axes[2],
            metrics.times,
            metrics.H,
            arrival_times=metrics.arrival_times,
            departure_times=metrics.departure_times,
        )
        LPanel(
            show_derivations=chart_config.show_derivations,
            with_event_marks=chart_config.with_event_marks,
        ).render(
            flat_axes[3],
            metrics.times,
            metrics.L,
            arrival_times=metrics.arrival_times,
            departure_times=metrics.departure_times,
        )
    return resolved_out_path


def plot_departure_flow_metrics_stack(
    metrics: FlowMetricsResult,
    filter_result: Optional[FilterResult],
    chart_config: ChartConfig,
    out_dir: str,
) -> str:
    layout = LayoutSpec(nrows=4, ncols=1, figsize=(12.0, 11.0), sharex=True)
    caption = resolve_caption(filter_result)
    decor = FigureDecorSpec(
        suptitle="Departure-Focused Flow Metrics",
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
        subdir="core",
        base_name="departure_flow_metrics",
    ) as (_, axes, resolved_out_path):
        flat_axes = axes if not isinstance(axes, np.ndarray) else axes.ravel()

        NPanel(
            show_derivations=chart_config.show_derivations,
            with_event_marks=chart_config.with_event_marks,
        ).render(
            flat_axes[0],
            metrics.times,
            metrics.N,
            arrival_times=metrics.arrival_times,
            departure_times=metrics.departure_times,
        )
        LPanel(
            show_derivations=chart_config.show_derivations,
            with_event_marks=chart_config.with_event_marks,
        ).render(
            flat_axes[1],
            metrics.times,
            metrics.L,
            arrival_times=metrics.arrival_times,
            departure_times=metrics.departure_times,
        )
        ThetaPanel(
            show_derivations=chart_config.show_derivations,
            with_event_marks=chart_config.with_event_marks,
        ).render(
            flat_axes[2],
            metrics.times,
            metrics.Theta,
            departure_times=metrics.departure_times,
        )
        WPrimePanel(
            show_derivations=chart_config.show_derivations,
            with_event_marks=chart_config.with_event_marks,
        ).render(
            flat_axes[3],
            metrics.times,
            metrics.w_prime,
            arrival_times=metrics.arrival_times,
            departure_times=metrics.departure_times,
        )
    return resolved_out_path


def plot_core_flow_metrics_charts(
    metrics: FlowMetricsResult,
    filter_result: Optional[FilterResult],
    chart_config: ChartConfig,
    out_dir: str,
) -> List[str]:
    show_derivations = chart_config.show_derivations

    path_stack = plot_core_stack(metrics, filter_result, chart_config, out_dir)

    path_LT_derivation = plot_LT_derivation_stack(
        metrics, filter_result, chart_config, out_dir
    )

    path_departure_stack = plot_departure_flow_metrics_stack(
        metrics, filter_result, chart_config, out_dir
    )

    path_N = NPanel(
        with_event_marks=chart_config.with_event_marks,
        show_derivations=show_derivations,
    ).plot(metrics, filter_result, chart_config, out_dir)

    path_L = LPanel(
        with_event_marks=chart_config.with_event_marks,
        show_derivations=show_derivations,
    ).plot(metrics, filter_result, chart_config, out_dir)

    path_Lam = LambdaPanel(
        with_event_marks=chart_config.with_event_marks,
        show_derivations=show_derivations,
        clip_opts=ClipOptions(
            pctl_upper=chart_config.lambda_pctl_upper,
            pctl_lower=chart_config.lambda_pctl_lower,
            warmup_hours=chart_config.lambda_warmup_hours,
        ),
    ).plot(metrics, filter_result, chart_config, out_dir)

    path_Theta = ThetaPanel(
        with_event_marks=chart_config.with_event_marks,
        show_derivations=show_derivations,
    ).plot(metrics, filter_result, chart_config, out_dir)

    path_indicator = EventIndicatorPanel(
        with_event_marks=chart_config.with_event_marks,
    ).plot(metrics, filter_result, chart_config, out_dir)

    path_A = ArrivalsPanel(
        with_event_marks=chart_config.with_event_marks,
        show_derivations=show_derivations,
    ).plot(metrics, filter_result, chart_config, out_dir)

    path_D = DeparturesPanel(
        with_event_marks=chart_config.with_event_marks,
        show_derivations=show_derivations,
    ).plot(metrics, filter_result, chart_config, out_dir)

    path_w = WPanel(
        with_event_marks=chart_config.with_event_marks,
        show_derivations=show_derivations,
    ).plot(metrics, filter_result, chart_config, out_dir)

    path_w_prime = WPrimePanel(
        with_event_marks=chart_config.with_event_marks,
        show_derivations=show_derivations,
    ).plot(metrics, filter_result, chart_config, out_dir)

    path_H = HPanel(
        show_derivations=show_derivations,
        with_event_marks=chart_config.with_event_marks,
    ).plot(metrics, filter_result, chart_config, out_dir)

    path_CFD = CFDPanel(
        with_event_marks=chart_config.with_event_marks,
        show_derivations=show_derivations,
    ).plot(metrics, filter_result, chart_config, out_dir)

    path_invariant = LLWPanel(
        with_event_marks=chart_config.with_event_marks,
    ).plot(metrics, filter_result, chart_config, out_dir)

    path_departure_invariant = LThetaWPrimePanel(
        with_event_marks=chart_config.with_event_marks,
    ).plot(metrics, filter_result, chart_config, out_dir)

    return [
        path_N,
        path_L,
        path_Lam,
        path_Theta,
        path_indicator,
        path_A,
        path_D,
        path_w,
        path_w_prime,
        path_H,
        path_CFD,
        path_invariant,
        path_departure_invariant,
        path_stack,
        path_LT_derivation,
        path_departure_stack,
    ]
