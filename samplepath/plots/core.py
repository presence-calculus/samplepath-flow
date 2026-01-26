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
import os
from typing import List, Optional, Sequence

from matplotlib import colors as mcolors, pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

from samplepath.filter import FilterResult
from samplepath.metrics import FlowMetricsResult, MetricDerivations
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
    add_caption,
    build_event_overlays,
    format_date_axis,
    render_line_chart,
    render_scatter_chart,
    render_step_chart,
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
        out_path: str,
        times: List[pd.Timestamp],
        N_vals: Sequence[float],
        *,
        arrival_times: Optional[List[pd.Timestamp]] = None,
        departure_times: Optional[List[pd.Timestamp]] = None,
        unit: str = "timestamp",
        caption: Optional[str] = None,
    ) -> None:
        with figure_context(out_path, nrows=1, ncols=1, unit=unit, caption=caption) as (
            _,
            axes,
        ):
            ax = _first_axis(axes)
            self.render(
                ax,
                times,
                N_vals,
                arrival_times=arrival_times,
                departure_times=departure_times,
            )


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
        out_path: str,
        times: List[pd.Timestamp],
        L_vals: Sequence[float],
        *,
        arrival_times: Optional[List[pd.Timestamp]] = None,
        departure_times: Optional[List[pd.Timestamp]] = None,
        unit: str = "timestamp",
        caption: Optional[str] = None,
    ) -> None:
        with figure_context(out_path, nrows=1, ncols=1, unit=unit, caption=caption) as (
            _,
            axes,
        ):
            ax = _first_axis(axes)
            self.render(
                ax,
                times,
                L_vals,
                arrival_times=arrival_times,
                departure_times=departure_times,
            )


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
            build_event_overlays(times, Lam_vals, arrival_times, [], drop_lines=True)
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
        out_path: str,
        times: List[pd.Timestamp],
        Lam_vals: Sequence[float],
        *,
        arrival_times: Optional[List[pd.Timestamp]] = None,
        unit: str = "timestamp",
        caption: Optional[str] = None,
    ) -> None:
        with figure_context(out_path, nrows=1, ncols=1, unit=unit, caption=caption) as (
            _,
            axes,
        ):
            ax = _first_axis(axes)
            self.render(
                ax,
                times,
                Lam_vals,
                arrival_times=arrival_times,
            )


@dataclass
class WPanel:
    show_title: bool = True
    title: str = "w(T) — Average Residence Time"
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
                drop_lines=True,
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
        out_path: str,
        times: List[pd.Timestamp],
        w_vals: Sequence[float],
        *,
        arrival_times: Optional[List[pd.Timestamp]] = None,
        departure_times: Optional[List[pd.Timestamp]] = None,
        unit: str = "timestamp",
        caption: Optional[str] = None,
    ) -> None:
        with figure_context(out_path, nrows=1, ncols=1, unit=unit, caption=caption) as (
            _,
            axes,
        ):
            ax = _first_axis(axes)
            self.render(
                ax,
                times,
                w_vals,
                arrival_times=arrival_times,
                departure_times=departure_times,
            )


@dataclass
class HPanel:
    show_title: bool = True
    title: str = "H(T) — Cumulative Presence Mass"
    show_derivations: bool = False

    def render(
        self,
        ax,
        times: Sequence[pd.Timestamp],
        H_vals: Sequence[float],
    ) -> None:
        render_line_chart(ax, times, H_vals, label="H(T) [hrs·items]", color="tab:blue")
        if self.show_title:
            ax.set_title(
                construct_title(self.title, self.show_derivations, derivation_key="H")
            )
        ax.set_ylabel("H(T) [hrs·items]")
        ax.legend()

    def plot(
        self,
        out_path: str,
        times: List[pd.Timestamp],
        H_vals: Sequence[float],
        *,
        unit: str = "timestamp",
        caption: Optional[str] = None,
    ) -> None:
        with figure_context(out_path, nrows=1, ncols=1, unit=unit, caption=caption) as (
            _,
            axes,
        ):
            ax = _first_axis(axes)
            self.render(
                ax,
                times,
                H_vals,
            )


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
        out_path: str,
        times: List[pd.Timestamp],
        arrivals_cum: Sequence[float],
        departures_cum: Sequence[float],
        *,
        arrival_times: Optional[List[pd.Timestamp]] = None,
        departure_times: Optional[List[pd.Timestamp]] = None,
        unit: str = "timestamp",
        caption: Optional[str] = None,
    ) -> None:
        with figure_context(out_path, nrows=1, ncols=1, unit=unit, caption=caption) as (
            _,
            axes,
        ):
            ax = _first_axis(axes)
            self.render(
                ax,
                times,
                arrivals_cum,
                departures_cum,
                arrival_times=arrival_times,
                departure_times=departure_times,
            )


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
            ax.vlines(
                x,
                0,
                y,
                colors=drop_colors,
                linewidths=0.5,
                alpha=0.25,
            )
            ax.hlines(
                y,
                0,
                x,
                colors=drop_colors,
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
        out_path: str,
        times: List[pd.Timestamp],
        L_vals: np.ndarray,
        Lam_vals: np.ndarray,
        w_vals: np.ndarray,
        *,
        arrival_times: Optional[List[pd.Timestamp]] = None,
        departure_times: Optional[List[pd.Timestamp]] = None,
        caption: Optional[str] = None,
    ) -> None:
        fig, ax = plt.subplots(figsize=(6.0, 6.0))
        self.render(
            ax,
            times,
            L_vals,
            Lam_vals,
            w_vals,
            arrival_times=arrival_times,
            departure_times=departure_times,
        )
        if caption:
            add_caption(fig, caption)
        fig.tight_layout(rect=(0.05, 0, 1, 1))
        fig.savefig(out_path)
        plt.close(fig)


# ---------------------------------------------------------------------------
# Composite stack (4-panel)
# ---------------------------------------------------------------------------


def plot_core_stack(
    out_path: str,
    times: List[pd.Timestamp],
    N_vals: np.ndarray,
    L_vals: np.ndarray,
    Lam_vals: np.ndarray,
    w_vals: np.ndarray,
    *,
    lambda_pctl_upper: Optional[float] = None,
    lambda_pctl_lower: Optional[float] = None,
    lambda_warmup_hours: float = 0.0,
    caption: Optional[str] = None,
    arrival_times: Optional[List[pd.Timestamp]] = None,
    departure_times: Optional[List[pd.Timestamp]] = None,
    with_event_marks: bool = False,
    unit: str = "timestamp",
    show_title: bool = True,
    title_N: str = "N(t) — Sample Path",
    title_L: str = "L(T) — Time-Average of N(t)",
    title_Lambda: str = "Λ(T) — Cumulative Arrival Rate",
    title_w: str = "w(T) — Average Residence Time",
    show_derivations: bool = False,
) -> None:
    layout = LayoutSpec(nrows=4, ncols=1, figsize=(12.0, 11.0), sharex=True)
    decor = FigureDecorSpec(
        suptitle="Sample Path Flow Metrics",
        suptitle_y=0.97,
        caption=caption,
        caption_position="top",
        caption_y=0.945,
        tight_layout=True,
        tight_layout_rect=(0, 0, 1, 0.96),
    )
    with layout_context(
        out_path,
        layout=layout,
        decor=decor,
        unit=unit,
        format_axis_fn=format_date_axis,
        format_targets="bottom_row",
    ) as (_, axes):
        flat_axes = axes if not isinstance(axes, np.ndarray) else axes.ravel()

        NPanel(
            show_title=show_title,
            title=title_N,
            show_derivations=show_derivations,
            with_event_marks=with_event_marks,
        ).render(
            flat_axes[0],
            times,
            N_vals,
            arrival_times=arrival_times,
            departure_times=departure_times,
        )
        LPanel(
            show_title=show_title,
            title=title_L,
            show_derivations=show_derivations,
            with_event_marks=with_event_marks,
        ).render(
            flat_axes[1],
            times,
            L_vals,
            arrival_times=arrival_times,
            departure_times=departure_times,
        )
        LambdaPanel(
            show_title=show_title,
            title=title_Lambda,
            show_derivations=show_derivations,
            with_event_marks=with_event_marks,
            clip_opts=ClipOptions(
                pctl_upper=lambda_pctl_upper,
                pctl_lower=lambda_pctl_lower,
                warmup_hours=lambda_warmup_hours,
            ),
        ).render(
            flat_axes[2],
            times,
            Lam_vals,
            arrival_times=arrival_times,
        )
        WPanel(
            show_title=show_title,
            title=title_w,
            show_derivations=show_derivations,
            with_event_marks=with_event_marks,
        ).render(
            flat_axes[3],
            times,
            w_vals,
            arrival_times=arrival_times,
            departure_times=departure_times,
        )


def plot_core_flow_metrics_charts(
    df: pd.DataFrame,
    args,
    filter_result: Optional[FilterResult],
    metrics: FlowMetricsResult,
    out_dir: str,
) -> List[str]:
    del df
    core_panels_dir = os.path.join(out_dir, "core")
    unit = metrics.freq if metrics.freq else "timestamp"
    caption = (
        filter_result.display if filter_result and filter_result.label else "Filters: "
    )
    show_derivations = getattr(args, "show_derivations", False)

    path_stack = os.path.join(out_dir, "sample_path_flow_metrics.png")
    plot_core_stack(
        path_stack,
        metrics.times,
        metrics.N,
        metrics.L,
        metrics.Lambda,
        metrics.w,
        lambda_pctl_upper=args.lambda_pctl,
        lambda_pctl_lower=args.lambda_lower_pctl,
        lambda_warmup_hours=args.lambda_warmup,
        caption=caption,
        arrival_times=metrics.arrival_times,
        departure_times=metrics.departure_times,
        with_event_marks=getattr(args, "with_event_marks", False),
        show_derivations=show_derivations,
        unit=unit,
    )

    path_N = os.path.join(core_panels_dir, "sample_path_N.png")
    NPanel(
        with_event_marks=getattr(args, "with_event_marks", False),
        show_derivations=show_derivations,
    ).plot(
        path_N,
        metrics.times,
        metrics.N,
        arrival_times=metrics.arrival_times,
        departure_times=metrics.departure_times,
        unit=unit,
        caption=caption,
    )

    path_L = os.path.join(core_panels_dir, "time_average_N_L.png")
    LPanel(
        with_event_marks=getattr(args, "with_event_marks", False),
        show_derivations=show_derivations,
    ).plot(
        path_L,
        metrics.times,
        metrics.L,
        arrival_times=metrics.arrival_times,
        departure_times=metrics.departure_times,
        unit=unit,
        caption=caption,
    )

    path_Lam = os.path.join(core_panels_dir, "cumulative_arrival_rate_Lambda.png")
    LambdaPanel(
        with_event_marks=getattr(args, "with_event_marks", False),
        show_derivations=show_derivations,
        clip_opts=ClipOptions(
            pctl_upper=args.lambda_pctl,
            pctl_lower=args.lambda_lower_pctl,
            warmup_hours=args.lambda_warmup,
        ),
    ).plot(
        path_Lam,
        metrics.times,
        metrics.Lambda,
        arrival_times=metrics.arrival_times,
        unit=unit,
        caption=caption,
    )

    path_w = os.path.join(core_panels_dir, "average_residence_time_w.png")
    WPanel(
        with_event_marks=getattr(args, "with_event_marks", False),
        show_derivations=show_derivations,
    ).plot(
        path_w,
        metrics.times,
        metrics.w,
        arrival_times=metrics.arrival_times,
        departure_times=metrics.departure_times,
        unit=unit,
        caption=caption,
    )

    path_H = os.path.join(core_panels_dir, "cumulative_presence_mass_H.png")
    HPanel(show_derivations=show_derivations).plot(
        path_H,
        metrics.times,
        metrics.H,
        unit=unit,
        caption=caption,
    )

    path_CFD = os.path.join(core_panels_dir, "cumulative_flow_diagram.png")
    CFDPanel(
        with_event_marks=getattr(args, "with_event_marks", False),
        show_derivations=show_derivations,
    ).plot(
        path_CFD,
        metrics.times,
        metrics.Arrivals,
        metrics.Departures,
        arrival_times=metrics.arrival_times,
        departure_times=metrics.departure_times,
        unit=unit,
        caption=caption,
    )

    path_invariant = os.path.join(core_panels_dir, "littles_law_invariant.png")
    LLWPanel(
        with_event_marks=getattr(args, "with_event_marks", False),
        title="L(T) vs Λ(T).w(T)",
    ).plot(
        path_invariant,
        metrics.times,
        metrics.L,
        metrics.Lambda,
        metrics.w,
        arrival_times=metrics.arrival_times,
        departure_times=metrics.departure_times,
        caption=caption,
    )

    return [
        path_N,
        path_L,
        path_Lam,
        path_w,
        path_H,
        path_CFD,
        path_invariant,
        path_stack,
    ]
