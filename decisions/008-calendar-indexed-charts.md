# ADR-008: Calendar-Indexed Observation Times and Chart Adaptations

**Status:** Implemented
**Date:** 2026-01-31
**Author:** Krishna Kumar

## Context

Flow metrics were originally computed and plotted at event timestamps only. Users working with calendar-aligned reporting (weekly sprints, monthly reviews, quarterly planning) needed observation times snapped to calendar boundaries. This required changes across the metrics engine, CLI, chart renderers, and event overlay system.

## Decisions

### 1. Calendar boundary alignment via `_align_to_boundary`

Pandas `Timestamp.floor()` only works for fixed-frequency offsets (day, hour, minute). For variable-width offsets (week, month, quarter, year) a probe-based alignment function `_align_to_boundary` finds the calendar boundary at or before a given timestamp by generating a short `date_range` and picking the last boundary that does not exceed the target.

### 2. Human-friendly frequency aliases

The CLI accepts human-readable frequency names (`day`, `week`, `month`, `quarter`, `year`) via `--sampling-frequency`, resolved internally to pandas offset aliases (`D`, `W-SUN`, `MS`, `QS-JAN`, `YS-JAN`). An optional `--anchor` argument customises the boundary anchor (e.g., `--anchor WED` for week, `--anchor APR` for quarter/year).

### 3. `sampling_frequency` threaded through panels, not renderers

Rather than embedding calendar awareness deep in renderers, each panel dataclass gained a `sampling_frequency` field. Panels pass it down to `render_line_chart` and `render_step_chart`, which use it to decide marker style and color de-emphasis. `ChartConfig` carries the value from the CLI to every panel instantiation site.

### 4. Color de-emphasis moved into renderers, conditional on mode

Previously, panels hard-coded `color="grey"` when overlays were present. This logic moved into `render_line_chart` and `render_step_chart`: grey is applied only when overlays exist *and* `sampling_frequency is None` (timestamp mode). In calendar mode the original series color is preserved because overlays render as rug plots at y=0, making de-emphasis unnecessary.

### 5. Rug-plot overlays in calendar mode

When `calendar_mode=True`, `build_event_overlays` places all events at y=0 with drop lines disabled. This avoids the mismatch where calendar-boundary observation times never coincide with actual event timestamps, which would cause events to be silently dropped in the default lookup-based overlay logic.

### 6. Calendar-aware x-axis tick formatting

`_calendar_tick_config` maps pandas offset types to appropriate matplotlib `DateLocator`/`DateFormatter` pairs (e.g., `WeekdayLocator` for weekly, `MonthLocator` for monthly/quarterly, `YearLocator` for yearly). `apply_calendar_ticks` applies these to an axis. `ChartConfig.freq_display_label()` produces human-readable axis labels like `"Time (week-MON)"` instead of raw pandas aliases.

### 7. Graceful degradation in limits.py

Calendar-sampled series can be short (e.g., quarterly data over a year yields ~4 points). Hard `assert` guards in `estimate_limit`, `estimate_linear_rate`, and `compare_series_tail` were replaced with early returns of `LimitResult(method="insufficient_data")`, preventing crashes on small datasets.

### 8. FutureWarning fix in `_is_fixed_frequency`

The deprecated `hasattr(off, "delta")` check was replaced with `isinstance(off, pd.tseries.offsets.Tick)` to suppress a pandas `FutureWarning` and align with the recommended API.

## Scope

- `samplepath/cli.py` -- `--sampling-frequency` and `--anchor` arguments.
- `samplepath/metrics.py` -- `_align_to_boundary`, `_is_fixed_frequency`, `_resolve_freq` updates.
- `samplepath/plots/chart_config.py` -- `sampling_frequency` field, `freq_display_label()`.
- `samplepath/plots/helpers.py` -- `_calendar_tick_config`, `apply_calendar_ticks`, conditional color/marker logic in renderers, `calendar_mode` in `build_event_overlays`.
- `samplepath/plots/figure_context.py` -- wired `apply_calendar_ticks` and human labels into `_format_axis_label`.
- `samplepath/plots/core.py`, `convergence.py` -- `sampling_frequency` threaded through all panels.
- `samplepath/limits.py` -- assert-to-graceful-return conversion.
- `samplepath/sample_path_analysis.py` -- anchor passthrough to `compute_finite_window_flow_metrics`.

## Consequences

**Pros:**
- All existing charts work unchanged in timestamp mode; calendar mode is purely additive.
- Calendar-aware ticks and labels make charts readable without manual matplotlib configuration.
- Rug-plot overlays preserve event visibility even when observation boundaries don't align with events.
- Graceful limit estimation prevents crashes on short calendar-sampled series.

**Cons:**
- `sampling_frequency` is now a parameter on every panel dataclass and both core renderers, adding surface area.
- The probe-based `_align_to_boundary` is an approximation; edge cases with exotic offsets may need attention.
