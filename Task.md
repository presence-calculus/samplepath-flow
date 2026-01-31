---
ID: 26
Task: Calendar indexed flow metrics
Branch: calendar-indexed-flow-metrics
---

Spec: We are going to adapt the existing charts so that they work correctly when calendar indexes
are provided rather than time stamps. In theory, much of this code should "just work", but in practice we
may have issues. This is an initial exploration to see how far we can get.


## Completed subtasks

| Commit | Subtask | Summary |
|---|---|---|
| 9b39e4a | 26.1: Add `--sampling-frequency` CLI argument | Wire up `--sampling-frequency` to `compute_finite_window_flow_metrics` and produce calendar-boundary observation times. |
| 42d7a7c | 26.2: Support non-fixed frequencies | Handle week/month/quarter/year offsets in `_align_to_boundary` and `_build_calendar_times`. |
| 23c549c | 26.3: Add `--anchor` CLI argument | Add `--anchor` for week day / quarter-year month and thread through to `_resolve_freq`. |
| 1627e71 | 26.4: Add markers to line charts in calendar mode | Add `marker="o"` to `render_line_chart` when `sampling_frequency` is set so individual data points are visible. |
| 28b3640 | 26.5: Calendar-aware x-axis tick formatting | Add `_calendar_tick_config` and `apply_calendar_ticks` in helpers; call from `format_date_axis` and `_format_axis_label`. |
| dee0eac | Fix: Graceful returns in limits.py | Replace hard asserts in `estimate_limit`, `estimate_linear_rate`, `compare_series_tail` with graceful `LimitResult(method="insufficient_data")` returns for small datasets. |
| 4bf0b38 | 26.5b: Human-readable x-axis labels | Add `ChartConfig.freq_display_label()` static method; labels now show `"Time (month)"` instead of `"Date (MS)"`. |
| 1678f0b | Fix: FutureWarning in `_is_fixed_frequency` | Replace deprecated `hasattr(off, "delta")` with `isinstance(off, pd.tseries.offsets.Tick)`. |
| 73656b1 | 26.6: Rug plot event overlays in calendar mode | Add `calendar_mode` param to `build_event_overlays`; when True, all events placed at y=0 (rug plot) with drop lines disabled. |
