---
ID: 27
Task: Duration units
Branch: duration-units
---

Spec: Currently we are hardcoding all the calculations of time durations to be in hours. This is impractical when we are looking at it in the UI over long time scales. We want to retain all durations in its finest resolution for computations but convert it to a suitable scale when displaying. Ideally we should just do this smartly, ie when have the list of residence times we can simply use the min and max values to infer whether it makes sense to show the time scale in hours, days, weeks, months, quarters etc. We can add the ability to override this with a cli argument, but that should not be needed.

## Implementation Plan

### Design decisions

- **Internal unit: seconds.** All `/3600.0` divisors are removed from
  `metrics.py`. Durations use raw `.total_seconds()`. This eliminates
  the hardcoded hour assumption at the computation layer.
- **Display conversion at the chart layer.** A scale-inference routine
  picks the best human-readable unit (minutes, hours, days, weeks) from
  the data range. Charts apply a divisor to y-values and use
  corresponding labels.
- **Rate units match duration units.** If durations display in days,
  Lambda/Theta display in 1/day. This keeps Little's Law
  (L = Lambda · w) visually consistent on charts.
- **`duration_hr` column stays.** The `csv_loader.py` `duration_hr`
  column is used by the filter layer (`outlier-hours`, `outlier-pctl`,
  `outlier-iqr`). This is a user-facing pre-processing boundary where
  hours is the documented CLI contract. Changing it is out of scope.
- **CLI warmup/horizon parameters convert at the boundary.**
  `lambda_warmup_hours` and `horizon_days` enter as user-facing values;
  ChartConfig converts them to seconds internally.

### Subtask breakdown

#### 27.1 — DurationScale dataclass + `infer_duration_scale` routine

New module: `samplepath/utils/duration_scale.py`

```python
@dataclass(frozen=True)
class DurationScale:
    name: str           # "seconds", "minutes", "hours", "days", "weeks"
    label: str          # "sec", "min", "hrs", "days", "weeks"
    divisor: float      # from seconds: 1, 60, 3600, 86400, 604800
    rate_label: str     # "1/sec", "1/min", "1/hr", "1/day", "1/week"

SCALES = [seconds, minutes, hours, days, weeks]  # ascending divisor

def infer_duration_scale(values_seconds: np.ndarray) -> DurationScale:
    """Pick the scale where the max value falls in a readable range."""
```

Logic: take the max of `abs(values)` (ignoring NaN), walk up the scale
ladder until the scaled max is < ~1000 (or the next scale would push
it < 1). Default to hours when values are empty.

Tests: parametrized over ranges that should resolve to each scale
(e.g. max=120s → minutes, max=7200s → hours, max=200000s → days, etc.).

#### 27.1 Review feedback

- The current implementation is missing a seconds scale and excludes seconds from the scale ladder, which conflicts with the spec (seconds, minutes, hours, days, weeks). Sub-minute values will always map to minutes as written.
- The inference logic doesn’t enforce an upper “readable range” bound (e.g., ~1000) described in the spec; it only ensures scaled max ≥ 1, which can still yield very large values.
- Tests currently encode the minutes-only behavior for sub-minute values; they will need to reflect the spec once seconds is included.


#### 27.1 Feedback response:

To align with the spec, the requirement is:

1) Add a **seconds** scale (`name="seconds"`, `label="sec"`, `divisor=1.0`, `rate_label="1/sec"`) and include it in the scale ladder as the smallest unit.
2) Update `infer_duration_scale` to honor the **readable range** rule: choose the **largest** unit such that the scaled max is **< ~1000** and **≥ 1**. If none meet both bounds, fall back to the largest unit with scaled max ≥ 1; if values are empty/all‑NaN, default to hours.
3) Update tests to reflect the seconds scale and the upper‑bound behavior (sub‑minute values → seconds; thresholds around 1000 should push to the next unit when possible).
  - seconds: 200,000 (>= 1, continue)  
  - minutes: 3,333 (>= 1, continue)  
  - hours: 55.6 (>= 1, continue)  
  - days: 2.3 (>= 1, continue)  
  - weeks: 0.33 (< 1, stop)


#### 27.2 — Remove `/3600.0` from `metrics.py`

In `compute_finite_window_flow_metrics` (lines 196, 211, 217):
- `dt_h = (t_ev - prev).total_seconds() / 3600.0` → `dt = (t_ev - prev).total_seconds()`
- Same for `elapsed_h` → `elapsed`
- H, Lambda, w, w_prime, Theta now in seconds-based units

In `compute_elementwise_empirical_metrics` (lines 543, 567, 579, 586):
- Remove all `/ 3600.0` from sojourn/residence/elapsed computations

Update `FlowMetricsResult` and `ElementWiseEmpiricalMetrics` docstrings:
comments change from "processes/hour" → "processes/second", "hours" →
"seconds", etc.

Rename internal variables: `dt_h` → `dt_s`, `elapsed_h` → `elapsed_s`.

**No changes to filter.py or csv_loader.py** — `duration_hr` is a
separate user-facing column used at the filter boundary.



#### 27.3 — Wire `DurationScale` through `ChartConfig`

Add to `ChartConfig`:
```python
duration_scale: Optional[DurationScale] = None  # None = auto-infer
```

Add `ChartConfig.with_duration_scale(scale)` or make it a mutable
post-init step.

In `sample_path_analysis.py` → `produce_all_charts`:
- After metrics + empirical_metrics are computed, call
  `infer_duration_scale` on the w(T) array (or sojourn_vals)
- Pass into ChartConfig (or create a new ChartConfig with the scale)

Convert `lambda_warmup_hours` and `horizon_days` to seconds at the
`ChartConfig.init_from_args` boundary:
- `lambda_warmup_seconds: float` (= input * 3600)
- `horizon_seconds: float` (= input * 86400)

Update downstream consumers of these fields (`_clip_axis_to_percentile`,
convergence, stability, advanced) to use seconds.

#### 27.4 — Update chart labels and y-values in `core.py`

Each panel that currently hardcodes `[hrs]` or `[1/hr]`:

| Panel | Current label | New label source |
|---|---|---|
| LambdaPanel | `Λ(T) [1/hr]` | `f"Λ(T) [{scale.rate_label}]"` |
| ThetaPanel | `Θ(T) [1/hr]` | `f"Θ(T) [{scale.rate_label}]"` |
| WPanel | `w(T) [hrs]` | `f"w(T) [{scale.label}]"` |
| SojournTimePanel | `W*(T) [hrs]` | `f"W*(T) [{scale.label}]"` |
| HPanel | `H(T) [hrs·items]` | `f"H(T) [{scale.label}·items]"` |
| WPrimePanel | `w'(T) [hrs]` | `f"w'(T) [{scale.label}]"` |
| SojournTimeScatterPanel | `Duration [hrs]` | `f"Duration [{scale.label}]"` |
| ResidenceTimeScatterPanel | `Duration [hrs]` | `f"Duration [{scale.label}]"` |

For duration-valued panels (W, W*, w', sojourn scatter, residence
scatter): divide y-values by `scale.divisor` before passing to render.

For rate-valued panels (Lambda, Theta): multiply y-values by
`scale.divisor` before passing to render.

For H: divide by `scale.divisor`.

L and N: no conversion (dimensionless).

Panels need access to the DurationScale — either via a new field on each
panel dataclass or via ChartConfig (which already flows to panels).

#### 27.5 — Update chart labels in `convergence.py`, `advanced.py`, `stability.py`

Same pattern: replace hardcoded `[hrs]`, `[1/hr]`, `hours` labels with
scale-derived labels, and apply divisor/multiplier to y-values.

Key locations:
- `convergence.py`: ProcessTimeConvergencePanel, RecurrenceConvergencePanel,
  `_plot_little_law_convergence`, `_plot_arrival_rate_convergence`,
  `_plot_llaw_coherence`
- `advanced.py`: `draw_dynamic_convergence_panel`,
  `draw_dynamic_convergence_panel_with_errors`, `compute_end_effects_errors`
- `stability.py`: `compute_total_active_age_series`,
  `plot_stability_analysis`

All `/3600.0` conversions in these files are replaced with
`.total_seconds()` (raw seconds), and display labels come from the
DurationScale.


# 27.5 Feedback

 Issue: compute_end_effect_series has a unit mismatch (bug)  

  The spec says: "All /3600.0 conversions in these files are replaced with .total_seconds() (raw seconds)"  

  Three internal computation functions were not converted from hours to seconds:  

  1. compute_end_effect_series (advanced.py:162-218) — This is a runtime bug:  
  - H_T = H_vals[i] — now seconds (from metrics.H after 27.2)  
  - H_full = df["duration_h"].sum() — still hours (line 180: / 3600.0)  
  - E_T = H_T - H_full → seconds minus hours = wrong  
  - rho = elapsed_h / Wstar_t where elapsed_h is hours (line 187) but Wstar_t is seconds → wrong  

  The test_end_effects.py fixtures apply a /3600.0 bridge to mask this, but plot_residence_time_convergence_error_charts (line 480-481) passes metrics.H and W_star_ts in raw seconds, triggering the mismatch at runtime.  

  2. compute_tracking_errors (advanced.py:223-247) — Less critical: the relative errors eW and eLam are correct (units cancel in the ratio), but elapsed_hours return value is in hours. compute_coherence_score (line 253) compares  
  elapsed_hours >= horizon_hours — the parameter name says hours but the caller would need to know to convert from chart_config.horizon_seconds.  

  3. compute_total_active_age_series (stability.py:24-103) — Returns R in hours (uses 3.6e12 ns→hours conversion). The workaround R_raw_scaled = R_raw * 3600.0 / duration_scale.divisor on line 152 converts hours→seconds→display_unit,  
  which is correct but is a patch over the underlying inconsistency.  

  Recommendation  

  Fix compute_end_effect_series before committing — it produces incorrect rH and rho at runtime. The fix: change duration_h to duration_s using raw .total_seconds(), and elapsed_h to elapsed_s. Then remove the /3600.0 bridge in  
  test_end_effects.py.  

  For compute_total_active_age_series, either convert the 3.6e12 → 1e9 (ns→seconds) or keep the workaround, but the current approach works correctly.  



#### 27.6 — Update `helpers.py` standalone recipes

`render_N_chart` and `render_LT_chart` don't reference hours.
`_clip_axis_to_percentile` parameter `warmup_hours` → `warmup_seconds`
(callers already converted at the ChartConfig boundary).

### Files modified (by subtask)

| Subtask | Files |
|---|---|
| 27.1 | `samplepath/utils/duration_scale.py` (new), `test/test_duration_scale.py` (new) |
| 27.2 | `samplepath/metrics.py`, `test/metrics/test_finite_window_flow_metrics.py` |
| 27.3 | `samplepath/plots/chart_config.py`, `samplepath/sample_path_analysis.py`, `test/test_figure_context.py` |
| 27.4 | `samplepath/plots/core.py`, `test/test_core_plots.py` |
| 27.5 | `samplepath/plots/convergence.py`, `samplepath/plots/advanced.py`, `samplepath/plots/stability.py` + their tests |
| 27.6 | `samplepath/plots/helpers.py`, `test/test_plots.py` |
