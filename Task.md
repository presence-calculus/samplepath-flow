---
ID: 37
Task: Analytical curves for time normalized metrics
Branch: analytical-curves
---

Review the task spec in specs/00014-analytical-curve-plots.md.

## Code Review Feedback — Full Analytical Curves Implementation

All 601 tests pass. 2154 lines changed across 8 files.

### Scope covered

| Metric | `analytic_curves.py` builder | `core.py` panel | `convergence.py` panel | `advanced.py` |
|--------|------------------------------|-----------------|------------------------|---------------|
| L(T)   | `build_L_curve`              | LPanel          | -                      | -             |
| Lambda | `build_lambda_curve`         | LambdaPanel     | ArrDep, CumArrival     | dyn_conv x2   |
| Theta  | `build_theta_curve`          | ThetaPanel      | ArrDep                 | -             |
| w(T)   | `build_w_curve`              | WPanel          | ProcessTime, Sojourn   | -             |
| w'(T)  | `build_w_prime_curve`        | WPrimePanel     | ProcessTime, Sojourn   | -             |

### What's correct

- All five builder functions implement the correct formulas per spec.
- `build_lambda_curve`/`build_theta_curve` emit jump metadata (spec sec 3) with
  correct left/right limits.
- `_build_residence_curve` (shared helper for w/w') correctly guards against
  division by zero when counts=0 and only emits jumps when both sides are defined.
- Event overlays consistently use original event-indexed values, not the dense
  analytic samples — preserving marker placement at the correct event points.
- Calendar mode guards (`sampling_frequency is None`) consistently skip analytic
  curves in all panels.
- Fallback behavior is safe everywhere: missing state vectors or empty builder
  output gracefully degrades to the original polyline.
- Jump rendering uses `ax.vlines` with proper scaling (rates multiplied by
  `duration_scale.divisor`; durations divided by it).
- Tests cover: analytic curve wiring, calendar mode skip, missing-args fallback,
  jump rendering with scale, overlay isolation, and `.plot()` state vector pass-through
  for all five metrics.

### Code feedback

**1. `plot_core_stack` — WPanel call missing state vectors (bug)**
`core.py:1925-1936`: `WPanel.render()` is called without `H_vals`, `N_vals`, or
`arrivals_vals`. The W panel in the core stack silently falls back to the
original polyline — the analytic curve is never used here.

Fix: add the three arguments matching the pattern in `WPanel.plot()`:
```
H_vals=metrics.H,
N_vals=metrics.N,
arrivals_vals=metrics.Arrivals,
```

**2. `plot_departure_flow_metrics_stack` — WPrimePanel call missing state vectors (bug)**
`core.py:2090-2101`: `WPrimePanel.render()` is called without `H_vals`, `N_vals`,
or `departures_vals`. Same issue — the w' analytic curve is never used in
this stack.

Fix: add the three arguments matching the pattern in `WPrimePanel.plot()`:
```
H_vals=metrics.H,
N_vals=metrics.N,
departures_vals=metrics.Departures,
```

### Testing feedback

**3. Add test: `plot_core_stack` passes W state vectors**
Once bug #1 is fixed, add a test verifying `WPanel.render` receives `H_vals`
when called from `plot_core_stack`.

**4. Add test: `plot_departure_flow_metrics_stack` passes WPrime state vectors**
Once bug #2 is fixed, add a test verifying `WPrimePanel.render` receives
`H_vals` when called from `plot_departure_flow_metrics_stack`.

### Observations (non-blocking)

**5. `ThetaPanel` and `WPrimePanel` got explicit `color="tab:blue"`.**
This is a cosmetic change beyond the analytical curves scope. It makes these
panels consistent with LambdaPanel and WPanel which already had it. Not wrong,
but it changes default color behavior.

**6. `advanced.py` — w(T) in `draw_dynamic_convergence_panel_with_errors` still
uses direct `axes[0].plot()`.**
Only Lambda gets the analytic curve treatment there. The w(T) line uses the
original polyline via `axes[0].plot(times, w_scaled, ...)`. This may be
intentional since these functions have a different rendering path, but worth
confirming.
