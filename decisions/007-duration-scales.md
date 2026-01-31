# ADR-007: Duration Scales and Seconds-Native Metrics

**Status:** Implemented  
**Date:** 2026-01-31  
**Author:** Krishna Kumar  

## Context

Flow metrics were historically computed and displayed in hours and 1/hr. As the plotting layer expanded, unit conversions became inconsistent across modules, leading to mismatched labels and values. We needed a consistent internal unit and a single, reusable way to infer human-readable display units.

## Decision

1) **Normalize internal duration and rate metrics to seconds.**
   - Metrics such as `w(T)`, `W*(T)`, `w'(T)`, `H(T)` are stored in seconds.
   - Rate metrics such as `Lambda(T)`, `Theta(T)`, and `lam_star` are stored in 1/second.

2) **Infer a display `DurationScale` from the data and apply it in the plot layer.**
   - `DurationScale` provides a `divisor`, `label`, and `rate_label` (e.g., sec/min/hr/day).
   - `ChartConfig` carries an optional `duration_scale` (auto-inferred when `None`).
   - Plot panels convert values for display:
     - Durations and `H(T)` are divided by `scale.divisor`.
     - Rates are multiplied by `scale.divisor`.

3) **Move warmup/horizon boundaries to seconds at the chart boundary.**
   - `lambda_warmup_seconds` and `horizon_seconds` are computed in `ChartConfig`.
   - Downstream percentile clipping and convergence checks use seconds.

4) **Fix unit mismatches in advanced/stability calculations.**
   - `compute_end_effect_series`, `compute_tracking_errors`, and `compute_total_active_age_series`
     now operate in seconds, removing mixed-unit computations.

## Scope

- `samplepath/metrics.py` produces seconds-native metrics.
- `samplepath/utils/duration_scale.py` defines the scale ladder and inference.
- `samplepath/plots/*` uses `DurationScale` to label and scale displayed values.
- Tests updated to assert seconds-native metrics and correct display scaling behavior.

## Consequences

- **Pros:**
  - One canonical unit for computation (seconds), preventing mixed-unit bugs.
  - Consistent, data-driven labels across charts.
  - Display scaling is centralized and testable.

- **Cons:**
  - Plotting code must always apply `DurationScale` when presenting durations or rates.
  - Some tests and helper functions required unit migration and re-baselining.

## Follow-ups

- End-to-end validation pass for chart outputs using representative datasets.
- Continue migrating remaining plot helpers to use display scaling consistently.
