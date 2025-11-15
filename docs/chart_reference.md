# Sample Path Chart Reference

This reference describes every chart produced by the `samplepath` CLI, grouped
by chart type. Each section includes cross-links back to the relevant CLI option
groups.

The examples charts drawn from the Polaris example scenario [complete-stories-outliers-removed](../examples/polaris/flow-of-work/complete-stories-outliers-removed) 
This scenario is discussed in more detail in our post [Little's Law in a Complex Adaptive System](https://www.polaris-flow-dispatch.com/i/172332418/sample-path-analysis-a-worked-example)
## Scenario root (top-level under `<scenario>/`)


| File | What it shows | What it means                                                                                                                                         |
| --- | --- |-------------------------------------------------------------------------------------------------------------------------------------------------------|
| `sample_path_flow_metrics.png` | Four-panel vertical stack: `N(t)`, `L(T)`, `Λ(T)`, `w(T)` over the same time axis. | One-glance view of the core finite-window Little’s Law metrics and how they co-evolve along the sample path.                                          |
| `sample_path_convergence.png` | Scatter of `L(T)` (x-axis) vs `λ*(t)·W*(t)` (y-axis) with `y=x` and an ε-band; annotated with a coherence score over a horizon. | Direct visual and quantitative test of whether the finite-window sample path obeys Little’s Law asymptotically (sample-path convergence / coherence). |

---

## `core/` — Core flow metrics

| File | What it shows | What it means                                                                                                                  |
| --- | --- |--------------------------------------------------------------------------------------------------------------------------------|
| `core/sample_path_N.png` | Step chart of `N(t)` (count of elements present in the boundary) vs time. | Raw sample path of WIP/presence: queues, surges, and droughts show up directly.                                                |
| `core/time_average_N_L.png` | Line chart of `L(T)` = time-average of `N(t)` over `[0, T]`. | Tracks how average WIP over the observation window converges (or doesn’t). This is the “L” in Little’s Law, measured pathwise. |
| `core/cumulative_arrival_rate_Lambda.png` | Line chart of `Λ(T)` (cumulative arrival rate `A(T)/(T−t₀)`), with optional percentile clipping and warmup exclusion. | Empirical demand rate over time, with tools to ignore early transients and outliers.                                           |
| `core/average_residence_time_w.png` | Line chart of `w(T)` (average residence time over the window, in hours). | Shows how the time items spend in the boundary evolves; long tails and slow drainage show up as increasing `w(T)`.             |
| `core/littles_law_invariant.png` | Scatter of `L(T)` (x-axis) vs `Λ(T)·w(T)` (y-axis) with `y=x` reference line, equal aspect ratio. | Pure Little’s Law invariant check: all finite points should lie near `y=x` if the metric calculations are consistent.          |

---

## `convergence/` and `convergence/panels/` — Equilibrium & coherence

| File | What it shows | What it means                                                                                                                               |
| --- | --- |---------------------------------------------------------------------------------------------------------------------------------------------|
| `convergence/arrival_departure_equilibrium.png` | Two-row stack: (1) cumulative arrivals `A(t)` vs cumulative departures `D(t)`; (2) `Λ(T)` vs throughput rate `θ(T)=D(T)/(T−t₀)` with masking after last departure. | Tests arrival/departure equilibrium: whether `A(t)` and `D(t)` grow together and arrival/throughput rates converge.                         |
| `convergence/panels/arrival_rate_convergence.png` | Single panel: `Λ(T)` and `λ*(t)` (empirical arrival rate) over time, with optional warmup and percentile-based y-limits. | Compares window-averaged arrival rate to the element-wise empirical rate; checks consistency of the two ways of measuring “arrival rate”.   |
| `convergence/panels/residence_time_convergence.png` | Single panel: `w(T)` vs `W*(t)` (empirical mean sojourn time of completed items) over time. | Coherence between residence-time and sojourn-time views: if the process is coherent, these two series should converge together.             |
| `convergence/residence_sojourn_coherence.png` | Two-row stack: (1) `w(T)` vs `W*(t)` overlay; (2) scatter of individual sojourn times against time, with `w(T)` as a reference line. | Ties the averaged quantities back to individual element sojourn times; helps see whether outliers or subpopulations are driving divergence. |
| `convergence/panels/residence_time_sojourn_time_scatter.png` | Line of `w(T)` over time with overlaid scatter of element *age* (if `--incomplete`) or sojourn time (if `--completed` / default). | Visualizes how individual element ages/sojourn times relate to the evolving average residence time, and whether a stable band emerges.      |

---

## `advanced/` — Error terms, end-effects, and manifold view

| File | What it shows | What it means                                                                                                                                                                         |
| --- | --- |---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `advanced/residence_convergence_errors.png` | Three-row stack: (1) `w(T)` vs `W*(t)` (dynamic); (2) `Λ(T)` vs `λ*(t)` (dynamic); (3) error magnitudes `e_W(T)` and `e_Λ(T)` with optional ε threshold. | Tracks *how* coherence is approached or violated, with explicit error terms for residence time and arrival rate over time.                                                            |
| `advanced/residence_time_convergence_errors_endeffects.png` | Four-row stack: same as above plus an end-effects panel with `r_A(T)` (mass share), `r_B(T)` (boundary share), and `ρ(T)=T/W*(t)`. | Decomposes residual errors into end-effect contributions and time-scaling, making it clear when divergence is driven by boundary conditions vs ongoing dynamics.                      |
| `advanced/invariant_manifold3D_log.png` | 3D log–log–log manifold plot: `x = log Λ(T)`, `y = log w(T)`, `z = log L(T)`, plotting the sample-path trajectory on the plane `z = x + y`. | Geometric view of Little’s Law as an invariant plane; lets you see whether the observed trajectory sticks to the manifold and how it moves across regimes in (rate, time, WIP) space. |

---

## `stability/` and `stability/panels/` — Rate stability

| File | What it shows | What it means                                                                                                                                                     |
| --- | --- |-------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `stability/panels/wip_growth_rate.png` | Two-row stack: (1) `N(t)` step chart; (2) WIP growth rate `N(T)/T` with reference lines at 0 and 1. | Tests whether WIP grows linearly (instability) or sublinearly/flat (bounded or stabilizing), using a rate perspective.                                            |
| `stability/panels/total_age_growth_rate.png` | Two-row stack: (1) total active age `R(t)` in hours; (2) age growth rate `R(T)/T` with reference lines at 0 and 1. | Looks at growth of total age of WIP: sustained growth points to accumulating, aging work and potential instability.                                               |
| `stability/rate_stability.png` | Four-row stack: (1) `N(T)/T`; (2) `R(T)/T`; (3) `λ*(T)`; (4) `w(T)` vs `W*(t)`. Captioned “Equilibrium and Coherence”. | Integrated stability view: WIP and age growth rates, empirical arrival rate, and residence/sojourn coherence all on one canvas to assess long-run rate stability. |

---

# Core Charts

Written under:

```
<scenario>/core/
```

## `sample_path_N.png`
Instantaneous WIP `N(t)` as a step chart.  
Shows congestion, bursts, and idle periods.

## `time_average_N_L.png`
Time-average WIP:

```
L(T) = (1/T) ∫ N(t) dt
```

Reveals whether average WIP stabilizes or drifts.

## `cumulative_arrival_rate_Lambda.png`
Window-average arrival rate:

```
Λ(T) = A(T) / (T − t0)
```

Supports warmup removal and percentile clipping.

## `average_residence_time_w.png`
Finite-window residence time:

```
w(T) = (1/T) ∫ R(t) dt
```

Tracks how “time in system” evolves over the sample path.

## `littles_law_invariant.png`
Scatter of `L(T)` vs `Λ(T)·w(T)` with `y = x`.  
Direct Little’s Law invariance check.

---

# Scenario-Level Summary

Written directly under:

```
<scenario>/
```

## `sample_path_flow_metrics.png`
Four-panel summary:

1. `N(t)`
2. `L(T)`
3. `Λ(T)`
4. `w(T)`

Primary dashboard for the finite-window LL metrics.

## `sample_path_convergence.png`
Scatter of `L(T)` vs `λ*(t)·W*(t)` with tolerance band.  
Highest-level convergence and coherence view.

Controlled by **Convergence Options**:  
`--convergence`, `--coherence-eps`, `--completed`, `--incomplete`, `--warmup`.

---

# Convergence Charts

Written under:

```
<scenario>/convergence/
<scenario>/convergence/panels/
```

---

## `convergence/arrival_departure_equilibrium.png`
Cumulative arrivals vs departures, plus arrival/throughput rate comparison.  
Tests for equilibrium.

## `convergence/panels/arrival_rate_convergence.png`
`Λ(T)` (window-average) and `λ*(t)` (empirical).  
Checks rate consistency.

## `convergence/panels/residence_time_convergence.png`
`w(T)` vs `W*(t)`.  
Coherence between window-average residence and empirical sojourn time.

## `convergence/residence_sojourn_coherence.png`
Two-row comparison of averages + individual sojourn scatter.

## `convergence/panels/residence_time_sojourn_time_scatter.png`
`w(T)` overlaid with individual ages or sojourns.

---

# Stability Charts

Written under:

```
<scenario>/stability/
<scenario>/stability/panels/
```

Controlled by **Stability Options**: `--stability`.

## `stability/panels/wip_growth_rate.png`
`N(t)` and its growth rate `N(T)/T`.  
Detects sublinear vs linear growth.

## `stability/panels/total_age_growth_rate.png`
Total age `R(t)` and its growth rate.  
Identifies aging accumulation.

## `stability/rate_stability.png`
Four-panel stability synthesis:

- `N(T)/T`
- `R(T)/T`
- `λ*(T)`
- `w(T)` vs `W*(t)`

---

# Advanced Charts

Written under:

```
<scenario>/advanced/
```

Controlled by **Advanced Options**: `--advanced`.

## `advanced/residence_convergence_errors.png`
`w(T)` vs `W*(t)`;  
`Λ(T)` vs `λ*(t)`;  
error magnitudes.

## `advanced/residence_time_convergence_errors_endeffects.png`
Adds end-effects: `r_A(T)`, `r_B(T)`, `ρ(T)`.

## `advanced/invariant_manifold3D_log.png`
3D log–log–log manifold:  
`(log Λ(T), log w(T), log L(T))` on the plane `z = x + y`.

---