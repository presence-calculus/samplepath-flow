# Spec: Render Event-Indexed Flow Metrics Using True Analytical Between-Event Curves

## Goal

Update the existing charting code for the event-indexed metrics

- $L(T)$
- $\lambda(T)$ (a.k.a. $\Lambda(T)$)
- $\theta(T)$ (a.k.a. $\Theta(T)$)


so that plots show **true analytical trajectories between events** rather than connecting event-indexed samples with straight-line interpolation.

The new charts must preserve the event-indexed points (optionally as markers), but the connecting geometry must be the **correct between-event curve implied by the sample-path calculus**.

## Definitions and Data Contract

Assume we have an event stream indexed by $i=0,\dots,n-1$ with strictly increasing timestamps:

- $t_i$: event time (calendar time)
- $T_i$: elapsed time since start, i.e. $T_i = t_i - t_0$ expressed in seconds (or any time unit)
- `event_type_i ∈ {A, D}` (arrival or departure)
- $A_i = A(T_i)$: cumulative arrivals at event $i$
- $D_i = D(T_i)$: cumulative departures at event $i$
- $N_i = N(t_i)$: instantaneous presence just after event $i$ (right-continuous convention)
- $H_i = H(T_i)$: cumulative presence at event $i$
- $L_i = L(T_i)$, $\lambda_i = \lambda(T_i)$, $\theta_i = \theta(T_i)$
- $w_i = w(T_i)$, $w'_i = w'(T_i)$

Right-continuous convention:
- For $T \in [T_i, T_{i+1})$, the “state” quantities are taken as $A(T)=A_i$, $D(T)=D_i$, $N(T)=N_i$.

If your current pipeline uses left-continuous quantities, that is OK, but it must be consistent across metrics. This spec assumes right-continuous step functions for counts and $N$.

## Mathematical Model (Between Events)

Let the interval between events be $[T_i, T_{i+1})$ and define $\Delta T_i = T_{i+1} - T_i > 0$.

### Step quantities between events

Between events, there are no new arrivals/departures:

- $A(T) = A_i$ for $T \in [T_i, T_{i+1})$
- $D(T) = D_i$ for $T \in [T_i, T_{i+1})$
- $N(T) = N_i$ for $T \in [T_i, T_{i+1})$

### Cumulative presence is piecewise linear

By definition,

$$
H(T) = \int_0^T N(t)\,dt
$$

and since $N(T)$ is constant on $[T_i, T_{i+1})$,

$$
H(T) = H_i + N_i (T - T_i), \quad T \in [T_i, T_{i+1}).
$$

### Time-normalized metrics introduce curvature

#### Rates

$$
\lambda(T) = \frac{A(T)}{T} = \frac{A_i}{T}, \quad T \in [T_i, T_{i+1}),
$$

$$
\theta(T) = \frac{D(T)}{T} = \frac{D_i}{T}, \quad T \in [T_i, T_{i+1}).
$$

These are hyperbolic (rectangular hyperbola) segments.

At event times:
- If event $i+1$ is an arrival, $A$ jumps by 1 (or by mark), so $\lambda(T)$ has a **vertical jump** at $T_{i+1}$.
- If event $i+1$ is a departure, $D$ jumps, so $\theta(T)$ has a **vertical jump** at $T_{i+1}$.

#### Time-average presence

$$
L(T) = \frac{H(T)}{T}
      = \frac{H_i + N_i (T - T_i)}{T}, \quad T \in [T_i, T_{i+1}).
$$

$L(T)$ is continuous at events (no jumps), but curved between events.

## Implementation Requirements

### 1) Replace linear interpolation with interval sampling

Current implementation likely plots a polyline through $(T_i, m_i)$ points.

Replace this with a renderer that, for each interval $[T_i, T_{i+1})$:

- constructs a small grid of times $T_{i,j}$ within the interval
- evaluates the analytical formula at those $T_{i,j}$
- appends them to a global (x, y) series for plotting

Suggested baseline:

- Use $k=80$ samples per interval (configurable)
- Provide an adaptive option (see below)

### 2) Handle $T=0$ singularity

Metrics that divide by $T$ ($\lambda, \theta, L$) blow up at $T=0$.

Rules:
- Drop the first point from plotting, or
- start plotting at $T \ge T_{\min}$ where $T_{\min}$ is a small positive cutoff
- document this behavior (it is mathematically correct but visually unhelpful)

### 3) Explicitly draw vertical jumps for $\lambda(T)$ and $\theta(T)$

Because $A(T)$ and $D(T)$ jump at event times, $\lambda$ and $\theta$ jump.

To render jumps:

At $T=T_{i+1}$, draw a vertical segment from the left-limit to the right-limit:

- left-limit: $A_i/T_{i+1}$ or $D_i/T_{i+1}$
- right-limit: $A_{i+1}/T_{i+1}$ or $D_{i+1}/T_{i+1}$

Only draw the jump for the metric whose numerator changes at that event type.

### 4) Preserve event markers

Overlay the event-indexed points as scatter markers:

- $(T_i, \lambda_i)$, $(T_i, \theta_i)$, $(T_i, L_i)$, $(T_i, w_i)$, $(T_i, w'_i)$

This makes it easy to validate that the analytic curve passes through the computed event values (or differs only because of left/right convention).

### 5) Units and scaling

Allow a time-unit option:

- internal compute in seconds
- plot in seconds/hours/days via a scale factor

If the stored $\lambda_i$ and $\theta_i$ are “per second” but you plot “per day”, scale accordingly:
- multiply by $86400$ for per-day rates when using seconds internally

### 6) Adaptive sampling (optional but recommended)

Fixed $k$ can be wasteful for short intervals and insufficient for long ones.

Provide an adaptive sampler such as:

- choose a density $d$ in points per unit time (e.g., 20 points/day)
- compute $k_i = clamp(k_{\min}, \lceil d\Delta T_i\rceil, k_{\max})$

Defaults:
- $k_{\min}=20$
- $k_{\max}=200$

### 7) API / Structure (suggested)

Implement a small “analytic renderer” module with pure functions.

Example function signatures (Python):

- `build_time_axis(events) -> np.ndarray[T_i]`
- `piecewise_curve(events, metric: str, k: int|adaptive) -> (x: np.ndarray, y: np.ndarray, jumps: list[tuple])`
- `plot_metric(ax, x, y, jumps, markers_x, markers_y, ...)`

Metric cases:

- `metric="lambda"` uses $A_i/T$
- `metric="theta"` uses $D_i/T$
- `metric="L"` uses $(H_i + N_i (T-T_i))/T$
- `metric="w"` uses $(H_i + N_i (T-T_i))/A_i$ if $A_i>0$
- `metric="wprime"` uses $(H_i + N_i (T-T_i))/D_i$ if $D_i>0$

## Proposed Tests

Use unit tests that verify both **math correctness** and **rendering structure**.

### A. Deterministic micro-case test (hand-checkable)

Construct a tiny event stream with known intervals:

Example:
- start at $T_0=1$ (avoid $0$)
- events at $T=\{1,2,4\}$
- event types: $A, A, D$
- right-continuous counts: $A=\{1,2,2\}$, $D=\{0,0,1\}$
- choose $N=\{1,2,1\}$ and compute $H$ consistently:
  - $H_0$ given
  - $H_{i+1}=H_i + N_i (T_{i+1}-T_i)$

Assertions:
- For several sample times inside each interval, analytic formulas match expected values.
- $L(T) = H(T)/T$ holds.
- Where defined, $w(T)=H(T)/A(T)$ and $w'(T)=H(T)/D(T)$.

### B. Invariant consistency test (property-based)

For random valid event streams (monotone times, integer counts, nonnegative $N$):

Sample random times inside random intervals and assert:

- $L(T) \approx \lambda(T) w(T)$ (when $A>0$)
- $L(T) \approx \theta(T) w'(T)$ (when $D>0$)

Use relative tolerance, e.g. `rtol=1e-9` with float64.

### C. Event-point interpolation test

For each metric curve:

- Evaluate analytic curve at $T_i$ using the chosen left/right convention.
- Assert it matches the event-indexed metric value in the dataset within tolerance.

If the dataset is left-continuous but renderer is right-continuous, this test should be structured to compare against the appropriate side (document the convention and test it explicitly).

### D. Jump rendering test (structure)

For $\lambda$ and $\theta$:

- Identify indices where numerator changes.
- Assert `jumps` list has exactly those indices.
- For each jump at $T_j$, assert the jump endpoints equal:
  - left = $A_{\text{before}}/T_j$
  - right = $A_{\text{after}}/T_j$
(or the $D$ equivalent)

### E. Complexity / performance sanity test

For a moderate dataset (e.g., 50k events):

- Ensure `piecewise_curve(..., adaptive=True)` completes under a reasonable time bound.
- Ensure output size is bounded by $O(\sum k_i)$ and `k_max` is honored.

### F. Edge-case tests

- $A=0$ early intervals: $w(T)$ should be NaN or absent until $A>0$.
- $D=0$ early intervals: same for $w'(T)$.
- Very short intervals: $k_i$ respects $k_{\min}$.
- Very long intervals: $k_i$ respects $k_{\max}$.
- $T$ near 0: first point dropped or clipped at $T_{\min}$.

## Acceptance Criteria

- Charts for $L(T)$, $\lambda(T)$, $\theta(T)$, $w(T)$, $w'(T)$ render as analytically correct piecewise curves.
- $\lambda(T)$ and $\theta(T)$ display visible vertical jumps at appropriate event times.
- Event-indexed points are still visible and align with the analytic curves (up to convention).
- Tests pass and confirm invariants and jump structure.
- Sampling density is configurable and defaults produce smooth plots without excessive compute.
