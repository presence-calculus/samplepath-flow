---
title: <strong>Chart Reference</strong>
author: |
  <a href="https://github.com/presence-calculus/samplepath"><em>The Samplepath Analysis Toolkit</em></a>

document-root: ../..
header-image: $document-root/assets/sample_path_flow_metrics.png

# Configure TOC
toc: true
toc-title: Contents
toc-depth: 3
# Configure section numbers
numberSections: true
sectionsDepth: 2
# Configure figures
figPrefix: Figure
# Configure citations
citations: false
---

> > **Note: _This document is a work in progress. This note will be removed when all the charts
> > produced by the analysis are documented_**.

# What is Sample Path Analysis?

Sample path analysis is a _deterministic_ analysis that can be run on any observed flow
process history. It lets us reason about the dynamics and stability of flow processes over the long run, and in real-time.

We use sample path analysis to determine whether a process is stable, and if not what
the causes of instability are, and what needs to be done to move the process towards
operationally acceptable range of stability.

**Please see our technical note**:
[Sample path analysis is not a statistical technique.]($document-root/articles/not-statistics)

## The charts.

The results of the analysis are a collection of charts that are written to an output
directory.

For input `events.csv`, output is organized as follows:

```bash
<output-dir>/
└── events/
    └── <scenario>/                 # e.g., latest
        ├── input/                  # a snapshot  of the input csv data
        ├── core/                   # core flow metrics & tables
        ├── convergence/            # convergence vs divergence
        ├── stability/              # boundedness and existence of limits
        ├── advanced/               # optional deep-dive charts
```

This reference describes every chart produced by the `samplepath` CLI, grouped by chart
type.

# A Worked Example

The example charts in this reference and the underlying narrative are
discussed in our post [Little's Law in a Complex Adaptive System](https://www.polaris-flow-dispatch.com/i/172332418/sample-path-analysis-a-worked-example)

The example charts in each section below are drawn from the Polaris scenario
[completed-stories-outliers-removed](https://github.com/presence-calculus/samplepath/tree/main/examples/polaris/flow-of-work/complete-stories-outliers-removed).

Note: All calculations are done in *continuous time* and all charts report time
accumulations in hours.

# Sample Path Flow Metrics

Time series charts showing the core functions that govern the long run dynamics of a [flow
process](https://www.polaris-flow-dispatch.com/i/172332418/flow-processes).

> To keep things grounded
> think of the csv file that you are providing as input to the analysis as the flow process under analysis.
>
> The current version of the library only analyzes binary flow processes - where start and end dates represent arrivals/departures events, the effect we are measuring is the presence or absence of items in the process, and the sample path represents the **counting process**: the number of items present in the process at any moment in time (aka the WIP).
>
> As we will note along the way, **every concept** in this document generalizes even for the much more general forms of flow processes once you plug in a different sample path. In particular, the set of outputs produced by sample path analysis are **identical** even after generalization!
>
> This is why this set of charts is the **foundation** of flow process dynamics.

**Conventions**

- All charts are defined over a finite observation horizon of length $T_{\max}$.

- Lower-case $t$ is a parameter denoting *instantaneous* time.

  Functions of $t$ (such as $N(t)$) denote sample paths: the values of some measurable property of a process at a specific moment $t \in [0,\,T_{\max}]$.

- Upper-case $T$ is a *prefix parameter*: a scalar value that determines a _time interval_ $[0,\,T]$.

  Functions of $T$ (such as $L(T)$, $\Lambda(T)$, and $w(T)$ below) are *functionals*: functions that take a sample path as input and compute a scalar value from the behavior of that function over that interval $[0,T]$. Here $T$ can range over any value in the observation window $(0,\,T_{\max}]$.

  These functionals are purely deterministic, pathwise calculations (definite integrals in our case) and should not be confused with statistical aggregates.

## The Core Calculations for Little's Law

The input to the analysis is a sample path $N(t)$.

There are four core functionals:

- $H(T)$: the area under the sample path over $[0,T]$.
- $L(T)$: the time average of $N(t)$, defined as $L(T)=\frac{H(T)}{T}$.
- $\Lambda(T)$: the cumulative arrival rate, defined as $\Lambda(T)=\frac{A(T)}{T}$.
- $w(T)$: the average residence time, defined as $w(T)=\frac{H(T)}{A(T)}$.

From these definitions it follows that

$$
L(T) = \frac{H(T)}{T}
     = \frac{A(T)}{T}.\frac{H(T)}{A(T)}.
$$

Equivalently,

$$
L(T) = \Lambda(T).w(T).
$$

This is the finite version of Little’s Law.

It is a _deterministic identity_ that holds for every finite observation window $[0,T]$ with $0 < T < \infty$ and $A(T) > 0$.
This implies that the three component functions _always_ evolve in a way that preserves the identity.

The finite version of Little's Law is thus a _deterministic constraint_ that governs the global dynamics of _any_ flow process, _even ones that operate in complex adaptive systems_. It is the foundation for adapting Little's Law for use in complex systems.

> **Note**: As we discuss in the [Many Faces of Little's Law](https://www.polaris-flow-dispatch.com/p/the-many-faces-of-littles-law), while the $L=\lambda.W$ form of Little's Law is incredibly powerful in operations management, much more can be unlocked once we consider the general $H=\lambda.G$ form of Little's Law.
>
> The current version of the library does not provide the representations needed to _compute sample paths_ for this general form of the law. But the key thing to note is that the analysis of this general form and all the charts we see here depend only on the _shape of the sample path_.
>
> Thus what is missing are the sample path construction algorithms that lie upstream of this analysis and the algorithms to compute the necessary functionals for $H(T)$. Once we provide this. the charts we produce are exactly the same for both forms of the law. This is why this set of charts is a very powerful tool to reason about flow dynamics.
>
> So while it is easier to relate to these concepts expressed in terms of familiar ideas like WIP, Arrivals/Departure and Residence Time, all these concepts have natural generalizations when viewed from the lens of the general $H=\lambda.G$ form.
>
> This is the generalization we provide in [The Presence Calculus](https://docs.pcalc.org/articles/intro-to-presence-calculus/).

## Summary chart

This chart can be found at the top level under `<scenario>/`

| File                           | What it shows                                                                      | What it means                                                                                                |
| ------------------------------ | ---------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------ |
| `sample_path_flow_metrics.png` | Four-panel vertical stack: `N(t)`, `L(T)`, `Λ(T)`, `w(T)` over the same time axis. | One-glance view of the core finite-window Little’s Law metrics and how they co-evolve along the sample path. |

![Fig 1. Sample Path Flow Metrics](images/sample_path_flow_metrics.png)

In this chart the main thing to pay attention to are the _relationships_ between the
_changes_ in each of these component charts over time.

- At any point in time review whether $L(T)$ the time average of $N(t)$ is changing or
  flat.
- If it is changing then the finite version of Little's Law tells us that this must be
  because at least one of $\Lambda(T)$ or $w(T)$ changed.

It means that these are the _only_ possible explanations:

- $\Lambda(T)$ changed: more or fewer things are arriving.
- $w(T)$ changed: things are taking more or less time to finish.
- A combination of the two.

In the last case there are two possibilities.

Since $L(T)$ is the product of $\Lambda(T)$ and $w(T)$:

- If they move in the same direction they lead to proportionally large changes in L(T)
- If they move in opposite directions, the changes cancel each other and $L(T)$ tends to
  flatten.

The last situation happens often when there is an internal feedback loop at play in the
process.

All these different types of dynamics can be at play at different points in the
evolution of the process. This chart is the place where all those long run dynamics will
show most directly and clearly.

It helps us answer "how did the process evolve to produce the flow metrics we are seeing
at any given moment in time?" These causal relationships allows us to monitor changes in
L(T) and then immediately investigate the cause of that change.

See our post
[The Causal Arrow in Little's Law](https://www.polaris-flow-dispatch.com/i/171471652/the-causal-mechanism)
for more discussion on what this means.

______________________________________________________________________

## Component metrics

Each panel in the main chart is also written under

```
<scenario>/core/
```

| File                                      | What it shows                                                                                                         | What it means                                                                                                                  |
| ----------------------------------------- | --------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------ |
| `core/sample_path_N.png`                  | Step chart of `N(t)` (count of elements present in the boundary) vs time.                                             | Raw sample path of WIP/presence: queues, surges, and droughts show up directly.                                                |
| `core/time_average_N_L.png`               | Line chart of `L(T)` = time-average of `N(t)` over `[0, T]`.                                                          | Tracks how average WIP over the observation window converges (or doesn’t). This is the “L” in Little’s Law, measured pathwise. |
| `core/cumulative_arrival_rate_Lambda.png` | Line chart of `Λ(T)` (cumulative arrival rate `A(T)/(T−t₀)`), with optional percentile clipping and warmup exclusion. | Empirical arrival rate over time, with tools to ignore early transients and outliers.                                          |
| `core/average_residence_time_w.png`       | Line chart of `w(T)` (average residence time over the window, in hours).                                              | Shows how the time items spend in the boundary evolves; long/fat tails and slow drainage show up as increasing `w(T)`.         |
| `core/littles_law_invariant.png`          | Scatter of `L(T)` (x-axis) vs `Λ(T)·w(T)` (y-axis) with `y=x` reference line, equal aspect ratio.                     | Pure Little’s Law invariant check: all finite points should lie near `y=x` if the metric calculations are consistent.          |

Their detail descriptions follow.

### The sample path $N(t)$

`sample_path_N.png`

Since we measure in continuous time, this chart is a *step chart*.

- The line will go up with each arrival and go down with each departure.
- The line will stay steady if departures and arrivals balance each other at a given
  instant.
- In between arrivals and departures, the line will stay flat.

This is a real time chart that reveals current congestion, bursts, and idle periods.

> If you are familiar with Cumulative Flow Diagrams, $N(t)$ represents the distance between the cumulative arrival line and the cumulative departure line in the diagram.

In stochastic process language, $N(t)$ is a _process_ over the flow process

the primary _object under analysis_. Little's Law describes the

![Sample Path](images/core/sample_path_N.png)

### The area under the sample path

A key quantity in sample path analysis is the *area under the sample path*. This is
calculated as the definite integral of the sample path curve over [0,T]

$$
H(T) = \int_0^T N(t) dt
$$

Since the area is a product of the number of items present over time, the units of H(T)
are in item-time. In the language of the Presence Calculus, $H(T)$ is _cumulative presence_ function
over the sample path $N(t)$.

The functions that drive flow-process dynamics are the time and item averages
of H(T): L(T) and w(T). These continuous functions of time are the key quantities in the
_finite version of Little’s Law_.

### $L(T)$: Time average of $N(t)$

`time_average_N_L.png`

Time-average WIP:

`L(T)` is the time average of the area under the sample path. May also be viewed as the
rate at which the area H(T) grows.

Its units are in items.

```
L(T) = (1/T) H(T)
```

Reveals whether long run average WIP diverges or converges. This is the key top level
indicator of process stability. A flat line here indicates a stable process.

Please note once again that this is _not a statistical average_.

![Time Average of WIP](images/core/time_average_N_L.png)

### $\Lambda(T)$: Cumulative arrival rate

`cumulative_arrival_rate_Lambda.png`

`Λ(T)` is the arrival rate of items that have arrived up to T (may include items that
started before the window):

```
Λ(T) = A(T) / T
```

It's units are items/time

If WIP was zero at the beginning of the observation window, then this is the same as the
arrival rate, otherwise this over-counts the arrival rate at the start, but as we
observe the process for longer periods, those initial end-effects get averaged out.

![Cumulative Arrival Rate](images/core/time_average_N_L.png)

### $w(T)$: Average residence time

`average_residence_time_w.png`

Average time items are observed as spending in the observation window up to time T
(clips the time that is spent outside the window and thus not observed).

```
w(T) =  H(T)/A(T)
```

Tracks how “observed time in system” evolves over the sample path. Its units are time.

Please see our posts
[What is Residence Time](https://www.polaris-flow-dispatch.com/p/what-is-residence-time)
and
[How long does it take](https://www.polaris-flow-dispatch.com/p/how-long-does-it-take)
for an explanation of what this metric means.

_Understanding the difference and relationship between residence time and familiar
metrics like Lead Time, Cycle Time and Work Item Age is crucial for understanding why
sample path analysis works and these posts explain this._

![Average Residence Time](images/core/average_residence_time_w.png)

## The finite version of Little's Law

`littles_law_invariant.png`

This plot visualizes the
[finite version of Little's Law](https://www.polaris-flow-dispatch.com/i/172332418/the-finite-version-of-littles-law)
at work.

It states that for all T, `L(T)=Λ(T)·w(T)`.

We verify this by showing that when we plot `L(T)` vs `Λ(T)·w(T)` on a scatter plot, all
the points will lie on the with `y = x`.

![Little's Law Invariant](images/core/littles_law_invariant.png)

Notice how points cluster around certain values of L(T). These are significant operating
modes for the process as it moves towards stable states.

______________________________________________________________________

<!--
# Convergence - Equilibrium & coherence

| File                          | What it shows                                                                                                                   | What it means                                                                                                                                         |
| ----------------------------- | ------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------- |
| `sample_path_convergence.png` | Scatter of `L(T)` (x-axis) vs `λ*(t)·W*(t)` (y-axis) with `y=x` and an ε-band; annotated with a coherence score over a horizon. | Direct visual and quantitative test of whether the finite-window sample path obeys Little’s Law asymptotically (sample-path convergence / coherence). |

![Sample Path Convergence](images/sample_path_convergence.png)

| File                                                         | What it shows                                                                                                                                                      | What it means                                                                                                                               |
| ------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------- |
| `convergence/arrival_departure_equilibrium.png`              | Two-row stack: (1) cumulative arrivals `A(t)` vs cumulative departures `D(t)`; (2) `Λ(T)` vs throughput rate `θ(T)=D(T)/(T−t₀)` with masking after last departure. | Tests arrival/departure equilibrium: whether `A(t)` and `D(t)` grow together and arrival/throughput rates converge.                         |
| `convergence/panels/arrival_rate_convergence.png`            | Single panel: `Λ(T)` and `λ*(t)` (empirical arrival rate) over time, with optional warmup and percentile-based y-limits.                                           | Compares window-averaged arrival rate to the element-wise empirical rate; checks consistency of the two ways of measuring “arrival rate”.   |
| `convergence/panels/residence_time_convergence.png`          | Single panel: `w(T)` vs `W*(t)` (empirical mean sojourn time of completed items) over time.                                                                        | Coherence between residence-time and sojourn-time views: if the process is coherent, these two series should converge together.             |
| `convergence/residence_sojourn_coherence.png`                | Two-row stack: (1) `w(T)` vs `W*(t)` overlay; (2) scatter of individual sojourn times against time, with `w(T)` as a reference line.                               | Ties the averaged quantities back to individual element sojourn times; helps see whether outliers or subpopulations are driving divergence. |
| `convergence/panels/residence_time_sojourn_time_scatter.png` | Line of `w(T)` over time with overlaid scatter of element *age* (if `--incomplete`) or sojourn time (if `--completed` / default).                                  | Visualizes how individual element ages/sojourn times relate to the evolving average residence time, and whether a stable band emerges.      |

## `convergence/`

```
<scenario>/convergence/
<scenario>/convergence/panels/
```

______________________________________________________________________

## `convergence/arrival_departure_equilibrium.png`

Cumulative arrivals vs departures, plus arrival/throughput rate comparison.\
Tests for equilibrium.

## `convergence/panels/arrival_rate_convergence.png`

`Λ(T)` (window-average) and `λ*(t)` (empirical).\
Checks rate consistency.

## `convergence/panels/residence_time_convergence.png`

`w(T)` vs `W*(t)`.\
Coherence between window-average residence and empirical sojourn time.

## `convergence/residence_sojourn_coherence.png`

Two-row comparison of averages + individual sojourn scatter.

## `convergence/panels/residence_time_sojourn_time_scatter.png`

`w(T)` overlaid with individual ages or sojourns.

## `sample_path_convergence.png`

Scatter of `L(T)` vs `λ*(t)·W*(t)` with tolerance band.\
Highest-level convergence and coherence view.

Controlled by **Convergence Options**:\
`--convergence`, `--coherence-eps`, `--completed`, `--incomplete`, `--warmup`.

______________________________________________________________________

# Stability Charts

## `stability/` and `stability/panels/` — Rate stability

| File                                         | What it shows                                                                                                          | What it means                                                                                                                                                     |
| -------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `stability/panels/wip_growth_rate.png`       | Two-row stack: (1) `N(t)` step chart; (2) WIP growth rate `N(T)/T` with reference lines at 0 and 1.                    | Tests whether WIP grows linearly (instability) or sublinearly/flat (bounded or stabilizing), using a rate perspective.                                            |
| `stability/panels/total_age_growth_rate.png` | Two-row stack: (1) total active age `R(t)` in hours; (2) age growth rate `R(T)/T` with reference lines at 0 and 1.     | Looks at growth of total age of WIP: sustained growth points to accumulating, aging work and potential instability.                                               |
| `stability/rate_stability.png`               | Four-row stack: (1) `N(T)/T`; (2) `R(T)/T`; (3) `λ*(T)`; (4) `w(T)` vs `W*(t)`. Captioned “Equilibrium and Coherence”. | Integrated stability view: WIP and age growth rates, empirical arrival rate, and residence/sojourn coherence all on one canvas to assess long-run rate stability. |

Written under:

```
<scenario>/stability/
<scenario>/stability/panels/
```

Controlled by **Stability Options**: `--stability`.

## `stability/panels/wip_growth_rate.png`

`N(t)` and its growth rate `N(T)/T`.\
Detects sublinear vs linear growth.

## `stability/panels/total_age_growth_rate.png`

Total age `R(t)` and its growth rate.\
Identifies aging accumulation.

## `stability/rate_stability.png`

Four-panel stability synthesis:

- `N(T)/T`
- `R(T)/T`
- `λ*(T)`
- `w(T)` vs `W*(t)`

______________________________________________________________________

______________________________________________________________________

# Advanced Charts

## `advanced/` — Error terms, end-effects, and manifold view

| File                                                        | What it shows                                                                                                                                            | What it means                                                                                                                                                                         |
| ----------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `advanced/residence_convergence_errors.png`                 | Three-row stack: (1) `w(T)` vs `W*(t)` (dynamic); (2) `Λ(T)` vs `λ*(t)` (dynamic); (3) error magnitudes `e_W(T)` and `e_Λ(T)` with optional ε threshold. | Tracks *how* coherence is approached or violated, with explicit error terms for residence time and arrival rate over time.                                                            |
| `advanced/residence_time_convergence_errors_endeffects.png` | Four-row stack: same as above plus an end-effects panel with `r_A(T)` (mass share), `r_B(T)` (boundary share), and `ρ(T)=T/W*(t)`.                       | Decomposes residual errors into end-effect contributions and time-scaling, making it clear when divergence is driven by boundary conditions vs ongoing dynamics.                      |
| `advanced/invariant_manifold3D_log.png`                     | 3D log–log–log manifold plot: `x = log Λ(T)`, `y = log w(T)`, `z = log L(T)`, plotting the sample-path trajectory on the plane `z = x + y`.              | Geometric view of Little’s Law as an invariant plane; lets you see whether the observed trajectory sticks to the manifold and how it moves across regimes in (rate, time, WIP) space. |

Written under:

```
<scenario>/advanced/
```

Controlled by **Advanced Options**: `--advanced`.

## `advanced/residence_convergence_errors.png`

`w(T)` vs `W*(t)`;\
`Λ(T)` vs `λ*(t)`;\
error magnitudes.

## `advanced/residence_time_convergence_errors_endeffects.png`

Adds end-effects: `r_A(T)`, `r_B(T)`, `ρ(T)`.

## `advanced/invariant_manifold3D_log.png`

3D log–log–log manifold:\
`(log Λ(T), log w(T), log L(T))` on the plane `z = x + y`.

______________________________________________________________________
-->
