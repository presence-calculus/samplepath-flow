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

# Scope

This document is a catalog of generated charts, ordered by the canonical sample-path
construction arc from the presentation.

For theory and formal definitions, see [Sample Path Theory]($document-root/articles/theory).
For CLI options and output contracts, see [Command Line Reference]($document-root/articles/cli).

# Navigation

The catalog is organized so you can navigate in two clicks:

1. Use the TOC to jump to a sequence table.
2. Click a chart link in the table to jump to that chart's detail section.

Each detail section shows:

- `with-events` chart (visible by default), and
- `no-events` chart inside a collapsed block (hidden by default in HTML).

# Canonical Sequence Tables

## Core and Invariant Sequence (1-14)

| Step | Chart | Short Name | Formula |
| --- | --- | --- | --- |
| 1 | [Point Process](#chart-01-point-process) | Event Stream | Input event stream |
| 2 | [A(T)](#chart-02-arrivals-a) | Cumulative Arrivals | $A(T)=\sum \text{arrivals in }[0,T]$ |
| 3 | [D(T)](#chart-03-departures-d) | Cumulative Departures | $D(T)=\sum \text{departures in }[0,T]$ |
| 4 | [CFD](#chart-04-cfd) | Cumulative Flow Diagram | $N(t)=A(T)-D(T)$ |
| 5 | [N(t)](#chart-05-sample-path-n) | Process State | $N(t)=A(T)-D(T)$ |
| 6 | [H(T)](#chart-06-presence-mass-h) | Presence Mass | $H(T)=\int_0^T N(t)\,dt$ |
| 7 | [L(T)](#chart-07-time-average-l) | Time-Average Presence | $L(T)=H(T)/T$ |
| 8 | [$\Lambda(T)$](#chart-08-arrival-rate-lambda) | Arrival Rate | $\Lambda(T)=A(T)/T$ |
| 9a | [w(T)](#chart-09-residence-w) | Residence per Arrival | $w(T)=H(T)/A(T)$ |
| 9b | [$L(T)=\Lambda(T)\cdot w(T)$ Invariant](#chart-10-arrival-invariant) | Arrival Invariant | $L(T)=\Lambda(T)\cdot w(T)$ |
| 10 | [Arrival Stack](#chart-11-arrival-stack) | Arrival Dashboard | $L(T)=\Lambda(T)\cdot w(T)$ |
| 11 | [$\Theta(T)$](#chart-12-departure-rate-theta) | Departure Rate | $\Theta(T)=D(T)/T$ |
| 12 | [w'(T)](#chart-13-residence-w-prime) | Residence per Departure | $w'(T)=H(T)/D(T)$ |
| 13 | [Departure Focused Invariant](#chart-14-departure-invariant) | Departure Invariant | $L(T)=\Theta(T)\cdot w'(T)$ |
| 14 | [Departure Focused Stack](#chart-15-departure-stack) | Departure Dashboard | $L(T)=\Theta(T)\cdot w'(T)$ |
| 17 | [Residence Time Scatter Plot](#chart-18-residence-scatter) | Residence Scatter | $w(T)=H(T)/A(T)$ with residence samples |
| 18 | [Sojourn Time Scatter Plot](#chart-19-sojourn-scatter) | Sojourn Scatter | $W^*(T)=\operatorname{AVG}(d_i-a_i)$ |

## Convergence Sequence (1-3)

| Step | Chart | Short Name | Formula |
| --- | --- | --- | --- |
| 1 | [$\Lambda(T)$-$\Theta(T)$ Rate Convergence](#chart-16-arrival-departure-rate-convergence) | Rate Convergence | $\Lambda(T)=A(T)/T$ vs $\Theta(T)=D(T)/T$ |
| 2 | [Process Time Convergence](#chart-17-process-time-convergence) | Time Convergence | $w(T)=H(T)/A(T)$ vs $W^*(t)$ |
| 3 | [Top-Level Convergence $L(T)$ vs $\lambda^*(t)\cdot W^*(t)$](#chart-20-sample-path-convergence) | Top-Level Convergence | $L(T)$ vs $\lambda^*(t)\cdot W^*(t)$ |

# Core and Invariant Details

### 1. Point Process - Event Stream {#chart-01-point-process}

**Derivation:** N/A (input event stream).

Builds from the event log itself: this is the base marked-point-process view that all
subsequent cumulative quantities depend on.

**Output file:** `core/panels/arrival_departure_indicator_process.png`

`with-events`

![Point Process (with-events)]($document-root/articles/chart-reference/chart_reference_small/with-events/core/panels/arrival_departure_indicator_process.png)

<details>
<summary>No-events version</summary>

![Point Process (no-events)]($document-root/articles/chart-reference/chart_reference_small/no-events/core/panels/arrival_departure_indicator_process.png)

</details>

### 2. A(T) - Cumulative Arrivals {#chart-02-arrivals-a}

**Derivation:** $A(T)=\sum \text{arrivals in }[0,T]$.

Builds on Step 1 by accumulating arrival marks over time into cumulative arrivals.

**Output file:** `core/panels/cumulative_arrivals_A.png`

`with-events`

![A(T) cumulative arrivals (with-events)]($document-root/articles/chart-reference/chart_reference_small/with-events/core/panels/cumulative_arrivals_A.png)

<details>
<summary>No-events version</summary>

![A(T) cumulative arrivals (no-events)]($document-root/articles/chart-reference/chart_reference_small/no-events/core/panels/cumulative_arrivals_A.png)

</details>

### 3. D(T) - Cumulative Departures {#chart-03-departures-d}

**Derivation:** $D(T)=\sum \text{departures in }[0,T]$.

Builds on Step 2 by accumulating departure marks, giving the second cumulative boundary
needed for flow geometry.

**Output file:** `core/panels/cumulative_departures_D.png`

`with-events`

![D(T) cumulative departures (with-events)]($document-root/articles/chart-reference/chart_reference_small/with-events/core/panels/cumulative_departures_D.png)

<details>
<summary>No-events version</summary>

![D(T) cumulative departures (no-events)]($document-root/articles/chart-reference/chart_reference_small/no-events/core/panels/cumulative_departures_D.png)

</details>

### 4. CFD - Cumulative Flow Diagram {#chart-04-cfd}

**Derivation:** $N(t)=A(T)-D(T)$.

Builds on Steps 2 and 3 by placing $A(T)$ and $D(T)$ together; the vertical gap becomes
instantaneous state and the enclosed area motivates presence mass.

**Output file:** `core/panels/cumulative_flow_diagram.png`

`with-events`

![CFD (with-events)]($document-root/articles/chart-reference/chart_reference_small/with-events/core/panels/cumulative_flow_diagram.png)

<details>
<summary>No-events version</summary>

![CFD (no-events)]($document-root/articles/chart-reference/chart_reference_small/no-events/core/panels/cumulative_flow_diagram.png)

</details>

### 5. N(t) - Process State {#chart-05-sample-path-n}

**Derivation:** $N(t)=A(T)-D(T)$.

Builds on the CFD gap: $N(t)$ is the pointwise difference between cumulative arrivals and
cumulative departures.

**Output file:** `core/panels/sample_path_N.png`

`with-events`

![N(t) sample path (with-events)]($document-root/articles/chart-reference/chart_reference_small/with-events/core/panels/sample_path_N.png)

<details>
<summary>No-events version</summary>

![N(t) sample path (no-events)]($document-root/articles/chart-reference/chart_reference_small/no-events/core/panels/sample_path_N.png)

</details>

### 6. H(T) - Presence Mass {#chart-06-presence-mass-h}

**Derivation:** $H(T)=\int_0^T N(t)\,dt$.

Builds on $N(t)$ by integrating it over elapsed time, producing cumulative presence mass.

**Output file:** `core/panels/cumulative_presence_mass_H.png`

`with-events`

![H(T) presence mass (with-events)]($document-root/articles/chart-reference/chart_reference_small/with-events/core/panels/cumulative_presence_mass_H.png)

<details>
<summary>No-events version</summary>

![H(T) presence mass (no-events)]($document-root/articles/chart-reference/chart_reference_small/no-events/core/panels/cumulative_presence_mass_H.png)

</details>

### 7. L(T) - Time-Average Presence {#chart-07-time-average-l}

**Derivation:** $L(T)=H(T)/T$.

Builds on $H(T)$ by normalizing by elapsed time, yielding time-average presence.

**Output file:** `core/panels/time_average_N_L.png`

`with-events`

![L(T) time average (with-events)]($document-root/articles/chart-reference/chart_reference_small/with-events/core/panels/time_average_N_L.png)

<details>
<summary>No-events version</summary>

![L(T) time average (no-events)]($document-root/articles/chart-reference/chart_reference_small/no-events/core/panels/time_average_N_L.png)

</details>

### 8. $\Lambda(T)$ - Arrival Rate {#chart-08-arrival-rate-lambda}

**Derivation:** $\Lambda(T)=A(T)/T$.

Builds on cumulative arrivals by converting counts to elapsed-time-normalized arrival
rate.

**Output file:** `core/panels/cumulative_arrival_rate_Lambda.png`

`with-events`

![Lambda(T) arrival rate (with-events)]($document-root/articles/chart-reference/chart_reference_small/with-events/core/panels/cumulative_arrival_rate_Lambda.png)

<details>
<summary>No-events version</summary>

![Lambda(T) arrival rate (no-events)]($document-root/articles/chart-reference/chart_reference_small/no-events/core/panels/cumulative_arrival_rate_Lambda.png)

</details>

### 9a. w(T) - Residence per Arrival {#chart-09-residence-w}

**Derivation:** $w(T)=H(T)/A(T)$.

Builds on $H(T)$ and $A(T)$ by expressing accumulated presence per arrival as average
residence per arrival.

**Output file:** `core/panels/average_residence_time_w.png`

`with-events`

![w(T) residence per arrival (with-events)]($document-root/articles/chart-reference/chart_reference_small/with-events/core/panels/average_residence_time_w.png)

<details>
<summary>No-events version</summary>

![w(T) residence per arrival (no-events)]($document-root/articles/chart-reference/chart_reference_small/no-events/core/panels/average_residence_time_w.png)

</details>

### 9b. $L(T)=\Lambda(T)\cdot w(T)$ Invariant - Arrival Invariant {#chart-10-arrival-invariant}

**Derivation:** $L(T)=\Lambda(T)\cdot w(T)$.

Builds on Steps 7-9a by verifying the finite-window arrival-side invariant at each
observation point.

**Output file:** `core/panels/littles_law_invariant.png`

`with-events`

![Arrival invariant (with-events)]($document-root/articles/chart-reference/chart_reference_small/with-events/core/panels/littles_law_invariant.png)

<details>
<summary>No-events version</summary>

![Arrival invariant (no-events)]($document-root/articles/chart-reference/chart_reference_small/no-events/core/panels/littles_law_invariant.png)

</details>

### 10. Arrival Stack - Arrival Dashboard {#chart-11-arrival-stack}

**Derivation:** $L(T)=\Lambda(T)\cdot w(T)$.

Builds on Steps 5, 7, 8, and 9a by presenting the arrival-side state, average, rate, and
residence components on one aligned dashboard.

**Output file:** `sample_path_flow_metrics.png`

`with-events`

![Arrival stack (with-events)]($document-root/articles/chart-reference/chart_reference_small/with-events/sample_path_flow_metrics.png)

<details>
<summary>No-events version</summary>

![Arrival stack (no-events)]($document-root/articles/chart-reference/chart_reference_small/no-events/sample_path_flow_metrics.png)

</details>

### 11. $\Theta(T)$ - Departure Rate {#chart-12-departure-rate-theta}

**Derivation:** $\Theta(T)=D(T)/T$.

Builds from the departure count path by converting cumulative departures to
elapsed-time-normalized departure rate.

**Output file:** `core/panels/cumulative_departure_rate_Theta.png`

`with-events`

![Theta(T) departure rate (with-events)]($document-root/articles/chart-reference/chart_reference_small/with-events/core/panels/cumulative_departure_rate_Theta.png)

<details>
<summary>No-events version</summary>

![Theta(T) departure rate (no-events)]($document-root/articles/chart-reference/chart_reference_small/no-events/core/panels/cumulative_departure_rate_Theta.png)

</details>

### 12. w'(T) - Residence per Departure {#chart-13-residence-w-prime}

**Derivation:** $w'(T)=H(T)/D(T)$.

Builds on $H(T)$ and $D(T)$ by expressing accumulated presence per departure.

**Output file:** `core/panels/average_residence_time_w_prime.png`

`with-events`

![w'(T) residence per departure (with-events)]($document-root/articles/chart-reference/chart_reference_small/with-events/core/panels/average_residence_time_w_prime.png)

<details>
<summary>No-events version</summary>

![w'(T) residence per departure (no-events)]($document-root/articles/chart-reference/chart_reference_small/no-events/core/panels/average_residence_time_w_prime.png)

</details>

### 13. $L(T)=\Theta(T)\cdot w'(T)$ Invariant - Departure Invariant {#chart-14-departure-invariant}

**Derivation:** $L(T)=\Theta(T)\cdot w'(T)$.

Builds on Steps 7, 11, and 12 by verifying $L(T)=\Theta(T)\cdot w'(T)$ pointwise.

**Output file:** `core/panels/departure_littles_law_invariant.png`

`with-events`

![Departure invariant (with-events)]($document-root/articles/chart-reference/chart_reference_small/with-events/core/panels/departure_littles_law_invariant.png)

<details>
<summary>No-events version</summary>

![Departure invariant (no-events)]($document-root/articles/chart-reference/chart_reference_small/no-events/core/panels/departure_littles_law_invariant.png)

</details>

### 14. Departure Focused Stack - Departure Dashboard {#chart-15-departure-stack}

**Derivation:** $L(T)=\Theta(T)\cdot w'(T)$.

Builds on Steps 5, 7, 11, and 12 by presenting the departure-side dashboard in aligned
panels.

**Output file:** `core/departure_flow_metrics.png`

`with-events`

![Departure stack (with-events)]($document-root/articles/chart-reference/chart_reference_small/with-events/core/departure_flow_metrics.png)

<details>
<summary>No-events version</summary>

![Departure stack (no-events)]($document-root/articles/chart-reference/chart_reference_small/no-events/core/departure_flow_metrics.png)

</details>

### 17. Residence Time Scatter Plot - Residence Scatter {#chart-18-residence-scatter}

**Derivation:** $w(T)=H(T)/A(T)$ with residence samples.

Builds on Step 9a by exposing the underlying residence-time distribution around the
average trajectory.

**Output file:** `core/panels/residence_time_scatter.png`

`with-events`

![Residence scatter (with-events)]($document-root/articles/chart-reference/chart_reference_small/with-events/core/panels/residence_time_scatter.png)

<details>
<summary>No-events version</summary>

![Residence scatter (no-events)]($document-root/articles/chart-reference/chart_reference_small/no-events/core/panels/residence_time_scatter.png)

</details>

### 18. Sojourn Time Scatter Plot - Sojourn Scatter {#chart-19-sojourn-scatter}

**Derivation:** $W^*(T)=\operatorname{AVG}(d_i-a_i)$.

Builds on Step 17 by contrasting completed-item sojourn dispersion with residence-time
behavior.

**Output file:** `core/panels/sojourn_time_scatter.png`

`with-events`

![Sojourn scatter (with-events)]($document-root/articles/chart-reference/chart_reference_small/with-events/core/panels/sojourn_time_scatter.png)

<details>
<summary>No-events version</summary>

![Sojourn scatter (no-events)]($document-root/articles/chart-reference/chart_reference_small/no-events/core/panels/sojourn_time_scatter.png)

</details>

# Convergence Details

### 15. $\Lambda(T)$-$\Theta(T)$ Rate Convergence - Rate Convergence {#chart-16-arrival-departure-rate-convergence}

**Derivation:** $\Lambda(T)=A(T)/T$ vs $\Theta(T)=D(T)/T$.

Builds from Steps 8 and 11 by directly comparing cumulative arrival and departure rate
trajectories.

**Output file:** `convergence/panels/arrival_departure_rate_convergence.png`

`no-events`

![Arrival-departure rate convergence (no-events)]($document-root/articles/chart-reference/chart_reference_small/no-events/convergence/panels/arrival_departure_rate_convergence.png)

<details>
<summary>With-events version</summary>

![Arrival-departure rate convergence (with-events)]($document-root/articles/chart-reference/chart_reference_small/with-events/convergence/panels/arrival_departure_rate_convergence.png)

</details>

### 16. Process Time Convergence - Time Convergence {#chart-17-process-time-convergence}

**Derivation:** $w(T)=H(T)/A(T)$ vs $W^*(t)$.

Builds from Step 9a by comparing finite-window residence behavior to empirical
process-time behavior.

**Output file:** `convergence/panels/process_time_convergence.png`

`no-events`

![Process time convergence (no-events)]($document-root/articles/chart-reference/chart_reference_small/no-events/convergence/panels/process_time_convergence.png)

<details>
<summary>With-events version</summary>

![Process time convergence (with-events)]($document-root/articles/chart-reference/chart_reference_small/with-events/convergence/panels/process_time_convergence.png)

</details>

### 19. Top-Level Convergence $L(T)$ vs $\lambda^*(t)\cdot W^*(t)$ {#chart-20-sample-path-convergence}

**Derivation:** $L(T)$ vs $\lambda^*(t)\cdot W^*(t)$.

Builds on the full chain by giving a top-level convergence diagnostic for the
finite-window Little's Law relation over the observation horizon.

**Output file:** `sample_path_convergence.png`

`no-events`

![Sample path convergence (no-events)]($document-root/articles/chart-reference/chart_reference_small/no-events/sample_path_convergence.png)

<details>
<summary>With-events version</summary>

![Sample path convergence (with-events)]($document-root/articles/chart-reference/chart_reference_small/with-events/sample_path_convergence.png)

</details>
