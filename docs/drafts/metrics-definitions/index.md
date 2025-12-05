# Finite-Window Flow Metrics Calculation

We consider a single observation window [0, T] and compute long-run flow metrics over
this finite horizon.

Let **A(t)** be the cumulative number of arrivals up to time *t*\
and **D(t)** be the cumulative number of departures up to time *t* for t ∈ [0, T]

From this we derive:

- **Sample Path** The *sample path* is the instantaneous number of items present in the
  system at time *t*: **N(t) = A(t) − D(t), for t ∈ [0, T]**

- **Cumulative Presence H(T)**\
  The total “item-time” accumulated by all active items during [0, T]. This is the area
  under the sample path over [0,T].\
  **H(T) = ∫₀ᵀ N(t) dt**

- **`L(T)` — Average Work-in-Process (WIP)**\
  The time-average of items present in the system over [0, T].\
  **L(T) = H(T) ⁄ T = (1 ⁄ T) ∫₀ᵀ N(t) dt**

- **`Λ(T)` — Cumulative Arrival Rate**\
  Let **A(T)** be the cumulative number of arrivals by time *T*. Then: **Λ(T) = A(T) ⁄
  T**

- **`w(T)` — Average Residence Time**\
  The average time items that *arrived over [0, T]* spend in the system. **w(T) = H(T) ⁄
  A(T)**

______________________________________________________________________

## Finite version of Little's Law

Together, these quantities satisfy the finite-window form of Little’s Law:

**L(T) = Λ(T) · w(T) for all t ∈ [0, T]**
