# Finite-Window Flow Metrics & Convergence Diagnostics

This tool turns time-stamped **start** and **end** events into a full set of **Little’s Law**-based flow metrics and convergence diagnostics. It implements the *finite-window identity* (`L(T) = Λ(T)·w(T)`), dynamic baselines, **coherence** checks, and an **end-effects** panel to explain when/why the finite-window averages closely approximate the per-item (“true”) averages.

It’s designed for real (non-stationary) systems where work mixes and speeds drift over time. You can chart **completed**, **incomplete (aging)**, or **all** items, and optionally filter outliers (hours, percentile, Tukey IQR), and filter by an optional `class` tag.

---

## Quick Theory (15-second version)

- **Finite-window identity (always true):**  
  For any observation window of length `T` starting at the first sample time,  
  `A(T) = ∫ N(t) dt` (WIP-hours); `L(T) = A(T)/T`; `Λ(T) = arrivals(T)/T`; `w(T) = A(T)/arrivals(T)`.  
  Hence **`L(T) = Λ(T) · w(T)`** (tautology).

- **Dynamic baselines (moving “truth”):**
  - `W*(t)` = cumulative mean of completed item durations ≤ `t`
  - `λ*(t)` = cumulative mean arrival rate ≤ `t`

- **Coherence (practical convergence):**  
  Relative errors `e_W = |w(T)−W*(t)|/W*(t)` and `e_λ = |Λ(T)−λ*(t)|/λ*(t)`.  
  After horizon `H`, you’re **ε-coherent** if both errors ≤ ε for most times.

- **End-effects:** items cut by the window cause bias. If `T` ≫ typical duration, end-effects are negligible.

---

## What the Script Produces

Outputs go to `charts/<csv-stem>/`.

**Core stacks (timestamped and daily):**
1) `N(t)` — active processes; 2) `L(T)`; 3) `Λ(T)` (optional y-axis clipping); 4) `w(T)`;  
5) (optional) `A(T)` — cumulative area (WIP-hours).

**Convergence & coherence:**
- `…_convergence.png`: vs constant `W*`, `λ*` over full span
- `…_convergence_dynamic.png`: vs `W*(t)`, `λ*(t)` (time-varying)
- `…_convergence_dynamic_errors.png`: adds `e_W`, `e_λ`, ε band
- **`…_convergence_dynamic_errors_endeffects.png`**: adds end-effects panel:
  - `r_A(T) = E(T)/A(T)` (mass share from boundary-cut items)
  - `r_B(T) = B(T)/starts` (share of started-but-unfinished items)
  - `ρ(T) = T/W*(t)` (secondary axis; window vs typical duration)

**Scatter overlay (optional):**  
A five-stack with plain `w(T)` (own y-axis) and `w(T)`+scatter (combined axis). Scatter points are per-item duration at completion (or age for incomplete view).

---

## Input CSV

Header (case-insensitive):
```
id, start_ts, end_ts[, class]
```
- `end_ts` may be empty (NaT) for incomplete items.
- Optional `class` enables `--classes` filtering.

**Validation**
- `start_ts` must parse.  
- If `end_ts` exists, it must be ≥ `start_ts`.  
- Data sorted by `start_ts`.

---

## Install

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install pandas numpy matplotlib
```

---

## Run

```bash
python finite_window_metrics_charts.py [OPTIONS] your_data.csv
```

**Common options**
- `--completed` | `--incomplete`
- `--with-A` (adds A(T) five-stack)
- `--with-daily-breakdown` (daily ΔA and avg WIP)
- `--classes "story,bug"` (requires `class` column)
- `--scatter` (durations at completion, or ages for incomplete)

**Outlier filters (completed durations)**
- `--outlier-hours H`
- `--outlier-pctl P`  (drop above P-th percentile)
- `--outlier-iqr K`   (drop above Q3+K·IQR; add `--outlier-iqr-two-sided` for low fence too)

**Λ(T) readability**
- `--lambda-pctl P`, `--lambda-lower-pctl P`, `--lambda-warmup H`

**Coherence**
- `--epsilon EPS` (default 0.10)
- `--horizon-days D` (default 28)

---

## How It Computes

**Event sweep:** create ordered events: start `(+1, arrival=1)`, end `(−1, arrival=0)`. Between sample times, `N(t)` is constant. We integrate area pieces: `A += N·Δt`. Then
`L=A/T`, `Λ=arrivals/T`, `w=A/arrivals` (if arrivals>0).

**Empirical targets:**  
`W*` = mean duration of completed items over the whole file; `λ*` = total starts / span hours.  
**Dynamic:** `W*(t)`, `λ*(t)` up to each `t`.

**Errors:** `e_W`, `e_λ` as above; score coherence after horizon `H`.

**End-effects:**  
- `A_full(t)` = sum of **full** durations of items with `end_ts ≤ t`  
- `E(T)=A−A_full`, `r_A = E/A`  
- `r_B = (# started by t but unfinished)/starts`  
- `ρ = T/W*(t)`  

If `r_A`, `r_B` small and `ρ` large, finite-window averages are excellent approximations even without stationary limits.

---

## Examples

Completed + A(T) + scatter; clip Λ at 99th pctl, warmup 48h:
```bash
python finite_window_metrics_charts.py \
  --completed --with-A --scatter \
  --lambda-pctl 99 --lambda-warmup 48 \
  results_32.csv
```

Aging view with IQR filter; ε=5%, horizon=60d:
```bash
python finite_window_metrics_charts.py \
  --incomplete --scatter \
  --outlier-iqr 1.5 \
  --epsilon 0.05 --horizon-days 60 \
  backlog.csv
```

Filter by class and 95th percentile:
```bash
python finite_window_metrics_charts.py \
  --classes "story,bug" --outlier-pctl 95 data.csv
```

---

## Troubleshooting

- Empty charts after filtering → relax filters.  
- Early NaNs in `w(T)` → no arrivals yet.  
- Spiky Λ → use percentile clipping and warmup.  
- No “flat” convergence → normal for non-stationary systems; rely on **dynamic** and **coherence** views.

---

## License

MIT
