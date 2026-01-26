# Metric Derivations — Spec

## Overview
Add an optional UI mode to display metric derivations in chart titles. This is controlled by a new CLI flag and centralized derivation definitions in the metrics module to keep math notation consistent across all usages.

## Scope (Phase 1)
- Panels in the **core module** only (`samplepath/plots/core.py`).
- Titles only (no legends/captions changes).
- Unicode math in titles (e.g., Λ, ∫, ρ, subscripts where appropriate).

## CLI Flag
- `--show-derivations` (bool, default: `False`).
- When `True`, core panel titles append the derivation string.
- When `False`, titles remain as they are today.

## Centralized Derivation Definitions
Create a single class in `samplepath/metrics.py` that holds the canonical derivations.

### Responsibilities
- Store the **title suffix** text for each metric.
- Provide a stable lookup by metric key.
- Serve as a single source of truth for documentation and UI.

### Example API
```python
class MetricDerivations:
    # map metric keys -> derivation string
    DERIVATIONS: dict[str, str]

    @classmethod
    def get(cls, key: str) -> str | None:
        ...
```

## Derivations (Unicode)
These should live in `MetricDerivations` and be referenced everywhere they are used.

Conventions:
- Upper-case `T` denotes a metric over the interval `[0, T]`.
- Lower-case `t` denotes an instantaneous metric at time `t`.
- We treat interval arithmetic as normalized to `[0, T]` (so denominators use `T`, not `T − t₀`).

So `H(T) = ∫₀ᵀ N(t) dt` is cumulative, and `N(t) = A(t) − D(t)` is instantaneous.

### Core panel metrics
- **A(T)**: `∑ arrivals in [0, T]`
- **D(T)**: `∑ departures in [0, T]`
- **N(t)**: `A(t) − D(t)`
- **H(T)**: `H(T) = ∫₀ᵀ N(t) dt`
- **L(T)**: `L(T) = H(T) / T`
- **Λ(T)**: `Λ(T) = A(T) / T`
- **w(T)**: `w(T) = H(T) / A(T)`


## Title Formatting Rules
When derivations are enabled, the title format is:

```
<existing title> : <derivation>
```

Examples:
- Without derivation: `Sample Path — N(t)`
- With derivation: `Sample Path — N(t): N(t) = A(t) − D(t)`
- Without derivation: `Λ(T) — Cumulative Arrival Rate`
- With derivation: `Λ(T) — Cumulative Arrival Rate: Λ(T) = A(T) / T`

## Implementation Requirements
1. **Centralization:** all derivation strings must be read from `MetricDerivations`.
2. **Core panels only:** `render_N`, `render_L`, `render_Lambda`, `render_w`, `render_H` should accept a `show_derivation: bool` (or equivalent) and use the centralized strings.
   - Title strings without derivations must be passed in as parameters (rather than hardcoded), so callers can override titles later.
3. **CLI wire-up:** `--show-derivations` must reach the core plot driver and be passed to these renderers.
4. **No behavior changes when off:** title strings remain byte-for-byte identical to the provided base title when `show_derivations` is `False`.

## Tests
- Unit tests for `MetricDerivations.get` (or equivalent).
- Core plot tests verifying:
  - Titles unchanged when `show_derivations=False`.
  - Titles include derivations when `show_derivations=True`.
  - Derivations sourced from the centralized class (mock/patch to assert lookup).

## Non-Goals (for Phase 1)
- No derivations for convergence/misc/stability/advanced charts.
- No legend/caption text changes.
- No LaTeX rendering, only Unicode strings.
