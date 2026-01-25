# ADR-003 Review: Agreed Changes and Recommendations

**Date:** 2026-01-22
**Reviewers:** Krishna Kumar, Claude

This document summarizes the review of ADR-003 and records agreed changes before implementation.

---

## Agreed Changes to ADR-003

### 1. Drop `_view` suffix

**Original:** `render_N_view`, `render_L_view`, `render_Lambda_view`, `render_w_view`

**Agreed:** `render_N`, `render_L`, `render_Lambda`, `render_w`, `render_A`

Rationale: `_view` is UI terminology that doesn't fit. Shorter names are clearer.

### 2. Keep explicit event parameters

**Original:**
```python
render_N_view(ax, times, N_vals, *, events=None, with_marks=False, ...)
```

**Agreed:**
```python
render_N(ax, times, N_vals, *, arrival_times=None, departure_times=None, with_event_marks=False, ...)
```

Rationale: Explicit parameters are self-documenting. An opaque `events` parameter hides structure and requires additional type definition.

### 3. Define ClipOptions dataclass

If bundling clip parameters, make it explicit:

```python
@dataclass
class ClipOptions:
    pctl_upper: Optional[float] = None
    pctl_lower: Optional[float] = None
    warmup_hours: float = 0.0
```

Rationale: `clip_opts` as an untyped parameter is underspecified. A dataclass documents the structure.

### 4. Include A(T) chart

**Original:** ADR-003 listed N, L, Λ, w but omitted A(T).

**Agreed:** Add `render_A` for cumulative area ∫N(t)dt.

Full chart list: `render_N`, `render_L`, `render_Lambda`, `render_w`, `render_A`

### 5. Keep `plot_*` naming for layouts

Standalone layouts use `plot_N`, `plot_L`, `plot_Lambda`, `plot_w`, `plot_A`.

Composite layout: `plot_core_stack`.

### 6. Add `render_overlays` helper

Extract the duplicated overlay rendering code (helpers.py lines 182-203 and 236-257) into a shared helper:

```python
def render_overlays(ax: Axes, overlays: List[ScatterOverlay]) -> None:
    """Render scatter overlays with optional drop lines."""
    for i, overlay in enumerate(overlays):
        if not overlay.x:
            continue
        if overlay.drop_lines:
            ax.vlines(overlay.x, 0, overlay.y, colors=overlay.color,
                      linewidths=0.5, alpha=0.5, zorder=4 + i)
        ax.scatter(overlay.x, overlay.y, color=overlay.color,
                   s=2, zorder=5 + i, label=overlay.label)
```

This eliminates duplication between `render_step` and `render_line`. Lives in `primitives.py`.

### 7. Separate modules per layer

```
samplepath/plots/
  primitives.py    # render_step, render_line, render_lambda
  charts.py        # render_N, render_L, render_Lambda, render_w, render_A
  layouts.py       # plot_N, plot_L, plot_core_stack, etc.
  helpers.py       # build_event_overlays, format_date_axis, ClipOptions, ScatterOverlay
```

---

## Final Layer Structure

| Layer | Module | Functions | Responsibility |
|-------|--------|-----------|----------------|
| 1. Primitives | `primitives.py` | `render_step`, `render_line`, `render_lambda`, `render_overlays` | Draw series + overlays on axes |
| 2. Charts | `charts.py` | `render_N`, `render_L`, `render_Lambda`, `render_w`, `render_A` | Complete chart definition (title, ylabel, colors, overlay logic) |
| 3. Layouts | `layouts.py` | `plot_N`, `plot_L`, `plot_core_stack` | Create figure, call charts, format axes, save |
| Shared | `helpers.py` | `build_event_overlays`, `ClipOptions`, `ScatterOverlay`, `format_date_axis` | Data types and utilities |

---

## Migration Mapping

| Current (ADR-002) | Proposed (ADR-003) | New Location |
|-------------------|-------------------|--------------|
| `render_step_chart` | `render_step` | primitives.py |
| `render_line_chart` | `render_line` | primitives.py |
| `render_lambda_chart` | `render_lambda` | primitives.py |
| (inline duplication) | `render_overlays` | primitives.py |
| `render_N_chart` | `render_N` | charts.py |
| `render_LT_chart` | `render_L` | charts.py |
| (new) | `render_Lambda` | charts.py |
| (new) | `render_w` | charts.py |
| (new) | `render_A` | charts.py |
| `draw_N_chart` | `plot_N` | layouts.py |
| `draw_LT_chart` | `plot_L` | layouts.py |
| `draw_four_panel_column` | `plot_core_stack` | layouts.py |
| `draw_five_panel_column` | `plot_five_panel_stack` | layouts.py |

---

## Implementation Order

1. Create module structure (`primitives.py`, `charts.py`, `layouts.py`)
2. Move primitives: `render_step`, `render_line`, `render_lambda` to `primitives.py`
3. Move/create chart recipes in `charts.py`: `render_N`, `render_L`, then add `render_Lambda`, `render_w`, `render_A`
4. Move layouts to `layouts.py` and rename to `plot_*`
5. Update imports in `core.py`, `stability.py`, `convergence.py`
6. Update/add tests for each layer
7. Remove deprecated functions from `helpers.py`

---

## Open Questions

- Should `ClipOptions` be used immediately, or keep explicit parameters until more charts need clipping?
- Should `helpers.py` be renamed to `types.py` or `common.py`?
- How to handle backward compatibility during migration (deprecation warnings vs. hard cutover)?
