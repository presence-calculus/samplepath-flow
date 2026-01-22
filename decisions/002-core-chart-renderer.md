# ADR-002: Centralized Chart Rendering

**Status:** Implemented
**Date:** 2026-02-02
**Author:** Krishna Kumar

## Context

Plotting code in `samplepath/plots/core.py` had significant duplication:

- The same chart (e.g., N(t) step chart) was rendered in 4+ places with minor variations
- Composite layouts (`draw_four_panel_column`, `draw_five_panel_column`) duplicated axis formatting, titles, labels, and legend handling
- Adding features like event overlays required changes in multiple locations
- No single source of truth for how each chart type should appear

## Decision

Introduce a two-layer architecture for chart rendering:

1. **Low-level render functions** — Generic plotting primitives that render to an existing `Axes`
2. **Chart recipes** — Complete chart definitions that encapsulate all visual aspects of a specific chart type

Chart recipes are the single source of truth for each chart type. They work identically whether rendering to a standalone figure or a panel in a composite layout.

---

## Architecture

### Layer 1: Low-Level Render Functions

Generic primitives in `helpers.py` that handle the mechanics of rendering:

```python
def render_step_chart(ax, times, values, *, label, color, fill, overlays) -> None
def render_line_chart(ax, times, values, *, label, color, overlays) -> None
def render_lambda_chart(ax, times, values, *, label, pctl_upper, pctl_lower, warmup_hours) -> None
```

These functions:
- Take an existing `Axes` object
- Render the series with specified styling
- Handle overlay rendering (scatter points, drop lines)
- Do not set titles, ylabels, or legends — that's the recipe's job

### Layer 2: Chart Recipes

Complete chart definitions that encapsulate all visual aspects:

```python
def render_N_chart(
    ax: Axes,
    times: Sequence[pd.Timestamp],
    N_vals: Sequence[float],
    *,
    arrival_times: Optional[List[pd.Timestamp]] = None,
    departure_times: Optional[List[pd.Timestamp]] = None,
    with_event_marks: bool = False,
    show_title: bool = True,
) -> None:
    """Single source of truth for N(t) charts."""
    overlays = build_event_overlays(times, N_vals, arrival_times, departure_times) if with_event_marks else None
    color = "grey" if overlays else "tab:blue"

    render_step_chart(ax, times, N_vals, label="N(t)", color=color, fill=True, overlays=overlays)

    if show_title:
        ax.set_title("N(t) — Sample Path")
    ax.set_ylabel("N(t)")
    ax.legend()
```

Chart recipes:
- Define the complete visual appearance (title, ylabel, legend label, colors)
- Build overlays from event times by mapping to series y-values
- Delegate actual rendering to low-level functions
- Support both standalone (`show_title=True`) and composite contexts

### Overlay Construction

The `build_event_overlays()` helper implements the key principle for point process overlays:

- **x-values** come from point events (arrival/departure times)
- **y-values** are looked up from the series at matching x positions
- Events with no matching x in the series are excluded

```python
def build_event_overlays(times, values, arrival_times, departure_times, drop_lines=True):
    time_to_idx = {t: i for i, t in enumerate(times)}
    arrival_x = [t for t in arrival_times if t in time_to_idx]
    arrival_y = [float(values[time_to_idx[t]]) for t in arrival_x]
    # ... same for departures
    return [ScatterOverlay(...), ScatterOverlay(...)]
```

### Standalone Wrappers

Thin wrappers that create a figure, call the recipe, and save:

```python
def draw_N_chart(times, N_vals, out_path, *, arrival_times, departure_times, with_event_marks, caption):
    fig, ax = init_fig_ax()
    render_N_chart(ax, times, N_vals, arrival_times=arrival_times, ...)
    format_date_axis(ax)
    format_fig(caption, fig)
    fig.savefig(out_path)
    plt.close(fig)
```

---

## Implementation

### Charts Consolidated

| Chart | Recipe | Standalone Wrapper |
|-------|--------|-------------------|
| N(t) sample path | `render_N_chart()` | `draw_N_chart()` |
| L(T) time-average | `render_LT_chart()` | `draw_LT_chart()` |

### Functions Updated

- `draw_four_panel_column()` — uses `render_N_chart`, `render_LT_chart`
- `draw_five_panel_column()` — uses `render_N_chart`, `render_LT_chart`
- `draw_five_panel_column_with_scatter()` — uses `render_N_chart`, `render_LT_chart`
- `plot_core_flow_metrics_charts()` — uses `draw_N_chart`, `draw_LT_chart`
- `stability.py` — uses `render_N_chart`

### Files Modified

- `samplepath/plots/helpers.py` — Added chart recipes, overlay helper, enhanced render functions
- `samplepath/plots/core.py` — Updated to use centralized recipes
- `samplepath/plots/stability.py` — Updated to use `render_N_chart`

---

## Consequences

**Positive:**
- Single source of truth for each chart type's visual appearance
- Event overlays work consistently in standalone and composite layouts
- Adding new visual features (colors, styling) requires changes in one place
- Clear separation between rendering mechanics and chart definition

**Negative:**
- Additional indirection (recipe → render function)
- Chart recipes have domain-specific knowledge (e.g., "grey when overlays present")

---

## Future Direction

This consolidation prepares for backend-agnostic chart rendering. A future ADR will address:

- `ChartSpec` dataclass as a pure data contract (no Matplotlib types)
- `RenderBackend` protocol for Matplotlib/Plotly/Bokeh backends
- Migration path from chart recipes to spec-based rendering

The current chart recipes can be mechanically converted to build `ChartSpec` objects when that abstraction is needed.

---

## Remaining Work

Charts not yet consolidated (still using direct `render_line_chart` calls):

- Λ(T) — has special percentile clipping, needs `render_Lambda_chart` recipe
- w(T) — needs `render_w_chart` recipe
- A(T) — needs `render_A_chart` recipe

These will follow the same pattern established by `render_N_chart` and `render_LT_chart`.
