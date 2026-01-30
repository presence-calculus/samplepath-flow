# Coding Style Guide

## General function signature rules
- Group related keyword args together in signatures: `show_title`, `title`, `show_derivations` (feature toggle first, then feature controls).
- Maintain a consistent order for calling functions with keyword args: they should be in the same order they are declared in the function signature.

## Creating charts and plots

- Review ADR in decisions/006-plotting-architecture-and-conventions.md
- After implementing a change to the plot review that the it is compliant with the conventions and report any issues with compliance.

### Panel Patterns

#### 1. Standard Panel (e.g., NPanel)

```python
@dataclass
class NPanel:
    show_title: bool = True
    title: str = "N(t) — Sample Path"
    show_derivations: bool = False
    with_event_marks: bool = False

    def render(self, ax, times, N_vals, *, arrival_times=None, departure_times=None):
        # Call primitives (render_step_chart, build_event_overlays)
        # Set title, ylabel, legend

    def plot(self, metrics, filter_result, chart_config, out_dir) -> str:
        # Use figure_context, call self.render(), return path
```

**Key points:**
- Dataclass with optional config fields
- `render()` takes an `Axes` + raw data, calls primitives, sets decorations
- `plot()` wraps `render()` in `figure_context`, returns output path

#### 2. Custom Panel (e.g., LLWPanel)

```python
@dataclass(frozen=True)
class LLWPanel:
    # Uses figsize=(6.0, 6.0), unit=None (not time-based)
    # Direct matplotlib calls (ax.scatter, ax.vlines, ax.hlines)
    # Square aspect ratio with ax.set_aspect("equal")
```

**Key points:**
- Custom rendering when primitives don't fit
- `unit=None` because axes aren't timestamps
- Still follows `render()`/`plot()` split

#### 3. Layout (e.g., plot_core_stack)

```python
def plot_core_stack(metrics, filter_result, chart_config, out_dir) -> str:
    layout = LayoutSpec(nrows=4, ncols=1, figsize=(12.0, 11.0), sharex=True)
    decor = FigureDecorSpec(suptitle="...", caption="...", ...)

    with layout_context(layout=layout, decor=decor, ...) as (_, axes, path):
        NPanel(...).render(axes[0], ...)
        LPanel(...).render(axes[1], ...)
        # ...
    return path
```

**Key points:**
- Uses `layout_context` (not `figure_context`)
- Calls `render()` directly on panels — **not** `plot()`
- Orchestrates multiple panels into one figure

## Metric computations vs chart rendering

A plot module function should *never* compute a derived metric. It should always take its metrics from
a precomputed argument passed in by a caller. If you find yourself neededing to compute a new metric inside
a plot or render function, stop and propose a change to the metrics api, and get it confirmed before proceeding.



### Testing Patterns

| Pattern | Approach |
|---------|----------|
| Mock axes | `ax = MagicMock()` |
| Patch primitives | `patch("samplepath.plots.core.render_step_chart")` |
| Patch contexts | Fake context manager yielding `(fig, ax, "out.png")` |
| Fixtures | `SimpleNamespace` for metrics with minimal data |
| Assertions | One per test, verify call args and kwargs |

### Call Flow Diagram

```
Single Chart:  Panel.plot → figure_context → Panel.render → primitives
Stack Layout:  plot_core_stack → layout_context → Panel.render (×N) → primitives
Custom Chart:  LLWPanel.plot → figure_context → LLWPanel.render → direct matplotlib
```
