# ADR-004: Generalized Figure Layout Context

**Status:** Proposed  
**Date:** 2026-02-03  
**Author:** Krishna Kumar

## Context

We introduced `figure_context` to centralize figure creation, axis formatting, captioning, saving, and cleanup. This works well for single-panel plots, but the 4-row core stack required a custom layout (suptitle + optional caption + custom `tight_layout(rect=...)`). We broke out of `figure_context` to match the legacy output.

We want a layout abstraction that:
- Works for single-panel and multi-panel stacks/grids.
- Supports the core stack behavior exactly (suptitle, caption, `tight_layout` rect, shared axes).
- Allows per-axis formatting with optional shared-x behavior.
- Keeps a clean call site for rendering code.

## Decision

Introduce a generalized **layout context** that can express:
1) **Layout grid** (rows, cols, sharex/sharey, figsize)  
2) **Figure-level decorations** (suptitle, caption, layout rect)  
3) **Axis formatting policies** (format only bottom axis when sharex is True)

This can be implemented as either:
1) A new `layout_context` (preferred) with a small layout spec, or
2) A new `figure_context` variant (`figure_context_v2`) that accepts layout configuration

We will implement a **new context** to avoid breaking existing call sites. `figure_context` remains as a thin wrapper for the simplest layout.

## Proposed API

### Types

```python
@dataclass
class LayoutSpec:
    nrows: int = 1
    ncols: int = 1
    figsize: tuple[float, float] | None = None
    sharex: bool = False
    sharey: bool = False

@dataclass
class FigureDecorSpec:
    suptitle: str | None = None
    suptitle_y: float = 0.97
    caption: str | None = None
    caption_position: Literal["top", "bottom"] = "bottom"
    caption_y: float | None = None
    tight_layout: bool = True
    tight_layout_rect: tuple[float, float, float, float] | None = None
```

### Context

```python
@contextmanager
def layout_context(
    out_path: str,
    *,
    layout: LayoutSpec,
    decor: FigureDecorSpec | None = None,
    unit: str | None = "timestamp",
    format_axis_fn: Callable[[Axes, str | None], None] = _format_axis_label,
    format_targets: Literal["all", "bottom_row", "left_col", "bottom_left"] = "bottom_row",
    save_kwargs: dict | None = None,
) -> Iterator[tuple[Figure, Axes | np.ndarray]]:
    ...
```

#### Behavior
- Create `fig, axes = plt.subplots(...)` using `layout`.
- Yield `fig, axes` to allow rendering.
- On exit:
  - Apply axis formatting based on `format_targets`:
    - `bottom_row`: last row only (typical for `sharex=True`)
    - `left_col`: first column only (typical for `sharey=True`)
    - `bottom_left`: both bottom row and left column
    - `all`: every axis
  - Apply `FigureDecorSpec`:
    - `suptitle` if present.
    - `caption` if present (via `fig.text`) with `caption_position`.
    - If `caption_y` is `None`, default to `0.945` for `top` and `0.005` for `bottom`.
    - `tight_layout` with an optional `tight_layout_rect`.
  - Save and close.

### `figure_context` compatibility wrapper

```python
def figure_context(out_path: str, *, nrows=1, ncols=1, figsize=None, sharex=False,
                   caption=None, unit="timestamp", save_kwargs=None, tight_layout=True,
                   format_axis_fn=_format_axis_label):
    layout = LayoutSpec(nrows=nrows, ncols=ncols, figsize=figsize, sharex=sharex)
    decor = FigureDecorSpec(
        caption=caption,
        tight_layout=tight_layout,
        tight_layout_rect=None,
    )
    return layout_context(
        out_path,
        layout=layout,
        decor=decor,
        unit=unit,
        save_kwargs=save_kwargs,
        format_axis_fn=format_axis_fn,
        format_targets="bottom_row" if sharex else "all",
    )
```

## Core Stack Mapping (Drop-in Replacement)

Legacy behavior (core stack) can be represented as:

```python
layout = LayoutSpec(nrows=4, ncols=1, figsize=(12, 11), sharex=True)
decor = FigureDecorSpec(
    suptitle="Sample Path Flow Metrics",
    suptitle_y=0.97,
    caption=caption,          # "Filters: " if empty to preserve spacing
    caption_position="top",
    caption_y=0.945,
    tight_layout=True,
    tight_layout_rect=(0, 0, 1, 0.96),
)

with layout_context(out_path, layout=layout, decor=decor, unit=unit) as (fig, axes):
    ax = axes.ravel()
    render_N(ax[0], ...)
    render_L(ax[1], ...)
    render_Lambda(ax[2], ...)
    render_w(ax[3], ...)
```

The above captures:
- 4x1 layout
- shared x-axis formatting only on bottom axis
- suptitle + caption + custom `tight_layout` rect

## Single-Panel Mapping

```python
layout = LayoutSpec(nrows=1, ncols=1)
decor = FigureDecorSpec(caption=caption, tight_layout=True)

with layout_context(out_path, layout=layout, decor=decor, unit=unit) as (fig, ax):
    render_N(ax, ...)
```

## Constraints & Notes

- The API must allow:
  - `tight_layout_rect` to be provided explicitly (for legacy matching).
  - Caption optional; but the caller can pass `"Filters: "` to preserve spacing.
  - Axis formatting policies (bottom row / left column / all) with shared axes.
- The context should not assume Matplotlib only forever; keep the interface backend-friendly.
- `layout_context` should not do chart-specific formatting; only figure/axis-level.
- `_first_axis` remains a utility helper and is not part of the context API.

## Migration Plan

1) Implement `layout_context` and `LayoutSpec`/`FigureDecorSpec` in `figure_context.py`.
2) Update `figure_context` to be a compatibility wrapper.
3) Switch `core2.plot_core_stack` to use `layout_context` with the legacy decor spec.
4) Migrate other multi-panel layouts as needed.

## Consequences

**Pros**
- Unified abstraction for single and multi-panel layouts.
- Preserves legacy layout behavior without custom per-layout code.
- Makes future backend adaptation easier (layout logic centralized).

**Cons**
- Slightly more configuration at call sites (layout + decor specs).
- Some chart-specific layout details still needed (e.g., caption spacing).

---

## Notes on Defaults

- If `LayoutSpec.figsize` is `None`, use the existing auto-scaling heuristic
  (e.g., `(10.0, 3.4 * nrows)`), matching the current `figure_context` behavior.
