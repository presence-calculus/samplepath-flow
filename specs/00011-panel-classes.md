# Panel Classes — Behavioral Spec

## Overview
Move core plotting panels from pure functions to panel classes that hold shared
configuration state (defaults + options). Panels expose `render(...)` and `plot(...)`
methods. Stacks remain functions focused on layout orchestration, receiving typed
panel instances.

This spec covers Phase 0 (prototype), Phase 1 (full N(t) migration), Phase 2.x
(migrate remaining panels), and prep work for core stack migration.

## Phase 0 — Prototype (N(t) panel class)
- Use dataclasses for panel objects.
- Panel instances own defaults and overrides `(title, derivation toggle, event marks, etc.).
- `render(...)` draws onto an axes using panel state.
- `plot(...)` establishes figure context and delegates to `render(...)`.
- Call sites use panel instances (no hidden defaults).
- Provide a migration path by keeping wrappers until all call sites are updated,
  then remove wrappers in a later pass.

## Non-Goals (Phase 0)
- Rewriting all panels beyond the prototype.
- Changing layout semantics or stack behavior.
- Reworking CLI arguments.

## Proposed API (Phase 0 prototype: N(t))

```python
@dataclass
class NPanel:
    show_title: bool = True
    title: str = "N(t) — Sample Path"
    show_derivations: bool = False
    with_event_marks: bool = False

    def render(self, ax, times, N_vals, *, arrival_times=None, departure_times=None) -> None:
        ...

    def plot(self, out_path, times, N_vals, *, unit="timestamp", caption=None,
             arrival_times=None, departure_times=None) -> None:
        ...
```

### Wrapper Policy (Phase 0)
- Keep `render_N(...)` and `plot_N(...)` as thin wrappers around `NPanel` to
  reduce disruption while call sites are migrated.

## Phase 1 — Migrate all N(t) call sites and remove wrappers
- Replace all `render_N(...)` usages with `NPanel(...).render(...)`.
- Replace all `plot_N(...)` usages with `NPanel(...).plot(...)`.
- Remove `render_N(...)` and `plot_N(...)` wrapper functions.
- Update tests accordingly.

## Phase 2.x — Migrate remaining panels (one per commit)
- Convert `render_L`, `render_Lambda`, `render_w`, `render_H`, `render_CFD`, etc.
  into panel classes with `render` and `plot`.
- Migrate call sites for each panel and remove wrappers per panel.
- Land each panel migration in a dedicated commit.

## Phase 3 — Prep for core stack migration
- Introduce typed panel groupings (e.g., `CorePanels`) to pass panel instances into
  stack functions.
- Refine stack signatures to accept panel instances rather than multiple kwargs.
- Preserve stack layout responsibilities within functions.

## Phase 4 — Stack architecture (core stack migration)
- Stacks remain functions, but accept a typed collection of panel instances:

```python
@dataclass
class CorePanels:
    n: NPanel
    l: LPanel
    lam: LambdaPanel
    w: WPanel
```

```python
def plot_core_stack(out_path, metrics, *, panels: CorePanels, layout_opts, caption=None, unit="timestamp"):
    with layout_context(...):
        panels.n.render(ax0, metrics.times, metrics.N, ...)
        panels.l.render(ax1, metrics.times, metrics.L, ...)
        panels.lam.render(ax2, metrics.times, metrics.Lambda, ...)
        panels.w.render(ax3, metrics.times, metrics.w, ...)
```

### Considerations for Phase 3/4
- How to share layout-level options (caption, suptitle, formatting) without moving
  them into panel classes.
- How to structure panel groups so stack functions remain ergonomic.
- Backward-compatibility or deprecation plan for existing stack function signatures.

## Testing Requirements (Phases 0–2.x)
- Unit tests for each panel class (`render` uses panel state).
- Update existing tests to reflect direct usage of panel instances.
- Remove tests for wrappers as wrappers are deleted.

## Acceptance Criteria (Phases 0–2.x)
- Phase 0: N(t) panel is implemented as a dataclass with `render` and `plot`.
- Phase 1: all N(t) call sites use the panel class; wrappers removed.
- Phase 2.x: each remaining panel migrated in its own commit with call sites updated.
