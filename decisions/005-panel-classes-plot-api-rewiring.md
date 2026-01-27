# 005 Panel Classes + Plot API Rewiring

## Status
Accepted

## Context
The charting layer was hard to extend because panel rendering relied on
long argument lists and ad hoc wiring at call sites. We need a clearer,
object-oriented surface that scales to new backends and simplifies the
call chain without losing explicit rendering.

This ADR consolidates the two implemented specs:
- `specs/00011-panel-classes.md`
- `specs/00012-plot-api-rewiring.md`

## Decision
1) **Panel classes** are the primary unit of chart rendering. Each panel
   owns its defaults and exposes:
   - `render(ax, ...)` for explicit rendering into an existing context
   - `plot(...)` for creating a figure context and delegating to `render`

2) **ChartConfig** is the shared, chart‑focused configuration derived
   from CLI args (and later other sources). It is passed into plot
   methods so panels can resolve behavior without depending on CLI args.

3) **plot_core_stack** now follows the same standard plot signature
   `(df, chart_config, filter_result, metrics, out_dir)` and relies on
   panel defaults instead of re‑declaring titles or show_title flags.

## Consequences
- Call sites are simpler and more uniform.
- Panel defaults are the source of truth unless explicitly overridden.
- Chart wiring is centralized in panel `plot(...)`, while `render(...)`
  stays explicit and composable.
- Future overrides (e.g., titles) can be added to ChartConfig without
  changing call sites.

## Follow‑ups
- Extend the plot‑signature pattern to remaining panels and stacks.
- Evaluate ChartConfig overrides once real use cases emerge.
