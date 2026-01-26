# 00012 Plot API rewiring

## Context
We want to simplify plot call sites and reduce long argument lists by shifting
data wiring into panel `plot(...)` methods. At the same time we want to decouple
panels from CLI args by introducing a global chart-focused config object that is
derived from the CLI's "Chart Configuration" section.

Today:
- Panel `plot(...)` methods take long, explicit argument lists and only provide
  a default figure context for `render(...)`.
- Top-level plot functions pass explicit data down to panel `plot(...)`.

## Decision
Introduce a global `ChartConfig` object and rewire panel `plot(...)` methods to
accept the top-level plot signature:

```
panel.plot(
    df: pd.DataFrame,
    chart_config: ChartConfig,
    filter_result: Optional[FilterResult],
    metrics: FlowMetricsResult,
    out_dir: str,
)
```

Responsibilities:
- `ChartConfig` contains chart behavior only (derived from CLI args).
- Panels use `ChartConfig` + `metrics` + `filter_result` to:
  - compute panel-specific data for `render(...)`
  - select output path
  - derive shared view details (caption, unit)
- `render(...)` remains explicit and accepts only the data needed for the plot.

Panel-specific overrides remain in the panel constructor (e.g., titles or
per-panel toggles), not in `ChartConfig`. These may be later externalized and merged into ChartConfig to provide
overrides say from a json representation, as well.

## Scope (Phase 0)
- Prototype with `NPanel` only.
- Add `ChartConfig` type in a shared location.
- Add a top-level mapping from CLI args to `ChartConfig` (in the plot entry). This should be of the form init_from_args so that we can potentially build this up from more sources than the cli args.
- Update `NPanel.plot(...)` to the new signature and call sites accordingly.
- Keep `NPanel.render(...)` unchanged and explicit.

## Non-Goals
- No stack layout changes in this phase.
- No refactor of other panels beyond the `NPanel` prototype.
- No changes to metrics computation or CLI flags.

## Open Questions
- Should `ChartConfig` carry `unit` (derived from `metrics.freq`) or should
  panels compute it directly from metrics? From metrics always.
- Where should caption formatting live (panel vs shared helper)? shard helper for now. we can revisit later.
