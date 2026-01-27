# 00013 SVG output

## Context
Current chart output defaults to PNG, which limits scalability and clarity for
web and large-format use. We need a configurable output format so users can
choose SVG for crisp, scalable charts and PNG for raster output.

## Decision
Introduce configurable chart output formats via CLI and `ChartConfig`.

### CLI (Chart Configuration)
- `--chart-format {png,svg}` (default: `png`)
- `--chart-dpi <int>` (optional; applies to PNG only)

### ChartConfig
Add fields:
- `chart_format: str`
- `chart_dpi: Optional[int]`

`ChartConfig.init_from_args(...)` maps CLI args to these fields.

### Save behavior
When saving figures:
- use `format=chart_format` for `fig.savefig(...)`
- apply `dpi=chart_dpi` only for PNG

### Scope (Phase 0)
- Implement format selection for core panel `NPanel.plot(...)`.
- Use a shared helper to build output path/extension and to save figures.
- Update tests for format routing and PNG dpi behavior.

## Non-Goals
- No multi-format output in the same run.
- No Plotly or HTML output in this phase.
- No stack layout changes beyond using the new save helper.

## Follow-ups
- Extend format selection to all panels and stacks.
- Add multi-format output (e.g., PNG + SVG) if requested.
