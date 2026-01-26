---
ID: 12
Task: Plot API rewiring
Branch: plot-api-rewiring
---

Spec: 00012-plot-api-rewiring

Proposal
- Add global ChartConfig derived from CLI "Chart Configuration" section.
- Panel plot signature becomes: plot(df, chart_config, filter_result, metrics, out_dir).
- Panel constructors retain per-panel overrides (titles, toggles).
- render(...) remains explicit and data-focused.

Phase 0 (this task)
- Add ChartConfig type and CLI->ChartConfig mapping at the plot entry.
- Migrate NPanel.plot to the new signature.
- Update NPanel call sites and tests.

Clarifications
- Provide ChartConfig.init_from_args(...) to allow future non-CLI sources.
- ChartConfig should not carry unit; panels derive unit from metrics.freq.
- Caption formatting should use a shared helper (panel-independent).
