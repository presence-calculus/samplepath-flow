---
ID: 5
Task: Centralize core chart plotting logic
Branch: adr-003-v2
---

Spec: See @decisions/003-plot-call-chain-simplification.md

Execution plan:
1. We will need to do this step-by-step. The previoius one-shot attempt was a failure.
2. I want to focus on migrating the functionality based on the modules that are invoked in sample_path_analysis.py:produce_all_charts.
3. We will migrate each of those top level functions to new implementations using the new architecture leaving the old architecture in place for the ones that have not been migrated.
4. We dont have too many tests for the plots.py module, so we will new tests for these as we implement them.
5. First migration target: `plot_core_flow_metrics_charts` (the first call in `produce_all_charts`).
   - Draft the new layering for this path only: a `plot_core_stack` layout that calls `render_N/L/Lambda/w` views and uses primitives for overlays/clipping.
   - Keep existing `plot_core_flow_metrics_charts` intact; add a v2 implementation alongside it and wire the call site when confident.
   - Add focused tests covering the new `plot_core_stack` layout (delegation + saved file) and the chart views it uses.
   - Once the v2 path is validated, switch `produce_all_charts` to the new function, leaving other plot families untouched for now.

---
Next task: We have ported the primary call site  from sample_path_analysis.py to the old module core, to core2. Now its time to port all the other call sites where these functions that were migrated from core to core2 are being called. There will still be some functions in core that we did not port over to core2 that can stay in core.
1. Build an inventory of call sites to port.
2. Analyze which tests are affected.
3. Build the plan to port the rest of the call sites over.
4. Let me review.

Put the plan below:

Inventory of call sites to port (migrated core chart functions still used elsewhere):
1) `samplepath/plots/stability.py` uses `render_N_chart` from helpers for WIP panels.
2) `samplepath/plots/core.py` uses `draw_N_chart`, `draw_LT_chart`, `draw_lambda_chart`, `draw_line_chart`,
   plus `render_N_chart`, `render_LT_chart`, `render_lambda_chart` inside legacy stack functions.
   These are legacy paths now that `produce_all_charts` uses core2, but still callable by direct imports.

Tests affected / needed:
- No direct plot tests cover stability or core legacy paths today. We should add targeted tests for any
  rewritten call sites (e.g., a stability plot panel calls `core2.render_N` or the new primitive).
- Existing core2 tests should continue to pass.

Plan:
1) Decide which legacy paths to keep vs port:
   - Update `stability.py` to use core2 chart views (`render_N`) or new primitives (`render_step_chart`) for N(t).
   - For `core.py`, either leave as legacy or make its entrypoints delegate to core2 (thin wrappers).
2) Implement the minimal call‑site rewires (start with `stability.py`).
3) Add unit tests for the updated call sites (mock renderers, one assertion per test).
4) Run focused tests and review outputs.
Progress:
- `stability.py` now uses `core.render_N` with a focused unit test.
- `draw_five_panel_column*` moved to `samplepath/plots/misc.py` to keep misc ownership.
- Legacy `core2.py` renamed to `core.py`; references updated.
Status: done
