---
ID: 5
Task: Centralize core chart plotting logic
Branch: core-chart-plotting-refactor
---

Spec: In the plotting logic in core.py, the logic to darw individual charts for  N(t), L(T) etc in plot_core_flow_metrics_charts.py and the plots where we show the composite stacks like draw_four_panel_column, we need a way to centralize the logic for display of each chart so that we can reuse it in different layouts.

So the exact same capabilities should be available to be dropped into a single chart or a composite stack.

Make a plan on how we can refactor the code to acheive this.
