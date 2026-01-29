---
ID: 20
Task: Core primitives
Branch: core primitives
---

Spec: Add three more primitive core panels and wire them up into plot_core_flow_metrics

1. Indicator process for the arrival departure process. This is a scatter plot with am x-value for each arrival or departure timestamp and a y-value of 1 for each such point. The plot should show a drop line from each point to the x-value. We can think of this as the event overlay over the indicator process for the underlying arrival/departure point process. Use the same conventions on arrival/departure colors for points and drop lines.
2
2. A step chart for A(T) that supports event overlays and derivations
3. A step chart for D(T) that supports event overlays and derivations
