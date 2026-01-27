---
ID: 15
Task: Add a new core stack to core.py
Branch: lt-derivation-stack-plot
---

Spec: We will add a few new plots to core to practice building new plots from scratch. The first plot is called the `LT_derivation_stack` which shows how L(T) is derived from the cumulative flow digram. The function should be named plot_LT_derivation_stack and it should write a plot in the core subdirectory with the base name `lt_derivation_stack` (format to be resolved from cli args)
The Title Should be "L(T) Derivation from Cumulative Flow Diagram"
 This chart shows the following panels in order
 - Cumulative Flow Diagram = A(T) vs D(T)
 - N(t) = A(T) - D(T)
 - H(T) = int_0^T N(t)dt
 - L(T) = H(T)/T
in that vertical order. It should support all the standard conventions and features of the existing plots including the ability to show event marks and derivations.

It should be called from plot_core_flow_metrics_charts
