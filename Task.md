---
ID: 23
Task: Process Time Convergence Stack
Branch: process-time-convergence-stack
---

Spec: In convergence.py, the function plot_residence_vs_sojourn_stack should be replaced with
a stack that uses the modern panel plot design. The top panel is the process time convergence panel that we just built in task 22. The bottom panel is a new panel that plots a scatter plot of sojourn times for departures vs the two average residence time plots - w(T) and w'(T). This is basically the bottom panel of the current plot_residence_time_vs_sojourn_stack but extended to include both w and w' instead of just w.

So the task is
1. Define and implement the new panel.
2. Wire it up under convergence charts and save it under convergence/panels/residence_time_sojourn_time_scatter_plot.png
3. Create a new stack with process_time_convergence as the top panel and this one as the bottom panel and save it as converegence/process_time_convergence_stack
