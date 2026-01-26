---
ID: 9
Task: A(T) chart updates
Branch: at-chart-updates
---

Spec: We  have been using the sympbl A for the cumulative presence mass in the code. I have been using the symbol H(T) for this in the documentation.
I want to align the code and the charts to use H instead of A. And Also instead of using cumulate area in the chart title, we should use the term Cumulative Presence Mass.

All references to the quantity A in FlowMetics result and derived quantities like variable names plot_A etc.. should be replaced by H.

Progress:
- Renamed FlowMetricsResult A → H across code/tests and updated chart labels/titles.
- Updated H chart filename to `cumulative_presence_mass_H.png`.
- Updated docs/decisions to use H for cumulative presence mass and r_H for end-effects.
Status: done
