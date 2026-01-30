---
ID: 25
Task: Core scatter plots
Branch: core-scatter-plots
---

Spec: We need two additional scatter plots in the core panels.

1. Sojourn time scatter plot: this is a point process, the very same plot we overlaid with average residence time lines in the convergence module. The x-axis is the sequence of departure time stamps and the y-axis has the sojourn times for each departure.
2. Residence Time scatter plot: this is a also a point process, but here the x-axis has arrival time stamps and the y-axis has the _current_ residence time for that arrival. For completed items it is the sojourn time, and for incomplete items it is the age of the item at the end of the overall observation window (T).

For these plots assume the following conventions:
1. Drop lines will use arrival/departure colors.
2. Points will use arrival colors for incomplete points and departure colors for complete points.

For each of these, write a standalone core panel and wire it up to the core chart driver.
