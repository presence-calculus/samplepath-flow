---
ID: 33
Task: Line color in convergence plots with events-enabled
Branch: convergence-plots-line-colors
---

Spec: Currently, we have the behavior that whevent overlays are enabled, the corresponding line are colored with grey. This does not work when there are multiple series on the chart like in the convergence panel, since all the series have the same color and cannot be distinguished from each other. Instead, let us change the behavior for these multiple series charts so that instead of greying out the lines, we increase their opacity so that the event markers become prominent relative to the lines. This way we can still visually distinguish the different series while still showing the markers and without overwhelming the chart as a whole.

I would like to start by making this change on the ProcessTimeConvergence chart. Once we fix this chart we can roll it out to the remaining charts in the convergence package. Limit the scope to the convergence package for now. No need to address the other packages.
