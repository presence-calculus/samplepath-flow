---
ID: 7
Task: Add point overlays to Little's Law Invariant Chart
Branch: llw-point-overlay
---

Spec: Currently the l_lambda_w chart does not support event overlays even though it suports vertical and horizontal drop lines. However, every point on the chart represents either an arrival or departure event, even though by the time we draw the chart we no longer are looking at a time stamp. We can capture the point information on the chart as follows:
1. For each point color the point purple if it is an arrival and color it green if it represents a departure.
2. (experiment) Use a color gradient or opacity to represent time, with lighter color representing earlier points and deeper color or opacity to represent later time. So we can look at the chart and understand the temporal trends of where values are clustering *now* vs *earlier*.

Progress:
- Added LLW event-colored points with drop lines, legend, and time-based opacity.
Status: done
