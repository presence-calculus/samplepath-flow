---
ID: 22
Task: Process time convergence
Branch: process-time-convergence
---

Spec: In the convergence.py module, add a new panel, using the new panel conventions and design.
This panel should plot w(T), w'(T) and W*(T) against T. It should support event overlays,
w(T) should show drop lines on arrivals, w'(T) and W*(T) on departures.

Additional info
1. This should be a standalone panel written to convergence/panels and wired into plot_convergence_cahrts
2. Title is "Process Time Convergence" y label is "hours"
