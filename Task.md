---
ID: 6
Task: Extract cumulative flow diagram as core panel
Branch: cfd-component
---

Spec: Currently the cumulative flow diagram is implemented as an inline panel in the arrival/departure equilibrium convergence stack convergence.py:360:369. We need to extract it as a standard reusable component in core and write it as a core panel that can be reused in the convergence module. So, following our patterns we need.
1. A render_CFD method that renders the arrivals and eeparture events as a step chart.
2. A plot_CFD method that saves it as standalone chart using figure_context.
3. Wiring to pull it into the charts produced as part of the core flow metrics charts.
4. Update to reuse this component in the divergence module.
5. Drive it with tests.
6. Add support for overlays in the CFD: The arrival step chart should use arrival times and the departure step chart should use departue times. This might need updates to logic of overlays in the core step chart drawing code.
7. Add a fill in between the two lines of the CFD.

Progress:
- Added render_CFD and plot_CFD in core and wired into core flow metrics outputs.
- Added CFD tests covering baseline visual properties and output wiring.
- Convergence stack now reuses render_CFD with focused test coverage.
- Added CFD overlays using explicit arrival/departure times and purple/green markers.
- Added grey fill between arrivals and departures in CFD.
Status: done
