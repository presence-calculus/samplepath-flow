---
ID: 4
Task: Add arrival/departures marks to N(t) chart
Branch: arrival-departure-marks-nt
---

Spec: A key property of sample path analysis is that all the charts are driven by arrival and departure events. So the discontinuities in the charts correspond to events. This is in contrast to most current flow metrics charts where the time axis is organized by calendar date.

To highlight this fact, I want to show an event indicator on every point in the key sample path flow metrics charts. We will start with N(t) the instantaenous WIP sample path chart.

For this chart, I want to show an arrival as purple colored dot and a departure as a green colored dot.

Add a command line option -with-event-marks to turn this feature on an off globally for all charts that support it.
