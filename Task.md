---
ID: 19
Task: Departure Focused Metrics
Branch: departure-metrics
---

Spec: Currently flow metrics are arrival focused, we use arrival rate \lambda and w(T) which is avg presence mass per arrival.
This task makes first class metrics that focus on departure rate, the departure rate and w'(T) which is avg presence mass per departure.

We currently show the departure rate charts in the convergence.py module, but I want to elevate it as a first-class core metric. In this task we will

1. Implement a Theta panel that plots \Theta(T) = D(T)/T the cumulative departures as function of the length of the observation interval.
2. Compute a w'(T) metric w'(T) = H(T)/D(T) in FlowMetrics
3. Display a w'(T) panel
4. Create a departure focused core flow metric stack showing N(T), L(T), \Theta(T) and w'(T).
5. Show the departure focused presence invariant L (T) = \Theta(T).w'(T)

We will implement these one at a time.

Follow ups:

# 6. Improving invariant visualizations

Looking at the llw chart and ltw' charts, we have some opportunties to clarify the symmetries between then in the visualization.
Both charts show the same set of points and show that *both* invariants hold at *all* points. But there is an opportunity to high light this with the
the event_marks. We should show drop lines at arrival marks in the arrival focused invariant (llw) and show departure drop lines in the departure focused invariant.
And to make things clear, we should color the arrival and departure marks in both cases, so we can see that there is a departure mark without a drop line in the
llw case colored green and an arrival mark without drop lines in the ltw' case.

# 7. w_prime panel must show only departure drop lines when with-event-marks is true.

# 8. For consistency we should rename w(T) avg. residence time per arrival in all the visible chart titles etc.

#9. Put together a consoldiated summary of the changes we made in this branch. Propose a set of edits to the chart reference documentation and let me review them.
