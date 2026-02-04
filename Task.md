---
ID: 35
Task: Flow cloud representation of a point process
Branch: point-flow-cloud
---

Spec: look at the diagram `beginnings-endings.png`. This is a visually evocative representation of flow
in a point process where the purple circles represent arrivals and green circles represent departures

The points are displayed from left to right in timestamp order. Points close to each other on the time axis are spread out a bit vertically, but still in timestamp order. The intention is visually indicate "flow" and congenstion and temporal order not necessary
precisely on the y-axis though.

Figure out how we can build this type of visualization using our point process representation and matplotlib.
