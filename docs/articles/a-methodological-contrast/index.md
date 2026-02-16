---
title: "<strong>Sample Path Analysis vs Statistics</strong>"
subtitle: "Contrasting Methodologies"
author: |
  <a href=""><em>Dr. Krishna Kumar</em></a>

citations: true
---

Many traditional flow metrics implementations (for example, those described by Vacanti [@vacanti2015]) compute metrics over uniform reporting intervals in order to preserve unit consistency and comparability across time. In this construction, the event-time process is partitioned into fixed calendar periods, and measurements are indexed by reporting period rather than by exact event timestamps. To the best of my knowledge, every commercial flow metrics tool on the market works this way.

 In these implementations the aggregation occurs *before* measurement: the reporting windows define the object being analyzed, not merely the way it is displayed. This is a key difference between the way calendar-index in sample path flow metrics works, and the current flow metrics tools

Unlike in sample path analysis, fine-grained event data are replaced with counts aggregated over reporting periods, and those aggregates become the _input_ to all subsequent measurements. Once this transformation occurs, the original event-time process is no longer recoverable from the data used to construct the flow metrics — including fundamental constructs such as the Cumulative Flow Diagram.

Formally, given a reporting grid $\{t_k\}$, the mapping from the ordered event timestamps $\{a_i\}$ to interval counts $\{A(t_k)\}$ is many-to-one. Information about event ordering and spacing within each reporting window is lost. The resulting analysis is therefore performed on an aggregated, interval-indexed process rather than on the original event-time trajectory itself.

This has consequences. Changing the index set from event time to reporting period replaces the original event-indexed process with an interval-indexed aggregate process. The intra-interval temporal structure — including exact ordering, spacing, and interaction of events — is not preserved in the metric definition itself.

Aggregation before measurement introduces structural ambiguity. Once fine-grained event information is discarded, multiple distinct event-time trajectories become observationally indistinguishable. Deterministic relationships that hold pathwise may then appear only as _statistical_ relationships in the aggregated representation.

It is natural then, to interpret flow metrics as statistical constructs derived from sampled reporting intervals rather than as deterministic functionals of a realized event-time trajectory. The object under analysis has changed: instead of reasoning about the evolution of a path, we reason about aggregates indexed by reporting periods.

A representational choice becomes a conceptual mismatch when relationships that are deterministic at the path level are treated as statistical properties of sampled data. Flow analysis then devolves into reasoning about statistical summaries over aggregates, averages, percentiles etc, rather than direct reasoning about the underlying deterministic pathwise relationships.

This issue becomes especially acute in non-stationary processes. Input aggregation implicitly assumes that behavior within each reporting window can be summarized without loss of essential structure. In a non-stationary environment — where rates, ordering, and interactions evolve over time — the intra-interval dynamics often determine future behavior. As we have seen, it is the tightly coupled relationships between the various flow state variables that determine the dynamics of the flow processes, and these dont conform to externally imposed calendar boundaries, Flow is endogeneous and bleeds across such boundaries, and recognizing this is key to measruring and analyzing flow.

Sample path analysis takes a more principled approach. Metrics are defined as deterministic functionals of the full event history, indexed by event time at the resolution at which the events were observed. Any subsequent sampling is applied only to the already-computed pathwise _metrics_ purely for visualization or reporting purposes. All the metrics being reported are sample consistently at the same _points in time_ at coarser granularities, and the presence invariant ensures that all the relationships that hold at every instant in the finest timescales also hold for the sampled metrics.


So Preserving information structure and invariants comes before preserving units.  Metrics in current flow analysis tools operate on an aggregated representation of the process, whereas event-indexed metrics in sample path analysis preserve the exact temporal evolution of the underlying events.

Consistency of units in displaying _metrics_ can still be achieved by scaling measurements computed at fixed event resolution into human-friendly units, as we routinely do when _presenting_ sample path metrics. This scaling preserves unit consistency without altering the underlying object being measured or discarding temporal structure.

Crucially, the validity of these pathwise relationships does not depend on stationarity. The same deterministic functionals apply to both stationary and non-stationary processes, because they are defined directly on the realized event history rather than on assumptions about long-run statistical equilibrium.

This is the methodological advantage of sample path analysis: it preserves the full event-time structure of the process and therefore retains access to deterministic relationships that interval-aggregated statistical approaches cannot represent.

I would argue that this is the only _correct_ way of measuring flow metrics.

## References
