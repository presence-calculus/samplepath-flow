---
title: "<strong>TBD</strong>"
subtitle: ""
author: |
  <a href=""><em></em></a>

document-root: "../.."
header-image: "-root/assets/"
---


# Rethinking Flow

When I first started writing about flow here on The Polaris Flow Dispatch, I was brought in to the conventional wisdom.

The measurement playbook was intuitive: we started with some chosen unit of work called a work-item, and followed
it through a development process from start to finish. Throughput and cycle time gave us measurements
for the quantity of finished work over a reporting period, and the time it took to complete them.  Work-in-process and work item age gave us the corresponding
metrics for incomplete work. Little's Law supplied the relationship between the metrics. It gave us measurable conditions that let us distinguish between stable and unstable flow and since software development processes are often unstable, it gave us formal ways to measure process stability.

Being able to measure whether a process has stable flow is important for several reasons.  

First, instability is a symptom of a process that takes in more work than it can complete, and this is very common in software development. This gives us an operational proxy to measure development capacity, a quantity that is notoriously hard to measure directly in software.

Second, in a stable process, the standard flow metrics settle down to stable values when measured over sufficiently long time windows.  Little's Law tells us how their values must relate _to each other_ when they stabilize. This allows us to reason about _the process as whole_ over those horizons. In a stable process, the differences between flow metrics for completed work and incomplete work becomes insignificant, allowing us to treat the work items as a population whose statistical properties accurately measure the properties of the process as a whole. Further, when flow metrics stabilize, it gives us a plausible basis to posit that the future behavior of the process, as measured by those statistics, will resemble past behavior.  This becomes useful for forecasting.

Third, if you try to improve an unstable process, your cant measure the improvement when the underlying metric is unstable. It not clear whether the measured changes in process are due to the changes you made, or due to transients that were already present _in the measurement_. Stabilizing the process gives us a baseline where this is no longer an issue. If you can show that you have moved the process one stable baseline to another, then it becomes much simpler to show how and where you have made improvements without ambiguity.

The improvement playbook then becomes clear: focus on the process where you are trying to improve the flow of work. Measure if it is stable and if not stabilize it by first ensuring that ensuring that the process does not take in more work that it can complete. Once you have a stable process and flow metrics, focus on process improvements that let you move to a new stable state, where you can show unambiguous improvement in one or more of your flow metrics.

In practical terms this is the standard playbook for operational improvement in the flow of work based on Lean principles, that you would get with any of the popular Kanban methods in the industry. What was missing was a theoretically defensible approach to showing improvement using flow metrics, a gap I thought would be straightforward to fill since we already had the machinery of flow metrics to work with.

But it turned out to be a lot more complicated than I expected. A key roadblock was the fact that this approach ties the definition of process stability to meeting the conditions required to satisfy Little's Law. But when I measured throughput, cycle time, work item age and work in process used by commercial flow metrics tools, it turned out to be nearly impossible to prove stability this way. This was somewhat concerning, since it pointed to potential deeper disconnects between theory and practice. Something that would need to be resolved before we could claim we had a theoretically defensible method for improving the flow of work in software development.

Unraveling this thread and developing a set of techniques to measure flow in software development so that processes behave the way the theory says they should, has taken the better part of the last three years. It meant going back the original mathematical proofs of Little's Law to understand _why_ it holds. Last years edition of The Polaris Flow Dispatch, was devoted to these proofs of Little's Law and how they tell us what needs to change in the way we measure flow, and how we can use them to measure stability of software development processes. In the current edition we will build upon this material and dig deep into the underlying mathematical techniques and show how they can enable a much more rigorous approach to operational measurement in software development.
