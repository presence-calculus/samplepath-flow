---
title: "<strong>Some questions on Sample Path Analysis</strong>"
author: |
  <a href=""><em>Dr. Krishna Kumar</em></a>

---

# Background

Last week at Flowtopia, I gave a presentation on Sample Path Analysis — a method for measuring and reasoning about flow that starts from fundamentally different assumptions than most of the techniques we use today.

The ideas themselves are not new. The theory has been around for nearly 50 years and is well established in queueing theory and stochastic process literature. But it has largely remained within academic circles. Over the past four years, my work has focused on understanding that theory deeply and adapting it for practical use in operational analysis of real-world digital delivery systems.

I’ve been cautious about speaking publicly on this topic. I wanted to make sure I understood it thoroughly and could validate that it had concrete, practical value — not just theoretical appeal.

The series I wrote on Little’s Law in the Polaris Flow Dispatch was my first attempt to lay out the historical context and mathematical foundations of these ideas. It covered the roots and the broader implications, but not yet the mechanics.

What I’m doing now — starting with the Flowtopia sessions — is opening up the mechanics. I want to make the underlying method accessible to others in the flow community who may want to apply it. At the same time, I’m learning how to communicate these ideas more clearly to a wider audience.

After the workshop, Thorsten Speil reached out on LinkedIn with several thoughtful questions. He generously agreed to let me publish them along with my responses so others can benefit from the discussion.

If this exchange raises additional questions for you, feel free to post them. I’m happy to continue the conversation.


# Thorsten’s Questions (reproduced verbatim)

>Hello Krishna!
    I watched the recording of your Flow metrics Flowtopia webinar from Feb 5 and have some questions, if you like.

    * I understand that in the regular way to e. g. calculate "average WIP" over some fixed time-window like a week or a month, you get a pretty jittery chart over time, so it can be hard to tell what is a real signal and what is not. Do I understand one of your critiques here right, or how would you phrase it?
      - Q1: wouldn't it still be possible to draw an upper and lower control limit into this "jitter-chart" to get some idea of when we see "real" trends?
      - Q2: if my tool doesn't allow sample path analysis and I want a "working approximation", couldn't I just calculate and chart a moving average over past 3, 6, 12 months? That should "calm down" the chart and move a lot closer to the real behaviour?

    You advise to use "both" in parallel, right? The sample path metrics for tracking "slower" changes, plus more tactical metrics on a shorter time-scale?

    Q3: I don't recall you try to define "arrival" and "departure". The times in which an element counts as entering the system or leaving it needs to be consistent for all elements.

    Steve Tendon tries to make this more precise for his "Flow Time" calculation. You know his segmentation, I guess? He says, otherwise, people often measure so differently and have so much "fluff" in their metrics that they become pretty unreliable.
    What do you think of that?

    I attached a chart from Steve's TameFlow book to this post.
    "Flow Time" is just one part of the overall end-to-end time.
    From what I understand you might advise a company to a) decide what would be valuable for them to track over time - like duration of quotes or prioritization of requirements or development etc. ... then b) define their "start" and "end" times for these segments each and start calculating the respective sample paths? So if you want to track 3 different types of work, ("quote", "prioritization", "development"), then you would get sample path metrics for each of those separately?

    If you find these questions good to explain more of the differences and your "point", we can gladly post these/your answers elsewhere "publicly" on LI, Flowtopia, ...

    Best regards! Thorsten

Lets take each of these questions in turn, as they are all very important in understanding some of the subtle but important ways in which sample path analysis fundamentally differs from current techniques for measuring flow.


## What is "Average WIP"?

> I understand that in the regular way to e. g. calculate "average WIP" over some fixed time-window like a week or a month, you get a pretty jittery chart over time, so it can be hard to tell what is a real signal and what is not. Do I understand one of your critiques here right, or how would you phrase it?

The jitter is part of the problem, but it’s not the main issue.

The deeper question is: *what does “average WIP” actually mean?*

Most existing books and tools don’t define Average WIP carefully or compute it correctly — if they compute it at all. They measure *instantaneous WIP*, and if they produce an “average,” it is often an arithmetic average of those sampled instantaneous values.

The tell-tale sign is the x-axis of the CFD. If it is indexed by calendar periods — days, weeks, months — then you have already discarded the information needed to compute "Average WIP" correctly. Even Dan Vacanti’s *Actionable Agile* does this, and that is how most of us were originally introduced to CFDs.

The arrival and departure lines in a precisely constructed CFD are always step charts. They jump between values at events and stay constant in between.

When defined properly, WIP itself is a discrete _state_ variable for the process. When  WIP changes, the process transitions state. This happens _only when events occur_ (arrivals or departures). Let:

- $N(t)$ = instantaneous WIP = $A(T) - D(T)$
This is the difference between _cumulative_ arrivals and departures over a time interval and represents the _instantaneous_ value of WIP at the end point of that interval.

$N(t)$ is also a step chart, but it can go up and down unlike the two lines in the CFD. But like the lines in the CFD it can only jump at event times.

 Now we can talk about *Average WIP*. This is a time average not an arithmetic average.

- $L(T)$ = time-average WIP over $[0,T]$  

defined as

$$
L(T) = \frac{1}{T} \int_0^T N(t)\,dt
$$

$L(T)$ is an "average state", taken with respect to time. More precisely, a *time-weighted average* of the states the process occupied over the observation window. It is fundamentally different from $N(t)$.

Also, its not a statistical average. There is no sampling or distributions involved.

For example, suppose over a 30-day window the process spends:

- 5 days at WIP = 4  
- 20 days at WIP = 1  
- 5 days at WIP = 0  

Then the time-average WIP is:

$$
\frac{5 \times 4 + 20 \times 1 + 5 \times 0}{30}
= \frac{40}{30}
= 1.33
$$

>The key is that durations here are not multiples of reporting/sampling intervals. It is an *exact accounting of time spent in each state*, between events that drive the state changes, normalized by the window length. This quantity is what appears on the left-hand side of Little’s Law.

The LHS quantity in Little's Law is not $N(t)$. Even Actionable Agile confuses this, and I am yet to find a flow metrics tool that computes this correctly.

>- Q1: wouldn't it still be possible to draw an upper and lower control limit into this "jitter-chart" to get some idea of when we see "real" trends?
      - Q2: if my tool doesn't allow sample path analysis and I want a "working approximation", couldn't I just calculate and chart a moving average over past 3, 6, 12 months? That should "calm down" the chart and move a lot closer to the real behaviour?

Once you start sampling arrivals and departures at fixed reporting intervals and computing CFDs and metrics from those samples, you have already moved away from the true object that determines flow, and with it any hope of leveraging the true value of Little's Law - cause and effect reasoning.

From that point on, the issue is not whether you use moving averages, percentiles, or control limits on the base data afterward. The issue is that once you discard event ordering and exact timing, you have lost the ability to reason about the "real behavior" using the physics of flow.

Sample path analysis makes two commitments:

1. The sample path preserves the exact event order and the exact elapsed time between events.
2. Every metric is computed directly from that event-indexed path, so each change in any metric can be traced to the event that caused it to change, in the order it occurred on the event timeline.

That gives you deterministic cause-and-effect traceability. There is no ambiguity about why a metric has the value it has. Each value of a sample path flow metric at a point in time carries with it the event that caused it to take on that value. This deterministic behavior is the key to why everything works.

Statistical summaries can work when the system is already stable and observation windows are long relative to average residence time. In manufacturing environments, that condition often holds. In digital delivery, it almost always does not.

All these calculations are correctly implemented in this toolkit. The most direct way to understand the difference is to run both approaches on your own data and compare what they actually measure. I give you very detailed charts that let you look at each computation step separately and how the _events_ drive the metrics. Each of these charts is independently useful if you are trying to really diagnose flow problems, but in general, the summary stacks are what you look at most of the time.

>You advise to use "both" in parallel, right? The sample path metrics for tracking "slower" changes, plus more tactical metrics on a shorter time-scale?

My answer would be that you are free to use any tool, provided you understand the limitations. Current flow metrics tools treat flow metrics as statistical artifacts, and in doing so they make a category error. It's not a question of slower vs faster changes. N(t) accurately measures changes at the finest necessary granularity (this is lost when we sample).  L(T) allows you to accurately assess long run implications. Long run residence times capture both sojourn time and aging in a single metric and are the correct metric for process time at all timescales rather than the way we currently measure cycle time and age separately.

So the question is whether the flow metrics you calculate are fit for purpose.

 You can continue to use existing tools if you are satisfied with having somewhat rough and largely gut-feel perceptions of the state of flow. They are useful that way. Drawing trend charts, scatter plots, visualizing patterns in the data etc,thinking in terms of leading and lagging metrics all have their uses. But the underlying measurements are ad hoc.

The main difference is that sample path analysis gives provably correct and verifiable calculations for flow metrics with precise cause and effect semantics. So if you ask me, to precisely measure, diagnose and improve flow, sample path analysis is the only option. There is just a little bit of a learning curve in understanding why they work, and we could probably use nicer tools that build upon the things in this toolkit.

## Defining process time and setting boundaries

> Q3: I don't recall you try to define "arrival" and "departure". The times in which an element counts as entering the system or leaving it needs to be consistent for all elements.  
>  
> Steve Tendon tries to make this more precise for his "Flow Time" calculation. You know his segmentation, I guess? He says, otherwise, people often measure so differently and have so much "fluff" in their metrics that they become pretty unreliable.  
> What do you think of that?  
>  
> I attached a chart from Steve's TameFlow book to this post.  
> "Flow Time" is just one part of the overall end-to-end time.  

As you can see from Steve’s diagram, there are many possible ways to define arrivals and departures. Each segment in that picture can be interpreted as a state, and an arrival or departure can be defined as the event in which an element enters or leaves that state.

But these are modeling choices.

A flow process is fully defined once we specify:

1. What constitutes an element.  
2. An event that marks its entry (to something).  
3. An event that marks its exit (from something).  

Once those are defined consistently, everything else in sample path analysis follows mechanically.

Little’s Law does not require knowledge of the internal structure of the system. It requires only that arrivals and departures are well-defined and that arrivals precede departures for each element. If we violate those constraints, the metrics behave pathologically (for example, negative WIP becomes possible). But once those basic causal rules are respected, the mathematics is completely determined by the event sequence and time between events.

The power of sample path analysis is that it operates on behavior — the ordered sequence of observed events — rather than on structural descriptions of stages or workflows. Structure can be layered on top, but it is not required for the core invariants to hold.

In practical situations, of course, we often care about multi-stage or nested processes. As in Steve's example, we might distinguish:

- Order-to-Cash  
- Flow Time within development  
- Quote preparation time  

Each of these is simply a different flow process defined by a different pair of entry and exit events. Each has its own:

- WIP process  
- Presence mass  
- Time-average WIP  
- Residence time
- Sojourn time

The Presence Invariant applies independently to each one. Flow Time, Order-to-Cash are just names we give to sojourn time for each of these flow processes so that we can tell them apart once we look at the system of flow processes as a whole. This is all modeling.

When processes are nested, the state becomes richer. Instead of a single WIP value, we may track a vector of WIP values across internal states. But mathematically, this is still constructed from a single event timeline with appropriately marked events.

Ultimately, flow analysis reduces to modeling state transitions and the time spent in state by accurate accounting on the event timeline. That requires:

- Measuring all quantities against the same event-indexed timeline.  
- Using consistent definitions of entry and exit.  
- Computing all metrics over the same observation window.  

This last condition is where most existing flow metrics break down. It is common to see WIP sampled at one time, throughput computed over a different window, and cycle time averaged over yet another window. Once the observation windows are inconsistent, the presence invariant fails to hold, and the causal interpretation is lost.

Understanding structure is useful. But for reasoning about flow, what matters most is understanding behavior — the sequence of events and the durations between them.

That said, so far, none of what I have talked about, written, or implemented in code, addresses these more complex configurations of flow processes. This is on my to-do list once I get the basic ideas of sample path analysis for simple flow process with a single element and single entry/exit event pair, written down and communicated.
