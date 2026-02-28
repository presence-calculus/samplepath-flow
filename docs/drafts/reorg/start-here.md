---
title: "<strong>Sample Path Analysis of Flow Processes</strong>"
subtitle: "A Gentle Introduction"
author: |
  <a href="https://polarisadvisor.com"><em>Dr. Krishna Kumar</em></a>

document-root: "../.."
header-image: "$document-root/assets/who-ordered-this.png"


# 1. Cross-Reference Settings (Sections & Figures)
numberSections: true
sectionsDepth: 2
figPrefix: "Figure"
figureTitle: "Figure"

# 2. Table of Contents
toc: true
toc-depth: 2

# 3. Citations
citations: true
link-citations: true
---

# Introduction

Flow metrics and flow-analysis techniques have been used for operations management in the software industry for nearly 20 years. This lineage runs from the Poppendiecks [@poppendieck2003; @poppendieck2006], who introduced lead time, throughput, and work in progress (WIP) to Lean software development, to David Anderson’s popularization of the Kanban method [@anderson2010] as an empirical management approach for knowledge work. As Value Stream Management matured and flow-based improvement became embedded in DevOps and product organizations, a broader toolkit of metrics and analytical methods, including those popularized by Dan Vacanti [@vacanti2015], was adopted for operational management and forecasting.

Measuring and managing flow is now a well-understood baseline practice, even if adoption and execution vary widely. It may be surprising, then, that I argue we need substantially different ways to measure, model, and reason about flow in the software industry. This document and related work make that case.

The argument at a high level is this: our application of these ideas in software is built on an incomplete understanding of what the metrics measure, why they are relevant in our context, and how precisely they can support decisions across different contexts.

For example, it is widely accepted that throughput and process-time metrics such as lead time and cycle time are important performance measures in a delivery context. But in adapting them to software development, we often do not account for the economic context in which we measure them. Throughput alone does not always translate into meaningful economic outcomes. Process time in its various forms, and cost of delay, are often equally important economically, yet we still have few useful ways to connect those quantities to operational decision-making [^-wsjf].

[^-wsjf]: WSJF job scheduling is, of course, one of the "standard" answers here, but it is a relatively crude heuristic and only a minor application of a deeper and more nuanced topic.

Further, [Little's Law](https://docs.pcalc.org/articles/littles-law) shows that these concepts are not independent. It implies structural relationships between process time and throughput that manifest as economic trade-offs, constraining the courses of action available as we pursue specific outcomes.

While Little's Law is widely name-checked to justify common practices, the throughlines between those practices and the underlying theory often rest on assumptions derived from their original application in manufacturing. There are software-development contexts where these assumptions are valid. For example, if you build and deliver software on a fixed-price contract basis, or your business model is based on marking up the specialized skills of your people, traditional economic models of throughput and resource utilization are often the right ones, and we can get direct leverage by adopting those ideas, as many careful practitioners do.

But if you are selling a SaaS product, product-development throughput is not nearly as important as careful selection of what goes into the product, understanding cost of delay, and optimizing delivery processes to stay responsive to shifting markets. Market performance depends on much more than delivering the right product. Time to market, network effects in the ecosystem in which the product operates, and related factors play a much larger role. So if we are focusing on product development in this context, we need tools that can incorporate these notions into our modeling assumptions _before_ we design operational processes.

Between these poles lies a range of contexts — internal IT, sales operations, marketing, R&D — each with different economic drivers and different relationships between process time, throughput, and value. We cannot apply the same techniques to reason about these quantities and expect them to work equally well in all contexts.

Yet today, we mostly live in that awkward middle ground: we allude to these larger concerns when talking about process improvement, but our prescriptions and methodologies largely remain shackled to a limited set of generic ideas borrowed from industrial production: reduce WIP, remove waste, small batches, faster feedback, and so on. It's not that these are not valid things to focus on, but they are treated as universal solutions, applicable without business context, with no reliable way to connect how and why adopting any of them improves business outcomes, and no systematic way to measure and prove they work in a given context.

Our first objective in this document is to shore up those foundations. Doing so reveals where current practices fit within the broader mathematical context provided by a rigorous theory of flow processes. It helps us understand where and when certain approaches are appropriate, and whether a particular improvement method is likely to be effective before implementation.

Such claims require proof: first, that a real limitation exists in prevailing methods; second, that the proposed alternatives resolve it. This document lays out that case at a high level. The broader project and documentation provide the mathematical foundation, derivations, and open-source tooling needed to validate these claims directly on your own data.

None of the core mathematical ideas we rely on to make this case are new. Many were established decades ago in queueing theory[^-queueing-theory]. In particular, the sample path analysis techniques underlying this work trace to Shaler Stidham’s 1972 deterministic proof of [Little’s Law](https://docs.pcalc.org/articles/littles-law/) and are mainstream complements to statistical and probabilistic analyses of stochastic processes, which is exactly how they are used here. Little’s Law is foundational to flow analysis, though in software delivery it is often invoked loosely, without exploiting its full structural implications. This is another key point of departure for our methods.

[^-queueing-theory]: Much of traditional queueing theory cannot be applied as-is in the environments where software development happens, yet those distinctions are often poorly understood. As we will see, several more foundational ideas need to be established before we tackle queueing directly. Once those are in place, a rigorous foundation for managing operational queues emerges naturally.

The theoretical foundation for our methods, which draws clear distinctions between concepts such as flow analysis, queue management, and economic analysis, is presented in *Sample Path Analysis of Queueing Systems* by El-Taha and Stidham [@eltaha1999]. Our contribution is practical: translating this theory into operational tools that can be applied directly to real software-delivery systems, and simplifying terminology where needed so it maps more clearly to the domain without losing rigor.

Applying these ideas requires conceptual shifts if you are very familiar with current methods. This document introduces those shifts and points to the theory and tooling needed to verify each claim. While the underlying mathematics is elementary, the perspective shift is significant and may be disorienting if you are comfortable with current techniques.

To make it easier to navigate these ideas, our toolkit is designed to work with _exactly_ the same inputs you use to measure and analyze flow today. The results will differ, but we will explain why each difference matters. Rather than read this as an abstract theoretical treatise, I encourage you to compare our methods with the methods you use today and assess the claims side by side. While I claim that these are the correct ways to model and measure operational processes, you don't have to take my word for it. Judge for yourself by connecting it to how you are doing this work today.


### The Presence Calculus Project

While the current toolkit is focused on helping practitioners draw contrasts with existing methods, this work is part of the larger research program known as [The Presence Calculus Project](https://docs.pcalc.org), developed over several years within my advisory practice, [The Polaris Advisor Program](https://polarisadvisor.com). The Presence Calculus is a generalization of the methods discussed here, and it is where economic and operational reasoning are brought under the same mathematical umbrella.

Before we can talk about economic reasoning, we need to recast operational flow models so they can integrate with economic models. That is the focus of this document. The current toolkit reinterprets flow analysis using techniques from the Presence Calculus and strictly generalizes conventional flow-metric models. The Presence Calculus itself extends beyond flow analysis to a wider class of operational measurement problems, including economic decision-making.

The generalization is beyond the scope of this document. It is the subject of [The Presence Calculus - A Gentle Introduction](https://docs.pcalc.org/articles/intro-to-presence-calculus/). While it stands alone, those ideas are easier to grasp if you first see how they manifest in familiar arrival-departure flow processes, which are the de facto model for operational flow analysis today.

# Arrival-Departure Processes

All current flow-metric models are based on what we may call arrival-departure processes: discrete _items_ arrive at a system or process boundary and depart after some time. Flow metrics measure key properties of this process: the average time between arrival and departure of individual items over a period (lead time, cycle time, and related variants, depending on boundary definition), arrival and departure rates over the same period (throughput), and the number of items in the system at a point in time (instantaneous WIP) or on average over a period (average WIP).

In general, measuring these quantities accurately requires careful attention to system boundaries, definitions of arrivals and departures, definition of WIP, units of time measurement (for both reported metrics and observation windows), and the exact formulas for the metrics being reported.

If metrics are meant to represent underlying process behavior accurately, these details matter considerably. Current techniques and tools vary widely in rigor. Serious treatments, such as Anderson’s work in the Kanban community and methods introduced by Dan Vacanti, pay much closer attention to these details than many ad hoc implementations that treat these numbers as reporting artifacts.

Even these stronger implementations, however, still face core methodological issues. In many cases, those issues are partially masked because the measurement techniques are paired with highly prescriptive methodologies that mitigate their impact. This makes the approaches fragile when used outside those methodological contexts, as they often are, and less suitable as general-purpose flow analysis techniques. This is particularly true when analyzing processes _before_ those methodological changes are adopted and comparing them to the state after adoption — precisely when measuring the impact of process changes matters most.

Our methods aim to provide process-agnostic flow metrics derived from a formal definition of an arrival-departure process, robust under _any_ realization that conforms to that model, even when underlying processes operate in volatile and changing environments.

In a sense, an arrival/departure process is to flow analysis what a single-celled organism is to biology. Simple enough to exhibit all the core principles involved, and rich enough to develop the full analytical machinery. We begin here because every structural property we need to study more complex configurations of these processes generalizes from this case. Understanding them well is the foundation of everything that follows.

## Flow Process Model for Sample Path Analysis

The process model we use for sample path analysis is a bit different from the standard arrival-departure model above. It is strictly more general and has fewer assumptions, but the differences are also more fundamental than that.

Formally, the domain of analysis consists of processes described by a set of _events_ that denote beginnings and endings we can observe in time. We continue to call these arrival and departure events to preserve continuity with existing practice, but the key difference is that we analyze the process primarily through the event definitions themselves.

![A Flow Process]($document-root/assets/beginnings-endings.png){#fig:begin-end}

In particular, we do not require that the events in [@fig:begin-end] are associated with well-defined items, or that structural boundaries are fixed in advance. The only requirement is that, by observation, we can determine whether an event denotes a beginning or an ending. The primary unit of analysis in this ontology is the domain event. We also do not require explicit correspondence between arrivals and departures, for example by matching them through an item identifier.

This may be surprising, because you may wonder how we can measure concepts such as lead time, cycle time, WIP, and throughput if we cannot identify items. That is precisely a clue that the generalization here runs deeper than simply changing the names of what we call events.

To make this more intuitive, consider a record of births and deaths in a population. We can treat these as arrivals and departures. We are implicitly talking about people being born and dying, and we can measure population in units of people and lifespans in units of time, without needing explicit correspondence between a specific birth and a specific death, or a single integrated record of both.

This reveals something important about flow and flow metrics: these are gestalt properties of a _process_, not simply properties derived by aggregating the experience of individual items traversing that process. Many of the most useful aspects of reasoning about flow for process improvement do not require that level of detail. We can go a long way without ever talking about items [^-items].

[^-items]: That said, there are implicit conservation assumptions. The elements that arrive are the ones that depart; we do not create departures from nothing, nor do elements disappear without departure if they have arrived. The underlying conservation laws are what determine what we perceive as flow, but they can be stated without requiring an explicit pairing between individual arrivals and departures. To avoid item-centric connotations, we use the term _elements_ to denote the units of a countable, conserved set that provides the basis for counting beginning and ending _events_. Quantities like WIP and throughput are defined from the resulting _counting processes_, not from any particular identity or pairing of elements, provided the conservation assumptions are satisfied.

In fact, this is a major source of mismatches in how flow is modeled and measured today. Most current methods measure flow in terms of statistical aggregates of item behavior: distributions of lead times, throughput, etc. We spend a great deal of effort discussing item size, granularity, and similarity. None of this is essential for defining the core structural properties that drive process behavior.

One key insight from sample path analysis is that the quantities needed to analyze the structural behavior of a flow process are determined entirely by the events. Once arrival and departure _processes_ are defined, the core flow relationships follow from them, without requiring detailed knowledge of individual items. This not only clarifies what must be measured to reason about flow (event structure), it also provides a more general framework that applies even in domains where the notion of discrete items is less tangible.

## What Is a Sample Path?

The term comes from probability theory. A non-deterministic process is one that can produce different outcomes under the same conditions. When this non-determinism is modeled using probability distributions, we call the process stochastic.

Consider a coin toss repeated at times $t_0, t_1, t_2, \dots$ as in [@fig:coin-toss].  
A single infinite sequence of outcomes

$$H, T, H, H, T, \dots$$

is one **sample path** of the process. It is one realized history. If we restart the experiment, we may obtain a different sequence.

- A deterministic process has exactly one sample path.  
- A non-deterministic (stochastic) process has many possible sample paths.

There are two complementary views of the same process, as shown in [@fig:coin-toss].

![Sample Paths and Random Variables]($document-root/assets/coin-toss.png){#fig:coin-toss}

- **Along a row (fixed $\omega$):** a single realized sequence evolving over time. This is a sample path.
- **Down a column (fixed $t$):** outcomes across all possible sequences at one point in time. Each column defines a random variable $X(t)$.

Probability theory studies properties that hold across _all_ sample paths via the random variables $X(t)$.  
For example: “The expected value of $X(t)$ is 0.5,” across all sample paths, if we encode Heads as 1 and Tails as 0.

Sample path analysis studies what is true within a single realized path.  
For example: “In this realized sequence up to time $T$, the cumulative number of heads is exactly 37.”

The latter statement is merely an empirical fact. By itself, it is not analytically interesting. The power of sample path analysis emerges when we study a class of processes and prove non-trivial structural properties that must hold along _any_ sample path that satisfies certain observable conditions. Arrival-Departure processes are an example of such a class.

By definition, sample path analysis requires no distributional or probabilistic assumptions. Its statements are conditioned on a single realized history. This makes it especially useful in operational settings, where we observe only one trajectory and cannot fully characterize the underlying sources of non-determinism outside that observed history.

If additional modeling assumptions are available — for example, a known a priori probability distribution — those can be layered on top. But they are not required.

So the two approaches are complementary. Sample path analysis works under more limited informational constraints and therefore relies on more elementary, but structurally robust, techniques.

This is the perspective we exploit: structural properties of flow processes can be proven to hold along any sample path that satisfies verifiable conditions, without committing to probabilistic assumptions about the ensemble properties that hold across all sample paths.
Since we generally don't have a probability distribution to work with, we will call the underlying process non-deterministic rather than stochastic.

## The Sample Path of a Flow Process

We begin by specifying how non-determinism enters the model. Imagine we are observing arrivals and departures over time.
We record:

- The **timestamp** of the event.  
- The **type** of event (arrival or departure).

We can think of this as a non-deterministic process on two random variables: the event type, which is analogous to a coin toss, and
the elapsed time _between_ events. The second random variable is recoverable from the timestamps and is a much richer object than the arrival/departure binary. Everything we think of as flow can be described in terms of the interactions of these two random variables.

Mathematically, this object is a **marked point process**: a sequence of timestamps, each carrying a mark. The minimal mark set here is {arrival, departure}. The timestamps determine the elapsed times between events. We may think of such a marked point process as a single sample path of an arrival/departure process whose broader non-deterministic structure may be unknown.

[@fig:mpp] shows an example of an arrival/departure marked point process. This realized event history is the _input_ to sample path analysis.

![Arrival/Departure Marked Point Process]($document-root/assets/arrival-departure-mpp.png){#fig:mpp}

Any observed history of a non-deterministic process is a common finite prefix of some infinite set of sample paths. Once we fix that prefix, any further non-determinism lies in those possible futures[^-non-determinism]. Once a finite prefix has been observed, every quantity we compute from it is determined by that prefix.

[^-non-determinism]: There is an implicit qualifier in this claim: we are assuming that the non-determinism we care about in our model is confined to the random variables under analysis. There are, of course, many other potential sources of non-determinism even in a single arrival/departure process, including measurement errors.


In the case of the coin toss process, this can be visualized as in [@fig:prefix].

![Sample Paths Prefixes]($document-root/assets/sample-path-prefix.png){#fig:prefix}

In the case of an arrival/departure process, we will show that if a finite prefix of some sample path is observed, all remaining flow metrics and empirical distributions derive from simple deterministic functions that measure properties of that observed path, much like the statement that the number of heads observed on the coin-toss sample path is 37.

Since the observed sample path in this case is a marked point process, this implies that the structure of flow is encoded directly in the realized event history. If the event history is captured accurately, the quantities we associate with flow can be read off mechanically from this history. In this context, non-determinism always lives in the future, and there is no need to invoke probabilistic or statistical assumptions or language when reasoning about *observed* flow. Specifically, the process of _measuring_ flow does not introduce any non-determinism beyond what is already encoded in the observed sample path [^-sampling].

[^-sampling]: Ensuring this means paying careful attention to the impact of techniques such as sampling on flow measurement. This is a key point of departure in our methods as well.

The machinery that makes this possible is what we call sample path analysis.

## Where Distributions Matter

Probability and statistics still matter, especially when we are reasoning about possible futures in prediction models. But they are not the starting point.

In current industry practice, flow measurement and analysis focus on producing item-level empirical distributions of metrics such as lead time and throughput. These are useful. They help quantitatively describe and characterize customer experience, tail risk, service levels, and a whole host of other operationally useful metrics.

However, given the non-deterministic model above, there is no _intrinsic_ non-determinism in those distributions beyond what is already encoded in the sample path of arrivals and departures. They are structurally constrained by the aggregate behavior of the underlying arrival and departure processes. Randomness in item-level flow metrics originates in randomness on the sample paths of the arrival/departure process. In that sense, treating these empirical distributions as first-class probabilistic or statistical constructs is fraught.

The primary quantities that govern how these distributions behave are the more primitive sample-path flow metrics we will develop. These are _structural properties of the process_, not _aggregate properties of item-level behavior_. They do not require item-level identification of arrivals and departures, nor do they require constructing distributions in order to reason about their properties.

If we want item-level distributions, we may extend the marked point process by explicitly pairing arrivals and departures with identifiers. Once that pairing is defined, the entire empirical distribution of item-level process times over a given sample-path prefix is determined by the realized event structure.

This establishes a hierarchy:

1. Non-deterministic event structure.  
2. Structural flow metrics derived from that structure.  
3. Distributional summaries derived from explicit item pairing.

Making this hierarchy explicit distinguishes sample path analysis from current practice and changes how these secondary artifacts, such as empirical distributions, should be interpreted, particularly in forecasting.

This is not an intuitive shift, and some of the language and machinery we develop will be unfamiliar. The payoff is a set of techniques for reasoning about flow that do not depend on distributional assumptions, and whose core results hold unconditionally on every realized sample path.

The remainder of this document, and the supporting material on this site, develops the details.

# Sample Path Analysis

Before we turn to the substance of sample path analysis, there are many new concepts to absorb and integrate, even in a model as simple as the arrival-departure process.

This chapter provides a high-level roadmap of the key ideas and arguments we develop in the remaining chapters, without defining each one in detail. Think of it as a guide to where each concept fits in the overall architecture of flow.

## Computing sample path flow metrics

The previous chapters established that everything in our methods hinges on observing events on the sample path — in the case of flow processes, a marked point process — and analyzing the event structure as observations unfold in time.

A good mental model for how we compute these metrics is as follows [^-mental-model]:

[^-mental-model]: Even though this is not the way the underlying algorithms are necessarily implemented in the toolkit.

- Starting from a fixed point in time, observe the sample path up to time $T$.
- Compute a set of metrics that describe the state of the process up to that point in time. These are finite-window quantities that are closely related to many of the metrics we measure today, but they are not the same quantities, and the differences matter.
- Wait for the next event, record the event type, and extend the window by the elapsed time, giving an extended sample path that includes realized values of the next random element (timestamp and mark)[^-random-variables].
- Recompute all metrics deterministically from their previous values and the incremental contribution of the new variables.

[^-random-variables]: At each step we are observing the realized value of the next random element in some underlying non-deterministic process. Conditioned on the observed prefix, the structural flow quantities defined over that prefix are deterministic functionals of the prefix. This is what we mean when we say "randomness lives in the future."

The fact that every flow metric in this model can be computed this way is not obvious. We have traditionally treated these metrics as statistics, averages and percentiles of distributions. Provably correct causal _attribution_ is what makes sample path analysis powerful. It lets us trace changes in flow metrics back to the _contributions of individual events on the timeline_, and this, in turn, gives us tools to shape the _event structure_ so metrics move in the direction we want. This principle is at the heart of why this technique is worth learning.

## Processes and Indexes

In stochastic process theory, the word *process* has a specific meaning: it is a _function_ that maps an index set to a set of states. Put more plainly, a process is something that has state, and whose state varies along some dimension — most commonly time. The definition is deliberately general, and it captures most things we informally call processes.

In an arrival–departure system, time is naturally one indexing dimension. But what does process state mean in this context? There are many possibilities. Let's consider one: the cumulative arrival count. At any point in time (the index) we record how many arrivals have occurred up to that moment (the state). If we observe that count at fixed reporting intervals — every minute, hour, or day — we obtain a _process_ indexed by calendar time. For the sample path in [@fig:mpp], this gives the processes in [@fig:calendar-indexed].

![Calendar-Indexed Cumulative Arrivals]($document-root/assets/calendar-indexed-arrivals.png){#fig:calendar-indexed}

But this is not the only way to _index_ the cumulative arrival count process. We can instead treat each arrival *event* as a state transition of the arrival-departure process. The cumulative count increases by one at each arrival and remains constant between arrivals. In this view, the arrival-departure process changes state when an arrival event occurs. We can index the cumulative arrival count process by these events, and then the index is no longer an arbitrary reporting schedule; it is the sequence of events that cause the state to change. This process is shown in [@fig:event-indexed].

![Event-Indexed Cumulative Arrivals]($document-root/assets/event-indexed-arrivals.png){#fig:event-indexed}

These two representations describe the states of the same underlying arrival-departure process, but they are not equivalent.

Calendar indexing _samples_ the state of the arrival-departure process at chosen time points. It records values, but not necessarily the exact transition structure that produced them. Event indexing, by contrast, is built from the transitions themselves. It records every state change and the order in which those changes occurred. It preserves the full evolution of the system at the resolution of events.

This is a material difference. From the event-indexed representation of cumulative arrival count, we can always produce a calendar-indexed one for any given reporting frequency. The reverse is not true. Once we observe only sampled or aggregated calendar states, the underlying transition history is no longer recoverable [^-agreement]. The information loss compounds when we derive new quantities by aggregating sampled values.

[^-agreement]: Note, however, that in this example, the two representations agree on the _values_ reported at the same time indexes. The information loss is not due to sampling error, but lies in recoverability of the underlying dynamics. The calendar-indexed view cannot say anything about what happens to the process in _between_ samples. The event-indexed view can, unambiguously. However, if we derive processes over these sampled values, then sampling errors do compound meaningfully.

Calendar indexing is the default way of reporting flow metrics today. Flow metric calculations today *begin* by sampling the input for some reporting period and then performing various aggregations over the sampled values. In doing this we lose the structural connection between metric changes and the events that generated them. Metrics plotted over time become trend lines without a direct link back to the events that generate those trends. The choice of straight lines connecting the points in [@fig:calendar-indexed] is not directly defensible, nor is any other interpolation for that matter. They become reporting artifacts.

By contrast, event indexing treats each flow metric _as a dynamic process in its own right_. The particular dynamics — how a metric behaves at events and between events — is part of the unique signature of that metric, and we can use it to reason about the metric's evolution over time. The metrics are _themselves_ deterministic processes _over a sample path from a non-deterministic process_ and the plots are sample paths of these processes. The key difference in [@fig:event-indexed] is that the shape of the sample path — in this case, a step chart that changes direction precisely at the event timestamps — follows from the definition of the metric. In [@fig:calendar-indexed], no such inference can be made.

With event indexing, every change in a derived flow metric can be tied to the specific state transition that caused it, up to event resolution. This is important if we want to understand *why* a specific change happened. Of course, this requires that the metric itself be defined so its dynamics can be described in this way. It is somewhat obvious with cumulative arrivals, but as we will show, we can do this for each one of the sample path metrics we develop, even when it is not so obvious. In fact, it is this explicit derivation of the dynamics for standard flow metrics that is the specific contribution of The Presence Calculus to flow analysis.

Specifically, the evolution of these sample paths follows predictable rules: events _cause_ the trajectory of the path to change in predictable ways and _in between events_ we can state how the path behaves deterministically. The only unpredictable element is what that next event is, and when it will change the trajectory of the path. As we will see, this is a much more tractable setup to reason about flow.

El-Taha & Stidham [@eltaha1999] use the term _processes with imbedded point processes_ to describe this class of mathematical objects. We will use the friendlier term _event-indexed processes_ to describe the same class instead. Each one of our sample path flow metrics is a deterministic _event-indexed process_ over a sample path of a non-deterministic process.

Viewed this way, each metric has its own unique dynamics that _interact_ when metrics _are composed as mathematical functions_. These interactions can also be derived deterministically. So both the mathematical definition of a process as a function over sample paths, and its dynamics play important roles in our ability to reason about flow using sample path flow metrics.

Unless otherwise stated, event-indexed processes are the default throughout. Calendar-indexed views will be introduced later, when we turn to reporting and aggregation over _event-indexed metrics_. The next section formalizes the shift from statistical inference over distributions to the study of deterministic dynamics along realized sample paths.


## Flow Dynamics and Flow Geometry

A dynamic model describes how a flow process evolves over time. It specifies the causal mechanisms: how arrivals, departures, and other exogenous inputs change process state, and how those changes update derived quantities along the sample path.

By contrast, process *geometry* describes the structural constraints on that evolution. Geometry does not tell us what will happen next or why; it tells us what *must* be true, regardless of how the process is driven. It encodes conservation laws, invariants, and deterministic relationships that bind the derived processes together.

When we ask why a process is behaving a certain way, we need both perspectives. The dynamic model explains how particular inputs produce specific state transitions and feedback effects. The geometry explains how those transitions must propagate to maintain internal consistency across the state of the process and its derived quantities.

Dynamics governs causal chains and loops. Geometry governs conservation and constraint. Together they determine the admissible trajectories of the process on its sample path.

In the case of arrival–departure processes, we develop the dynamic model by extending the cumulative arrival and departure counts into a richer representation of process state. The finite form of Little’s Law — expressed as the *Presence Invariant* — provides the geometric structure. It defines the global constraint that binds counts, rates, and derived quantities into the relationships we need to reason about flow.

Taken together, these give us a fully deterministic measurement substrate for reasoning about flow. Conditioned on a realized sample path, we can answer unambiguously: how did the process evolve to reach its current state? That, in turn, is the foundation we need to reason about the consequences of the process being in that state. This includes economic consequences, but is not limited to them.

We begin with a bird's-eye view of the moving parts, without full technical detail, so that the overall arc is visible before we examine each piece. The metrics reference document develops each concept in full.

### Flow Dynamics
We've already seen the core concepts involved in modeling metrics as dynamic processes in the cumulative arrival count $A(T)$ metric. We extend this idea to cumulative departure counts $D(T)$ and then derive a number of processes from there, each of which captures a higher-order notion of the "state" of the arrival-departure process.

The chain of processes that model both the short-run and long-run dynamics of an arrival-departure process is shown in [@fig:flow-dynamics] below.[^-differential-equations]

[^-differential-equations]: The formal way to describe such models is as a system of difference or differential equations, and this is possible here as well. But we are more interested in explaining the underlying concepts intuitively, so we will opt for plain English here. The metrics reference has a more technical treatment with precise mathematical definitions of the concepts involved.

![The Dynamics Model]($document-root/assets/flow-dynamics.png){#fig:flow-dynamics}

In [@fig:flow-dynamics] each oval represents a flow metric that models a specific aspect of the observed dynamics of the underlying arrival-departure process. Each metric depends on one or more previous metrics, and time plays a crucial role throughout. Changes in every metric are traceable back to the events on the sample path.

Let's go through the individual metrics briefly in order, starting with cumulative arrival count and cumulative departure count — the ones that are directly calculated from the sample path.

- **Cumulative Arrival Count — $A(T)$**: We've already seen this one. It counts the number of arrivals observed in a time interval $T$.

  *Dynamics*: $A(T)$ increases by 1 with every arrival and remains unchanged on departure events, and in between events.

- **Cumulative Departure Count — $D(T)$**: The departure process counterpart.

    *Dynamics*: It increases by 1 with every departure and remains unchanged otherwise.

- **Instantaneous Presence — $N(t) = A(T) - D(T)$**: This metric measures _imbalance_ between cumulative arrival and departure counts at an instant. We call this the instantaneous presence.[^-presence]

    Since $A(T)$ and $D(T)$ represent cumulative states of the arrival-departure process, $N(t)$ also encodes a process state — a higher-order state representing the imbalance between the two cumulative counts. Think of $N(t)$ as instantaneous WIP as you connect it to the familiar flow metrics.[^-wip]

     *Dynamics*: The value of $N(t)$ increases by 1 with every arrival, decreases by 1 with every departure, and remains constant in between.

[^-presence]: In general, presence is a quantity that represents the flow of some measurable quantity, and here we are measuring its instantaneous value. The Presence Calculus allows us to generalize this simple notion to much more general mathematical settings. The arrival-departure count imbalance is one of the simplest notions of presence we can establish for an arrival-departure process. See [The Presence Calculus, A Gentle Introduction](https://docs.pcalc.org/articles/intro-to-presence-calculus/) for more general definitions of Presence and many more examples.
[^-wip]: The reason we don't define it as such is that WIP is a specific *interpretation* that applies to specific domains. A more general concept here might be occupancy, but even this requires specific assumptions that are not necessary to reason about flow, so we will stick with the least restrictive definition of $N(t)$ as imbalance. Further, both WIP and occupancy are a type of presence, but not all presence is of this type. That is the key thing to remember.

- **Cumulative Presence Mass — $H(T)=\int_0^T N(t)\,dt$**: This is accumulated presence over the interval $(0,T]$ (the area under $N(t)$, equivalently the area between $A(T)$ and $D(T)$). It is the key integrated quantity that carries process history in element-time units.

  *Dynamics*: $H(T)$ does not jump at arrivals or departures. Arrivals/departures change $N(t)$, which changes the **slope** of $H(T)$. Between events, $H(T)$ is linear with slope equal to the current $N(t)$ (flat when $N(t)=0$).

- **Time-Average Presence — $L(T)=H(T)/T$**: This is the time-average of presence over $(0,T]$, i.e. the moving average of $N(t)$ over the observed prefix. It is the left-hand side quantity in the Presence Invariant.

  *Dynamics*: $L(T)$ is continuous at event times (no jumps). Between events it adjusts toward the current state: it rises when $N(t)>L(T)$ and falls when $N(t)<L(T)$. Its responsiveness decays over time (roughly at rate $1/T$), so transient fluctuations are smoothed while persistent effects remain visible.

In the table below, we summarize the dynamics model.

| Metric | Derivation Formula | On arrival | On departure | Between events                            |
|---|---|------------|--------------|-------------------------------------------|
| $A(T)$ | $A(T)=\sum \text{ arrivals in }(0,T]$ | +1         | Unchanged    | Constant.                                 |
| $D(T)$ | $D(T)=\sum \text{ departures in }(0,T]$ | Unchanged  | +1           | Constant.                                 |
| $N(T)$ | $N(T)=A(T)-D(T)$ | +1         | −1           | Constant.                                 |
| $H(T)$ | $H(T)=\int_0^T N(t)\,dt$ | Unchanged  | Unchanged    | Increases linearly, with slope $N$.       |
| $L(T)$ | $L(T)=H(T)/T$ | Unchanged  | Unchanged    | Seeks $N$: rises if $N>L$, falls if $N<L$. |

We can think of this chain as encoding memory about process behavior across different timescales. Events are instantaneous and discrete and carry no memory. Cumulative counts remember how many events have occurred, but not what happened between them. From an observers perspective, cumulative arrival and departure counts evolve independently: the arrival departure model has no intrinsic way to connect process state across the two processes.   _Presence_ is the quantity that models the interaction between the two. Instantaneous presence models the instantaneous imbalance between arrival and departure counts as a process state that evolves over time. At this point we have the basic machinery we need to talk about state transitions of the arrival-departure process as a whole.

In the underlying arrival departure model, the time between events determines the _how long_ the process remains in a state. At each event, arrival or departure, the process changes state and in between events, it stays in the same state. Cumulative presence encodes time-weighted presence in a given state - this is what the integral in $H(T)$ does. If we observe an arrival-departure process for a finite amount of time, the value of $H(T)$ as $T$ evolves can be interpreted as a continuous state variable for the process, one that encodes the entire history of the process as observed so far. We can think of this as the global state of the process at any moment in time.

The rules by which each of these processes evolve are completely determined by the dynamics of the underlying arrival-departure process, and these in turn are specified by the event type and time between events in the observed sample paths. So we now have a clean causal chain that explains precisely how the global state of the process evolves from its instantaneous states. As we will see shortly, $H(T)$ and it's relationship to $N(t)$ via integration,  encodes everything we need to reason about flow in the arrival-departure process. In a very real sense, these are the fundamental flow metrics on which all our existing flow metrics depend upon. Everything we normally think of and measure as flow metrics: throughput, process time, occupancy metrics, costs etc all can be derived deterministically given $H(T)$ and the chain of processes that lead to it.

You'll notice we have conspicuously avoided including $L(T)$ in this chain, even though we have included it in the dynamic model and provided rules for its evolution. This is intentional. $L(T)$ plays a different role in the model compared to its inputs. We can think of the chain of processes that lead to $L(T)$ as unnormalized metrics that are measured and monitored on a real timescale.

$L(T)$ and the remaining metrics we will show are normalized by the length observation window, giving us a basis to reason about relative changes in the observed behavior over time. $L(T)$ in particular, is a half-open moving average of instantaneous presence: the left endpoin is fixed and the right endpoint varies continuously. This allows us to distinguish between the significance of transient presence (process states held for short periods of time) and stable presence (states that persist or the process returns to repeatedly over its history). It also lets us identify stable states, and detect convergence and divergence of the instantaneous presence from these stable states. As we derive in the metrics reference, the relationship between $N(t)$ and $L(T) is precisely the relationship that measures the sensitivity of a moving average with the underlying value it is tracking. This is the entire basis of why $L(T)$ behaves as it does.

So why did we include this in the dynamics model? Its because this behavior of the moving average introduces time averaging as a distnct new causal mechanism that drives the behavior of the remaining flow metrics, which are also time nomalized metrics in every sense. These relationships are _constrained_ by the finite version of Little's Law, which we call the Presence Invariant for reasons that will become clear shortly. Throughput and process time (in its various forms) are essentially deterministic accounting identities
Given L(T) as derived above, the invariant constrains how throughput and process time must adjust in order to conserve the total value of cumulative presence. We call this the principle the conservation of cumulative presence. Arrival rates, throughputs and process times can be viewed as ways to factor a give cumulative presence value given $L(T)$, The Presence Invariant is the constraint that governs admissible factorizations such that the cumulative presence is conserved. The invariant has a very clear geometric interpretation that makes these ideas crystal clear.

So to recap, $L(T)$ is the bridge between flow dynamics and flow geometry. The numerator of $L(T)$ is driven by arrival-departure events. The denominator brings in the effects of time normalization. These are the causal drivers of the dynamics of all the remaining metrics we will derive. The dynamics of those metrics are simply a result of the dynamics of $L(T)$ intersecting with the Presence Invariant, which in turn encodes the principle of conservation of presence.










# References
