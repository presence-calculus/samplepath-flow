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

For example, it is widely accepted that throughput and process-time metrics such as lead time and cycle time are important performance measures in a delivery context. But in adapting them to software development, we often do not account for the economic context in which we measure them. Throughput alone does not translate into meaningful economic outcomes. Process time in its various forms, and cost of delay, are often more critical, yet we still have few useful ways to connect those quantities to operational decision-making [^-wsjf].

[^-wsjf]: WSJF job scheduling is, of course, one of the "standard" answers here, but it is a relatively crude heuristic and only a minor application of a deeper and more nuanced topic.

Further, Little's Law shows that these concepts are not independent. It implies structural relationships between process time and throughput that manifest as economic trade-offs, constraining the courses of action available as we pursue specific outcomes.

While Little's Law is widely name-checked to justify common practices, the throughlines between those practices and the underlying theory often rest on assumptions derived from their original application in manufacturing. There are software-development contexts where these assumptions are valid. For example, if you build and deliver software on a fixed-price contract basis, traditional economic models of throughput and resource utilization are often the right ones, and we can get direct leverage by adopting those ideas, as many careful practitioners do.

But if you are selling a SaaS product, product-development throughput is not nearly as important as careful selection of what goes into the product, understanding cost of delay, and optimizing delivery processes to stay responsive to shifting markets. Market performance depends on much more than delivering the right product. Time to market, network effects in the ecosystem in which the product operates, and related factors play a much larger role. So if we are focusing on product development in this context, we need tools that can incorporate these notions into our modeling assumptions _before_ we design operational processes.

Today, we mostly live in an awkward middle ground: we allude to these larger concerns when talking about process improvement, but our prescriptions and methodologies largely remain shackled to a limited set of generic ideas borrowed from industrial production: reduce WIP, remove waste, small batches, faster feedback, and so on. They are treated as universal solutions, applicable without business context, with no reliable way to connect how and why adopting any of them improves business outcomes, and no systematic way to measure and prove they work in a given context.

Our first objective in this document is to shore up those foundations. Doing so reveals where current practices fit within the broader mathematical context provided by a rigorous theory of flow processes, helps us understand where and when certain approaches are appropriate, and whether a particular improvement method is likely to be effective before implementation.

Such claims require proof: first, that a real limitation exists in prevailing methods; second, that the proposed alternatives resolve it. This document lays out that case at a high level. The broader project and documentation provide the mathematical foundation, derivations, and open-source tooling needed to validate these claims directly on your own data.

None of the core mathematical ideas are new. Many were established decades ago in queueing theory[^-queueing-theory]. In particular, the sample path analysis techniques underlying this work trace to Shaler Stidham’s 1972 deterministic proof of [Little’s Law](https://docs.pcalc.org/articles/littles-law/) and are mainstream complements to statistical and probabilistic analyses of stochastic processes, which is exactly how they are used here. Little’s Law is foundational to flow analysis, though in software delivery it is often invoked loosely, without exploiting its full structural implications. This is another key point of departure for our methods.

[^-queueing-theory]: Much of traditional queueing theory cannot be applied as-is in the environments where software development happens, yet those distinctions are often poorly understood. As we will see, several more foundational ideas need to be established before we tackle queueing directly. Once those are in place, a rigorous foundation for managing operational queues emerges naturally.

The theoretical foundation for our methods, which draws clear distinctions between concepts such as flow analysis, queue management, and economic analysis, is presented in *Sample Path Analysis of Queueing Systems* by El-Taha and Stidham [@eltaha1999]. Our contribution is practical: translating this theory into operational tools that can be applied directly to real software-delivery systems, and simplifying terminology where needed so it maps more clearly to the domain without losing rigor.

Applying these ideas requires conceptual shifts if you are very familiar with current methods. This document introduces those shifts and points to the theory and tooling needed to verify each claim. While the underlying mathematics is elementary, the perspective shift is significant and may be disorienting if you are comfortable with current techniques.

To make it easier to navigate these ideas, our toolkit is designed to work with _exactly_ the same inputs you use to measure and analyze flow today. The results will differ, but we will explain why each difference matters. Rather than read this as an abstract theoretical treatise, I encourage you to compare our methods with the methods you use today and assess the claims side by side. While I claim that these are the correct ways to model and measure operational processes, you don't have to take my word for it. Judge for yourself by connecting it to how you are doing this work today.


### The Presence Calculus Project

While the current toolkit is focused on helping practitioners draw contrasts with existing methods, this work is part of the larger research program known as [The Presence Calculus Project](https://docs.pcalc.org), developed over several years within my advisory practice, [The Polaris Advisor Program](https://polarisadvisor.com). The Presence Calculus is a generalization of the methods discussed here, and it is where economic and operational reasoning are brought under the same mathematical umbrella.

Before we can talk about economic reasoning, we need to recast operational flow models so they can integrate with economic models. That is the focus of this document. The current toolkit reinterprets flow analysis using techniques from the Presence Calculus and strictly generalizes conventional flow-metric models. The Presence Calculus itself extends beyond flow analysis to a wider class of operational measurement problems, including economic decision-making.

The generalization is beyond the scope of this document. It is the subject of [The Presence Calculus - A Gentle Introduction](https://docs.pcalc.org/articles/intro-to-presence-calculus/). While it stands alone, those ideas are easier to grasp if you first see how they manifest in familiar arrival-departure flow processes, which are the de facto model for operational flow analysis today.

# The Process Models

We will pay close attention to model assumptions as we develop the case. Clarifying hidden assumptions is central to understanding the strengths and weaknesses of any analytical technique. Many arguments here hinge on exposing those assumptions and mapping them to the contexts in which these techniques are applied today.


## Arrival-Departure Processes

All current flow-metric models are based on what we may call arrival-departure processes: discrete _items_ arrive at a system or process boundary and depart after some time. Flow metrics measure key properties of this process: the average time between arrival and departure of individual items over a period (lead time, cycle time, and related variants, depending on boundary definition), arrival and departure rates over the same period (throughput), and the number of items in the system at a point in time (instantaneous WIP) or on average over a period (average WIP).

In general, measuring these quantities accurately requires careful attention to system boundaries, definitions of arrivals and departures, definition of WIP, units of time measurement (for both reported metrics and observation windows), and the exact formulas for the metrics being reported.

If metrics are meant to represent underlying process behavior accurately, these details matter considerably. Current techniques and tools vary widely in rigor. Serious treatments, such as Anderson’s work in the Kanban community and methods introduced by Dan Vacanti, pay much closer attention to these details than many ad hoc implementations that treat these numbers as reporting artifacts.

Even these stronger implementations, however, still face core methodological issues. In many cases, those issues are partially masked because the measurement techniques are paired with highly prescriptive methodologies that mitigate their impact. This makes the approaches fragile when used outside those methodological contexts, as they often are, and less suitable as general-purpose flow analysis techniques. This is particularly true when analyzing processes _before_ those methodological changes are adopted and comparing them to the state after adoption — precisely when measuring the impact of process changes matters most.

Our methods aim to provide process-agnostic flow metrics derived from a formal definition of an arrival-departure process, robust under _any_ realization that conforms to that model, even when underlying processes operate in volatile and changing environments.

So what is that model, precisely? This is where fundamental differences begin to emerge.

## Flow Process Model

The process model we use for sample path analysis is subtly different from the standard arrival-departure model above. It is strictly more general and has fewer assumptions, but the differences are also more fundamental than that.

Formally, the domain of analysis consists of processes described by a set of _events_ that denote beginnings and endings we can observe in time. We continue to call these arrival and departure events to preserve continuity with existing practice, but the key difference is that we analyze the process primarily through the event definitions themselves.

![A Flow Process]($document-root/assets/beginnings-endings.png){#fig:begin-end}

In particular, we do not require that the events in [@fig:begin-end] are associated with well-defined items, or that structural boundaries are fixed in advance. The only requirement is that, by observation, we can determine whether an event denotes a beginning or an ending. The primary unit of analysis in this ontology is the domain event. We also do not require explicit correspondence between arrivals and departures, for example by matching them through an item identifier.

This may be surprising, because you may wonder how we can measure concepts such as lead time, cycle time, WIP, and throughput if we cannot identify items. That is precisely a clue that the generalization here runs deeper than simply changing the names of what we call events.

To make this more intuitive, consider a record of births and deaths in a population. We can treat these as arrivals and departures. We are implicitly talking about people being born and dying, and we can measure population in units of people and lifespans in units of time, without needing explicit correspondence between a specific birth and a specific death, or a single integrated record of both.

This reveals something important about flow and flow metrics: these are gestalt properties of a process, not necessarily properties derived by aggregating the experience of individual items traversing that process. Many of the most useful aspects of reasoning about flow for process improvement do not require that level of detail. We can go a long way without ever talking about items [^-items].

[^-items]: That said, there are implicit conservation assumptions. The elements that arrive are the ones that depart; we do not create departures from nothing, nor do elements disappear without departure if they have arrived. The underlying conservation laws are what determine what we perceive as flow, but they can be stated without requiring an explicit pairing between individual arrivals and departures. To avoid item-centric connotations, we use the term _elements_ to denote the units of a countable, conserved set that provides the basis for counting beginning and ending _events_. Quantities like WIP and throughput are defined from the resulting _counting processes_, not from any particular identity or pairing of elements, provided the conservation assumptions are satisfied.

In fact, this is a major source of mismatches in how flow is modeled and measured today. Most current methods measure flow in terms of statistical aggregates of item behavior: distributions of lead times, throughput, etc. We spend a great deal of effort discussing item size, granularity, and similarity. None of this is essential for defining the core structural properties that drive process behavior.

One key insight from sample path analysis is that the quantities needed to analyze the structural behavior of a flow process are determined entirely by the events. Once arrival and departure _processes_ are defined, the core flow relationships follow from them, without requiring detailed knowledge of individual items. This not only clarifies what must be measured to reason about flow (event structure), it also provides a more general framework that applies even in domains where the notion of discrete items is less tangible.

## What Is a Sample Path?

The term comes from probability theory. A non-deterministic process is one that can produce different outcomes under the same conditions. When this non-determinism is modeled using probability distributions, we call the process stochastic.

Consider a coin toss repeated at times $t_0, t_1, t_2, \dots$ as in [@fig:coin-toss].  
A single infinite sequence of outcomes

$$H, T, H, H, T, \dots$$

is one **sample path** of the process. It is one realized history. If we restart the experiment, we may obtain a different sequence.

A deterministic process has exactly one sample path.  
A non-deterministic (stochastic) process has many possible sample paths.

These are two complementary views of the same process, as shown in [@fig:coin-toss].

![Sample Paths and Random Variables]($document-root/assets/coin-toss.png){#fig:coin-toss}

- **Along a row (fixed $\omega$):** a single realized sequence evolving over time. This is a sample path.
- **Down a column (fixed $t$):** outcomes across all possible sequences at one time. Each column defines a random variable $X(t)$.

Probability theory studies properties that hold across all sample paths via the random variables $X(t)$.  
For example: “The expected value of $X(t)$ is 0.5,” across all sample paths, if we encode Heads as 1 and Tails as 0.

Sample path analysis studies what is true within a single realized path.  
For example: “In this realized sequence up to time $T$, the cumulative number of heads is exactly 37.”

The latter statement is merely an empirical fact. By itself, it is not analytically interesting. The power of sample path analysis emerges when we study a class of processes and prove structural properties that must hold along _any_ sample path that satisfies certain observable conditions.

Sample path analysis requires no distributional or probabilistic assumptions. Its statements are conditioned on a single realized history. This makes it especially useful in operational settings, where we observe only one trajectory and cannot fully characterize the underlying sources of non-determinism.

If additional modeling assumptions are available — for example, a known probability distribution over sample paths — those can be layered on top. But they are not required.

The two approaches are complementary. Sample path analysis works under stricter informational constraints and therefore relies on more elementary, but structurally robust, techniques.

This is the perspective we exploit: structural properties of flow processes can be proven to hold along any sample path that satisfies verifiable conditions, without committing to probabilistic assumptions about the ensemble properties that hold across all sample paths.


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

In the case of an arrival/departure process, we will show that if a finite segment of the sample path, a marked point process, is observed, all remaining flow metrics and empirical distributions derive from simple deterministic functions that measure properties of the observed sample path, much like the statement that the number of heads observed on the coin-toss sample path is 37.

In other words, the structure of flow is encoded directly in the realized event history. If the event history is captured accurately, the quantities we associate with flow can be read off mechanically from this history. In this context, randomness always lives in the future, and there is no need to invoke probabilistic or statistical assumptions or language when reasoning about *observed* flow.

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

We now turn to the substance of sample path analysis. There are many new concepts to absorb and integrate, even in a model as simple as the arrival-departure process.

This chapter provides a high-level roadmap of the key ideas and arguments we develop in the remaining chapters, without defining each one in detail. Think of it as a guide to where each concept fits in the overall architecture of flow.

## The High Level Arc

The previous chapters established that everything in our methods hinges on observing events on the sample path — in the case of flow processes, a marked point process — and analyzing the event structure as observations unfold in time. A good mental model for this is as follows:

- Starting from a fixed point in time, observe the sample path up to time $T$.
- Compute a set of metrics that describe the state of the process up to that point in time. These are finite-window quantities that are closely related to many of the metrics we measure today, but they are not the same quantities, and the differences matter.
- Wait for the next event, record the event type, and extend the window by the elapsed time, giving an extended sample path that includes that event.
- Recompute all metrics deterministically from their previous values and the incremental contribution of the new event.

The fact that we can compute every flow metric this way is not obvious. We have traditionally treated these metrics as statistics, averages and percentiles of distributions. Causal attribution is what makes sample path analysis powerful. It lets us trace changes in flow metrics back to the _contributions of individual events on the timeline_, and this, in turn, gives us tools to shape the _event structure_ so metrics move in the direction we want. This principle is at the heart of why this technique is worth learning.












# References
