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

Flow metrics and flow-analysis techniques have been used in software for nearly 20 years. This lineage runs from the Poppendiecks [@poppendieck2003; @poppendieck2006], who introduced lead time, throughput, and work in progress (WIP) to Lean software development, to David Anderson’s popularization of the Kanban method [@anderson2010] as an empirical management approach for knowledge work. As Value Stream Management matured and flow-based improvement became embedded in DevOps and product organizations, a broader toolkit of metrics and analytical methods, including those popularized by Dan Vacanti [@vacanti2015], was adopted for operational management and forecasting.

Measuring and managing flow is now baseline practice across much of the software industry. It may be surprising, then, that I argue we need substantially different ways to measure, model, and reason about flow if the goal is systematic _process improvement_.

As AI-augmented engineering reshapes how work is performed, the methods we use to reason about flow must remain robust under structural change. This is a real limitation in how we measure processes today.

Most measurement techniques in use today rely implicitly on strict process assumptions for accuracy and coherence. In a period of disruption, volatility, and structural change in software delivery, those assumptions will become harder to defend.

The techniques shown here use _exactly_ the same inputs you already use to analyze flow. The difference is _what_ we measure and _how_ we measure it. The answers are often different from those produced by current methods. The claim is that these are the mathematically correct answers in the more general settings we now operate in.

Such claims require proof: first, that a real limitation exists in prevailing methods; second, that the proposed alternatives resolve it. This document lays out that case at a high level. The broader project and documentation provide the mathematical foundation, derivations, and open-source tooling needed to validate these claims directly on your own data.

None of the core mathematical ideas are new. Many were established decades ago in queueing theory. In particular, the sample path analysis techniques underlying this work trace to Shaler Stidham’s 1972 deterministic proof of [Little’s Law](https://docs.pcalc.org/articles/littles-law/) and are mainstream complements to statistical and probabilistic analyses of stochastic processes, which is exactly how they are used here. Little’s Law is foundational to flow analysis, though in software delivery it is often invoked loosely, without exploiting its full structural implications. This is another key point of departure for our methods.

The theoretical foundation for our methods is presented in *Sample Path Analysis of Queueing Systems* by El-Taha and Stidham [@eltaha1999]. Our contribution is practical: translating this theory into operational tools that can be applied directly to real software-delivery systems, and simplifying terminology where needed so it maps more clearly to the domain without losing rigor.

Applying these ideas requires conceptual shifts if you are very familiar with current methods. This document introduces those shifts and points to the theory and tooling needed to verify each claim. While the underlying mathematics is elementary, the perspective shift is significant and may be disorienting if you are comfortable with current techniques.

This work is part of the larger research program known as [The Presence Calculus Project](https://docs.pcalc.org), developed over several years within my advisory practice, [The Polaris Advisor Program](https://polarisadvisor.com). The current toolkit reinterprets flow analysis using techniques from the Presence Calculus and strictly generalizes conventional flow-metric models. The Presence Calculus itself extends beyond flow analysis to a wider class of operational measurement problems.

We begin with the simpler and well-understood case of arrival-departure flow processes, which all current flow models build on, to expose key concepts in a familiar setting. These concepts generalize beyond the arrival-departure case while keeping the modeling and measurement techniques analytically tractable. That is what makes these ideas powerful, beyond simply being a better way to measure flow metrics.

The generalization is beyond the scope of this document. It is the subject of [The Presence Calculus - A Gentle Introduction](https://docs.pcalc.org/articles/intro-to-presence-calculus/). While it stands alone, those ideas are easier to grasp if you first see how they manifest in the simpler arrival-departure case, which is what this document covers.

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

What is a sample path for an arrival/departure process?

We begin by specifying how non-determinism enters the model. As we observe a flow process over time, events occur sequentially. At each event we observe:

- The **timestamp** of the event.
- The **type** of event (arrival or departure).

Equivalently, we may think of the inter-event time and the event type as the two random variables governing the process.

Mathematically, this object is a **marked point process**: a sequence of timestamps, each carrying a mark. The minimal mark set here is {arrival, departure}. The timestamps determine the elapsed times between events.

Such a marked point process may be viewed as a single sample path of an arrival/departure process whose broader non-deterministic structure may be unknown.

[@fig:mpp] shows an example of an arrival/departure marked point process. This realized event history is the _input_ to sample path analysis.

![Arrival/Departure Marked Point Process]($document-root/assets/arrival-departure-mpp.png){#fig:mpp}

The key insight from sample path analysis is this:

Any observed history is a finite prefix of an infinite set of possible future continuations. The non-determinism lies in those possible futures. Once a finite prefix has been observed, every deterministic computation derived from it is fully determined by that prefix.
In the case of the coin toss process, this can be visualized as in [@fig:prefix]

![Sample Paths Prefixes]($document-root/assets/sample-path-prefix.png){#fig:prefix}

If we model an operational process as an arrival/departure marked point process, then once a finite segment of the sample path is observed, all flow metrics over that segment are _deterministic functionals_ of the sample path. No distributional assumptions are required to compute or reason about them.

The structure of flow is encoded directly in the realized event history. This gives us simpler and fully deterministic tools for analyzing flow in any process where the event history can be captured accurately. The machinery that makes this possible is what sample path analysis develops.

Thus sample path analysis lives strictly within the domain of operational analysis. Distributions are still important, especially if we are reasoning about those possible futures in prediction models. But that is not the only place where they are useful.

## Where Items and Distributions Matter

Much of operational flow analysis focuses on describing item-level distributions of flow metrics, particularly of lead time and throiughput. These are very useful, helping us assess customer experience, tail risk, etc. However, as we will see, Little’s Law implies that there is no intrinsic non-determinism in these distributions beyond what is already encoded in the sample path above. They are structurally constrained by, and coupled through, the aggregate behavior of the underlying arrival and departure processes. Randomness in item-level flow metrics comes entirely from randomness on the sample paths of the arrival/depature process.

The primary quantities that govern how these distributions behave are the more primitive sample-path flow metrics we will develop.   They are structural properties of the process, not the aggregate properties of item level behavior. These dont require item-level identification of arrivals and departures, nor do they require us to construct distributions to reason about their properties.

To produce those item level distributions,  we must extend the MPP by explicitly identifying arrival and departure events with ids. Then the entire empirical distribution of item-level process times is also completely determined by the realized event structure, for a given sample path _prefix_. In this sense, the structural properties of the arrival and departure events are primary, and all other flow metrics and their distributions are derived.

Making this hierarchy explicit — non-deterministic event structure first, structural flow analysis second, distributional summaries third — is a major distinction between sample path analysis and current methods. It also has significant implications for how these secondary artifacts should be interpreted and used in applications such as forecasting.

This is not a trivial concept and requires more careful explanation. This document and much of the rest of supporting documentation and tools on this site are intended to provide those explanations.







# References
