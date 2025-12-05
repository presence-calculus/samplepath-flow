---
title: <strong>Package Overview</strong>
author: |
  <a href=""><em>Sample Path Analysis Toolkit</em></a>

document-root: ../..
header-image: $document-root/assets/sample_path_N.png
---

# Sample Path Analysis

This library implements _sample path analysis,_ a technique for analyzing _macro
dynamics_ of flow processes in complex adaptive systems. It provides deterministic tools
to analyze the _long-run_ behavior of stochastic flow processes:

- Arrival/departure equilibrium
- Process time coherence, and
- Process stability

The relies on the finite-window formulation of
[**Little’s Law**](https://docs.pcalc.org/articles/littles-law).

The focus of the analysis is a single _sample path_ of a flow process: a _continuous_
real-valued function that describes a particular process behavior when observed over a
long, but finite period of time (see figure).

A key aspect of this technique is that it is _distribution-free_. It does not require
well-defined statistical or probability distributions to reason rigorously about a flow
process. Please see
[sample path analysis is not a statistical method](../not-statistics) for more details.

As a result, this technique allows us to extend many results from stochastic process
theory to processes operating in complex adaptive systems, where stable statistical
distributions often don't exist. Our focus is operations management in software
development, but the techniques here are much more general.

## History and Origins

Sample path analysis is not new. The formal theory has been worked out thoroughly by
researchers in stochastic process theory and has been stable for over 30 years. They are
just not familiar in the software industry.

Dr. Shaler Stidham discovered the technique when he provided the first deterministic
proof of Little's Law in 1972. So the core ideas here are nearly 60 years old!

In the years since Dr. Stidham and other researchers in stochastic process theory have
shown the power and versatility of this technique to provide deterministic sample path
oriented results of many classical results in queueing theory that required very
stringent probabilistic assumptions like stationarity and ergodicity to prove
previously.

The canonical reference detailing this work is the textbook
[Sample Path Analysis of Queueing Systems](https://www.researchgate.net/publication/303785171_Sample-Path_Analysis_of_Queueing_Systems)
by Muhammed El-Taha and Shaler Stidham (a downloadable PDF is available at the link).
This package directly implements many of the concepts in this textbook.

See our article on [Little's Law](https://docs.pcalc.org/articles/littles-law) for
comprehensive background and source references.

## Why this is significant

Conditions like non-stationarity and non-ergodic behavior and lack of stable
distributions are _exactly_ the dividing line between complex adaptive systems and
simpler, ordered systems when viewed from an operations management lens.

This is the situation we face in digital operations management. The default assumptions
that underpin operational analysis in ordered domains -- that processes are stable and
that steady state behavior can be observed and analyzed -- breaks down in this
environment.

Sample path analysis allows us to adapt these principles to processes that operate far
from equilibrium and whose behavior is history sensitive, path dependent and sensitive
to initial conditions. All these are hallmarks of a CAS.

There are several capabilities we get from this:

- We can precisely define, measure and reason about properties such as equilibrium,
  coherence and stability, of real-world process using transaction data from operations
  management tools.
- We can retrospectively reason about cause and effect in observed behavior of these
  operational processes, and do so in a deterministic fashion - _even for process that
  are not stable_.

These are precisely the conditions that confound traditional statistical and
probabilistic reasoning about operational processes in the digital domain. Thus sample
path analysis is _the_ technical bridge to rigorously model and measure flow processes
in such contexts.

______________________________________________________________________

# Key Concepts

Please see our continuing series on Little's Law and sample path analysis at
[The Polaris Flow Dispatch](https://www.polaris-flow-dispatch.com) for accessible
overviews of the theory.

In particular,

- [The Many Faces of Little's Law](https://www.polaris-flow-dispatch.com/p/the-many-faces-of-littles-law).
- [Little's Law in a Complex Adaptive System](https://www.polaris-flow-dispatch.com/p/littles-law-in-a-complex-adaptive)

cover most of the background needed to work with this library at a high level.

The example analyses in these posts were produced using this library and can be found in
the [examples](./examples/polaris) directory together with their original source data.

Please subscribe to [The Polaris Flow Dispatch](https://www.polaris-flow-dispatch.com)
if you are interested in staying abreast of developments and applications of these
concepts.

## Flow processes

A [flow process](https://www.polaris-flow-dispatch.com/i/172332418/flow-processes) is
simply a timeline of events from some underlying operational domain, where events have
*effects* that persist beyond the time of the event. The effects are encoded using
metadata (called marks) to describe them. The generality of the model comes from the
fact that marks can be arbitrary real-valued functions of time that meet some very weak
requirements.

Typically data for analyzing a flow process are extracted from real-time transaction
logs of digital operations management tools.

The current version of the library only supports the _offline_ analysis of _binary flow
processes_. These are flow processes where the marks denote the start or end of an
observed presence of a domain element within some system boundary.

All queueing processes fall into this category, as do a much larger class of general
input-output processes. These are the simplest kind of flow processes we analyze in the
presence calculus, but they cover the vast majority of operational use cases we
currently model in software delivery, so we will start there. They are governed by the
L=λW form of Little's Law.

We highly recommend reading
[The Many Faces of Little's Law](https://www.polaris-flow-dispatch.com/p/the-many-faces-of-littles-law)
for background on these concepts.
