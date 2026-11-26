---
title: "<strong>Sample Path Analysis</strong>"
subtitle: "An Introduction"
author: |
  <a href=""><em></em></a>

document-root: "../.."
header-image: ""
---


Sample path analysis is a technique for measuring and analyzing the behavior of a system by observing it continuously over time.
By behavior, we mean that we will observe one or more measurable quantities, which we will call observables of the system and how they evolve over time.

Two aspects of this definition are worth highlighting. First, we observe the behavior of the system without necessarily knowing how that behavior is generated. Second, we are concerned with evolution of the observables over a single continuous observation period, which may be open-ended.  In general, a system may have many such observables, and we will use the term _process_ to refer to the evolution over time of any selected collection of these observables.

When there is uncertainty in the values that any observable can take at any point in time, we call the process _stochastic_. A variable that represents the uncertain value of such an observable is called a random variable. A stochastic process can be represented as an indexed collection of such random variables.

Traditionally, probability theory is the mathematical language we use to quantify uncertainty and the classical theory of stochastic processes is built upon the foundations of probability theory. But the language of probability is not the only option we have. Sample path analysis provides an alternate way to reason about stochastic processes without probabilistic assumptions. In certain problem domains, it gives us techniques to reason _in the presence of uncertainty_ without invoking probability laws. Probability theory remains the overarching theory for reasoning about uncertainty, but when sample path analysis methods are available, they help us distinguish questions we can answer rigorously without resorting to probability theory from those that require probabilistic reasoning. This makes the techniques valuable in analyzing real-world systems, because they let us reason about stochastic processes with fewer and more operationally tractable assumptions.
