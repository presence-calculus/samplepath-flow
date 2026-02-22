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

We’ve been using flow metrics and flow-analysis techniques in the software industry for nearly 20 years. It began when the Poppendiecks [@poppendieck2003; @poppendieck2006] introduced lead time, throughput, and work-in-progress (WIP) as core concepts of Lean software development in the mid-2000s, adapting these ideas from concepts used in manufacturing processes. Those ideas became mainstream as David Anderson popularized the Kanban method [@anderson2010] as a general empirical management approach for knowledge work. More recently, as Value Stream Management practices matured and flow-based improvement became embedded in DevOps and product organizations, a richer suite of flow metrics and analytical tools — including those popularized by Dan Vacanti [@vacanti2015] — became widely adopted for operational management and forecasting of software delivery processes.


So, the idea of measuring and managing flow with flow metrics is now widely accepted as a baseline process-management practice across much of the software industry. It may be surprising then, that I argue we need substantially different approaches to measure, model, and reason about flow if our goal is systematic process improvement in the digital knowledge work domain. As AI-augmented engineering reshapes how work is performed, measurement techniques tied to implicit process assumptions are increasingly likely to break down. If the structure of work changes, the mathematics used to reason about it must be robust to that change.

Such extraordinary claims require proof: first, that a real limitation exists in prevailing methods; second, that the proposed alternatives genuinely resolve that limitation. This document lays out that case at a high level. The broader project and documentation provide both the mathematical foundation and open-source tooling needed to validate the claims directly on your own data.

None of the core mathematical ideas are new. Many were established decades ago in queueing theory. In particular, the sample path analysis framework underlying this work traces to Shaler Stidham’s 1972 deterministic proof of [Little’s Law](https://docs.pcalc.org/articles/littles-law/). Little’s Law is foundational to flow analysis, though in practice it is often invoked loosely in the software delivery domain, without exploiting all its structural implications. This is a key point of departure for our methods.

The theoretical foundation for our methods is presented in *Sample Path Analysis of Queueing Systems* by El-Taha and Stidham [@eltaha1999]. Our contribution is practical: translating dense mathematical theory into operational tools that can be applied directly to real software delivery systems, and simplifying some of the arcane terminology so that it is easier to relate the domain that is being modeled, without losing rigor.

Applying these ideas requires several conceptual shifts, especially if one’s starting point is mainstream flow analytics. This document introduces those shifts and points to the detailed theory and tooling needed to verify each claim. While the underlying mathematics is elementary, the perspective change is significant and may be disorienting, especially if you are very comfortable with current techniques. This document motivates that change.

## The Presence Calculus Project

This work is part of the larger research program known as [The Presence Calculus Project](https://github.io/presence-calculus), developed over several years within my advisory practice, [The Polaris Advisor Program](https://polarisadvisor.com). The current toolkit reinterprets flow analysis using techniques from the Presence Calculus and strictly generalizes conventional flow-metric models. The Presence Calculus itself is more general still, extending beyond flow analysis to a wide class of operational measurement problems.

We begin here with the simpler and well-understood case of arrival–departure flow processes which all current flow models build upon, simply to expose key new general concepts in a familiar setting. All these concepts generalize beyond the simple arrival-departure process, while keeping the modeling and measurement techniques analytically tractable. This is what makes these ideas really powerful, well beyond as better way to measure flow metrics.

But that is beyond the scope of this document. That generalization is the subject of [The Presence Calculus - A Gentle Introduction](https://docs.pcalc.org/articles/intro-to-presence-calculus/). While it stands alone, the ideas there will be easier to grasp if you first understand how they manifest in the simpler case of arrival-departure processes, which is what this document does.







# References
