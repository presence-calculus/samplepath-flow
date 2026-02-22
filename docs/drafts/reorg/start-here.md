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

We've been using flow metrics and flow-analysis techniques in the software industry for nearly 20 years. It started with the Poppendiecks and David Anderson introducing the ideas of measuring lead time and throughput and managing work in progress (WIP) as key performance metrics for a software-delivery process in the mid to late 2000s. Those ideas became mainstream as Lean and Kanban concepts were widely adopted in the industry. More recently, as Value Stream Mapping and Flow Engineering became more familiar in the context of continuous improvement, a much richer suite of flow metrics and tools, including those originally popularized by Dan Vacanti [@vacanti2015], also became widely adopted for operational management of software-delivery processes.

So the idea of measuring and managing flow with lead time, throughput, WIP, work item age, and flow efficiency is now well understood as a baseline process-management practice in many parts of the industry.

It may be surprising, then, that I will argue we need substantially different approaches to measure, model, and reason about flow in the software industry, if our goal is systematic process improvement. As AI-augmented engineering upends how we work, better measurement techniques will become even more important, because many existing methods were tied closely to process assumptions that are likely to become invalid as the ground shifts under our feet.

Such claims require proof, both that a real problem exists and that the proposed solutions are worth studying because they actually solve the problem that is claimed to exist. This document is intended to lay out that case at a high level. This project, and the rest of this documentation site, is intended to provide that proof both mathematically and with open-source tools you can use to validate these claims on your own data, if mathematics is not your thing and you prefer to verify things hands-on.

This project is part of a much larger research program called [The Presence Calculus Project](https://docs.pcalc.org), which I have been running for several years as part of my advisory practice, [The Polaris Advisor Program](https://polarisadvisor.com). The current version of the toolkit is a reinterpretation of flow analysis and flow metrics using techniques from the Presence Calculus, and it is a strict generalization of the current flow-analysis models in existing approaches. The [Presence Calculus](https://docs.pcalc.org) itself is considerably more general than the types of analysis we implement here, and can be extended to cover the entire gamut of common measurement problems we commonly encounter in operations management, but we will build up to that systematically by solving the simpler case of flow analysis first.

None of the concepts here are new. Most were worked out over 30 years ago by researchers studying queueing theory. In particular, the core concept of sample path analysis, which is the foundation of the techniques we use here, was discovered by Dr. Shaler Stidham in 1972 when he gave the first deterministic proof of Little's Law. Little's Law is deeply intertwined with flow analysis and flow metrics, although in the software industry we often pay only lip service to it. We will change that here.

The theoretical foundation for this project is laid out in the textbook *Sample Path Analysis of Queueing Systems* by El-Taha and Stidham [@eltaha1999]. Our contribution is to take that dense mathematical material and turn it into practical tools that can solve real-world operations-management problems in the software industry.

Before you can apply these ideas, there are several foundational conceptual leaps to make, especially if your baseline assumptions come from mainstream flow-analysis techniques. This document outlines those conceptual leaps and guides you to detailed reading and tools so you can verify the claims for yourself. There is a meaningful paradigm shift involved here, even though the core ideas are intuitive. This document motivates that shift.




# References
