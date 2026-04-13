---
title: "<strong>Processes, Elements and Boundaries</strong>"
subtitle: ""
author: |
  <a href=""><em></em></a>

document-root: "../.."
header-image: "-root/assets/"


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

Our models of flow are derived from the language of queueing systems. A queueing system in this context is defined in terms of discrete objects called items, which arrive at some rate to a system. The stream of arrivals enters the system, joins one of more queues, where it can receive service, and then exits the system in stream of departures. This is the model, for example, that Dr. John Little uses to describe queuing systems in his survey article about Little's Law [@Little2011]. This model is very general and by suitably defining items, queues and their topology, we can model many useful operational processes in terms of the flow of items through a queueing system.

The focus of queueing theory is the analysis of _waiting_ in such processes. Queues form when the underlying services process items at a slower rate than the rate at which the items arrive at the service, and queueing theory is the mathematical framework that allows us to rigorously analyze the behavior of such systems under various assumptions of how items arrive, receive service, and eventually depart the services. The theory establishes and rigorouls proves relationships between arrival and departure rates, buildup of items in queues, and delays that are introduced in the system as a result of mismatches between demand (arrival rates) and capacity (service rates). These are typically the questions that most operational managers responsible for business processes need to understand and answer about the business processes they manage, and a clear understanding of the underlying principles of queueing theory gives us reliable tools to reason about such systems.

We now introduce a more general model called the element-boundary-observer model that strictly generalizes the queueing system model described above.
The "items" in this model are processes, which we will call elements. A process is defined as a mapping of a set of states to an index set. In our model, the index set is time, and we will define a process as anything that can be described as an evolution of states, described as sequence of state transitions over time.  

Given a set of elements with associated state transitions, a boundary is defined as a proper subset of states together with the transitions between the states in the boundary and its complement. We will assume that the set of states that define the boundary is countable and that there are only finitely many state transitions between the boundary set and is complement in a finite interval of time.   In particular, we do not require that the complement of the boundary set is countable. We will call process transitions into the states in the boundary arrivals and transitions from states in the boundary to its complement departures.

The third component of the model is time-interval which selects arrivals and departures to/from the boundary. This interval may extend from $ -\infty to +\infty $ in principle, but in operational settings we focus on intervals with a fixed starting point representing when an observer started observing arrivals and departures to/from the boundary set. There may or may not be a finite end point for the observation window.

While this definition may appear more abstract than the queueing system model we began with, the abstraction is not merely academic. By generalizing from items to processes as the fundamental unit of analysis, we shift attention away from the physical objects that appear to _flow through a process_ and toward interactions between the process trajectories of those elements. In many operational settings, items such as cars or customers provide tangible objects that we use to manage flow, but they are not what ultimately determines system behavior and process dynamics within a boundary. What matters is how the underlying processes interact in time particularly under resource constraints.

These interactions are not observed directly at the level of individual trajectories, but through their aggregate effect at the boundary. Flow manifests as the evolution of the state of the boundary over time.  What we call _flow_ manifests directly as observable, measurable properties of the _state of the boundary as viewed by an observer. The element-boundary-observer model makes all parts of this explicit and allows us to model, measure and reason rigorously about flow very precisely.  

This shift has important consequences. It allows us to apply the tools of flow analysis in operational settings where there are no natural or stable notions of “items,” or where such notions are ambiguous, heterogeneous, or constantly changing. This is not merely a theoretical concern. In digital operations management, for example, much effort is spent defining and classifying items before meaningful flow analysis can begin, and significant confusion arises around what assumptions must hold for those definitions to be valid.

In the element-boundary-observer model, these difficulties largely disappear. Instead, we rely only on the existence of observable state transitions across a boundary over time. The processes generating these transitions may be heterogeneous, evolving, and only partially understood, yet we can still reason rigorously about flow. This makes the framework particularly well-suited to processes that operate in complex daptive systems and in the model VUCA environments in which modern digital businesses typically operate. The element-boundary-observer model is the conceptual foundation that let us adapt and generalize flow analysis to such domains. While much of queueing theory can be applied directly, this model also provides a much more natural way to talk about many other types of systems where processes interact over time.
