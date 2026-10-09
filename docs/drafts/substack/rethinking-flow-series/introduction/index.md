---
title: "<strong>Work items and other fictions</strong>"
subtitle: "Towards a processual theory of measurement for software development"
author: |
  <a href=""><em></em></a>

document-root: "../.."
header-image: "-root/assets/"
---

The Polaris Flow Dispatch turned four this month. As we begin the fifth edition, I want to use this opening post to
frame the questions that have animated this publication over its first four years, and to set the stage for a more ambitious pivot
this year. The fundamental question we have continued to ask directly and indirectly over the first for editions, is this: what does it mean to measure software development.

For most of its relatively short history, the software industry has defaulted to
methods used to measure service businesses or a factories to manage software development. They are appealing for a
very simple reason. Managers face similar business pressures: reliably meeting release goals, and promising due dates for deliveries
while maintaining quality, ensuring capital investments yield returns on investment, and increasing labor productivity.
To the extent that we have invested any serious effort in measuring software development, it has been to help executives answer these questions.

We have rarely paused to develop a defensible theory of measurement that takes into account the  particulars of software development as a management domain. Rather, the tendency has been to borrow and superficially adapt a ready made answer from some other domain. Often not fit for purpose,
such solutions provide plausible answers with relatively little effort, and for the most part, this has been the deciding criterion.

Want to measure developer productivity? Measure the number of pull requests merged per developer.
Want to measure the performance of  cross-functional development teams, measure throughput and lead times for business facing deliverables: stories, epics, features etc.
Want to plan and deliver complex work with dependencies? Use the techniques of project management.


The early days of Agile were a bottom-up attempt to create alternatives to the then dominant project management paradigms, by on the particulars of software development and what made it different from traditional project domains. The early proponents pioneered techniques and a distinct vocabulary that put software development front and center as a specific management domain. But the movement's resistance to conventional measurement left many of the underlying management questions unresolved.

The pressures facing managers did not change, and there was no shortage of management metrics that arose to fill the gap. Metrics like story point and velocity that continue to exert a strong gravitational pull for managers, even as their authors object to the uses to which they were being put twenty years later.


Lean software development and Kanban gave us a more considered model for measurement, but it was borrowed and adapted  from industrial production. The playbook was intuitive and superficially easy to adapt to software. Choose a unit of work - the work item. Model its flow through a development process, usually represented as a linear sequence of stages of the SDLC. Define throughput as the  number of units produced in a standard business facing reporting period. Process time measures like lead/cycle times were reported as empirical distributions over completed units. Since the units of work often had durations that span reporting periods, work item age gave us metrics that let us measure the impact of internal delays. The theory of constraints gave us a language for reasoning about how and where to improve the system. The primary objective of measurement was to improve the flow of work-items through the development process.

These were concepts that had been used fruitfully for decades in managing factories. These were effective tools for measuring and reasoning about work at the team level. The underlying concepts and methods, grounded in queueing theory and operations research were built on strong theoretical foundations compared to its predecessors. Indeed this was how I approached measurement in software development for years in my practice, and my early writing in The Polaris Flow Dispatch reflects that.

But there were acknowledged limitations. The techniques were increasingly hard to scale beyond single teams. Typically they required elaborate process changes and expensive tooling support simply to get data to answer the most basic operational questions once you started measuring work at portfolio scale. "Data hygiene" was an ongoing concern.

Beyond scalability, there were fundamental conceptual issues. The work item was not an accurate representation of the work. Unlike the idealize flow the work rarely flowed linearly or uniformly through the SDLC. But rather than modeling and measuring the real flow of work, flow improvement focused on changing processes to make work items flow linearly.

But by far, the biggest disconnect was in the economics. The industrial production model works well in those situations where operational improvements lead directly to measurable economic outcomes. In software development by contrast, there is no general theory that lets us measure the economic value created by improving the flow of work items.

Resolving these disconnects have been my main pre-occupations in The Polaris Flow Dispatch over the last four years.  We have examined these questions from a number of different angles seeking to understand where the current methods break down and why. The key question is whether the limitations can be overcome or whether we need to start from a fundamentally different foundation than the ones bequeathed us from industrial production.

As we enter the fifth year of this publication this week, I feel more confident that the answer is within reach.  It builds upon many of the ideas we have explored here over the past four years: adapting classical operations research techniques for non-stationary processes for operations management in software, value network models for value creation, the physics of attention as the driving force of flow in software development, understanding when and how to use classical queueing theory and probabilistic models in reasoning about software development.

We introduced an analytical technique called Sample Path Analysis in our series on Little's Law last year. It builds on a mathematical theory of processes and has more than five decades of history in operations research. The flow metrics we use today can be constructed as derived measures built from more primitive concepts from mathematical process theory. But in measuring software development processes,  we need to drop down to that lower level of detail in order to accurately model the conditions which obtain in real world software development.

A general theory of measurement for software development must account for non-stationary and path-dependent processes, for processes that change through learning and adaptation, and for measurements that depend on our choice of events and observation windows. These conditions occur often enough in real software development that we cannot simply assume them away.

In fact, we will argue that to represent the dynamic nature of software development processes, we have to move beyond simple models of predefined units of work moving linearly through predefined workflows. Mathematical process theory gives us the formal tools to represent and measure processes under more general conditions. Work items cease to be fundamental primitives once we understand their place in an ontology grounded in that theory. We will argue that they are largely convenient fictions designed to simplify measurement, neither necessary nor sufficient as a general foundation for modeling or measuring real-world software development.

These arguments will take a little time to develop carefully, and I expect to do so over next five or six posts in this edition of the Dispatch. But the focus here will be mostly in drawing contrasts, both technical and philosophical, between the status quo and the new directions. Much of the detailed technical work will be documented and implemented through an open source research program called The Presence Calculus Project. Most of the measurement machinery measuring the flow of work, is already implemented in the sample-path analysis toolkit. The broader theory, including connecting the flow of work to value creation, still needs to be fleshed out in more detail. The research project will provide the technical material and tools for readers who want to engine with and examine the arguments more deeply. The Dispatch will continue as the channel for higher-level, often more informal discussions that link back to that work.

The Presence Calculus Project is a long-term initiative. I expect many parts to change as the work develops, and I have no illusions about how quickly these ideas might gain traction, if they do at all. But our thesis remains. If we are to make fundamental progress in how we manage software development, we need better models of the value-creation processes that underpin the work, and measurement systems that let us reason about both work and value. Building them requires us to reconsider some familiar assumptions about what we are measuring and how.
