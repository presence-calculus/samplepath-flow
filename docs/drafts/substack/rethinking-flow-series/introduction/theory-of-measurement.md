---
title: "<strong>Sample Path Analysis</strong>"
subtitle: "Towards a theory of measurement for software development"
author: |
  <a href=""><em>Dr. Krishna Kumar</em></a>

document-root: "../.."
header-image: "-root/assets/"
---

# Series Introduction and Article Setup

The broad thesis: if we think of how theories develop and mature in a discipline, there are
multiple stages - first we get anecdotal evidence that that something works, then we establish that
evidence empirically, then we develop informal explanatory theories for why those results hold, and perhaps
test those theories empirically with more precise data and measurements, and finally we develop
mathematical theories with clear definitions and provable claims the explain logically why those theories must be true.

The final stage is what allows us to reason about the limits of the methods we develop and develop new methods, and often
posit the existence of things or phenomena that may have never been observed empirically, but follow as logical consequence of
the mathematics. In fact, then empirical validation of those phenomena is used to
strengthen the validity of the mathematical theory as an accurate representation
of reality.

Much of our scientific knowledge of the physical world has evolved with this interplay between field observation, empirical
theories, scientific postulates and mathematical explanations.

The argument is that in the software industry we mostly stuck in the anecdotal observation stage. We lack even clearly articulated informal
theories that can be rigorously tested empirically. Even in areas like flow and devops, where we have theories, they are imported from other domains
and dont apply cleanly in our domain which makes it harder to build rigorous practices around understanding the limits of methods, and when
we can apply them. This is still left to the opinions of thought leaders. We are missing the most valuable upper two loops of science;
coherent explanatory theories, and the mathematics that show why those theories work.

This is the gap we are seeking to answer for specific operations management problems with our techniques. We want to motivate that with some concrete examples.


# Flow, Kanban, and Making software delivery predictable.

The concept of flow in software development were developed in the Lean software development community in the early 2000s, and in particular, by the developers and practitioners of the Kanban method. It arose indirectly, in the context of finding practical methods to solve a ubiquitous problem in software delivery: managers have relatively little visibility into how much work a development team can deliver, and so they routinely overpromise and teams underdeliver.

The solution was to make the _impact_ of this strategy, the buildup of unfinished work sitting idle in the delivery pipeline, _visible_ to managers and teams.
The Kanban board, with its representation of discrete work-items moving through a linear progression of workflow stages starting with the incoming work and ending in the delivered work, was a simple but vital element of this solution.

Once we visualize this flow of work-items on the board, two things become easier to reason about.

1. The imbalance between the number of work-items that have been committed and the work-items that have completed.
2. The imbalance between the number of work-items that have started but not completed.

The first allows us to visually estimate the degree of overcommitment for a team as a whole, the second allows us to visualize the impact: the workflow stages where work-items build up over time, giving clues to where the bottlenecks and system constraints live.

Improving the flow of work, then is a matter of devising policies that guide how many work-items can enter the process either overall, or at particular stages. Then
we can work backwards from those constraints to decide how much work can be committed to. We use tools like WIP limits to implement these policies. Once work has started, we want to ensure it keeps moving through the process without unnecessary delays.

While stage/column level buildups reveal where work might be getting stuck, additional tools like flagging blocked work, tracking the overall age of work-items in the process, and the time they have spent in various stages gives us metrics that allow us to narrow in on the specific work-items that need attention.

Improving the flow of work through the system then turns into a systematic program of managing the movement of work-items through the stages/columns on the Kanban board. The benefits of this are clear: managers and teams both have tools to reason about the delivery process explicitly, and use data to understand how to commit to work and manage the delivery processes to meet those commitments consistently. The result is a delivery process that stakeholders perceive as being more predictable - promises made are promises kept, more often than not. It is a system that has been empirically proven to work over and over again, in the 20+ years since it was originally developed.

But why does it work? Is it possible to take the empirical evidence and turn it into a rigorous explanatory theory?
We have theories of flow derived from Lean, but they are not a direct fit for reasons we can explain in detail.

But more importantly we dont have a mathematical theory that explain how these imbalances work. This is what we will provide with sample path analysis..
etc.

# Continuous Deployment - improving stability by deploying more frequently

Curiously, there are parallels between the circumstances that motivated the Kanban method
and those that led to the small batch delivery practices that are core part of modern DevOps.

Well into the early 2010s (and indeed in many large enterprises, even today), the big batch release
of software was the norm. Changes to production were delivered in monthly or quarterly releases and releases
were generally considered a high-risk, all-hands-on-deck affair. While practices like Scrum and XP had brought
continuous integration into the mainstream on Agile teams, releases were still highly orchestrated ceremonies done
at fixed intervals. It was accepted that releasing was a risky affair, and so to manage that risk, managers and teams were
conditioned to do it infrequently under highly controlled circumstances.

Meanwhile several teams were finding success in a seeming paradoxical approach: push code into production as soon as it was integrated
into the main line while ensuring the stability and integrity of the main line is maintained at all times. There was empirical evidence
that far from increasing delivery risk, continuous delivery practices, when done in a systematic, disciplined manner, allowed teams to release
more frequently, while _reducing_ the risk of post release incidents. Rather than increasing delivery risk, continuous delivery let teams deliver software
more frequently while increasing the overall stability of the system in production.

As these practices began to take hold in the industry, the DORA project set out to provide empirical evidence for this claim.
The research team analyzed industry wide survey data and showed the teams that deployed more frequently also reported fewer post-deployment
incidents, and also recovered from incidents faster.

Once again, the interesting question is, why does this happen? Is there a way to build an explanatory theory?

In both scenarios, improving predictability by managing the flow of work-items through the delivery process
and improving system throughput and stability by reducing deployment batch size and deploying more frequently.

But the DORA survey showed only that teams that deployed more frequently had a lower change failure rate.
At the time the original surveys were done, we could assume that teams that deployed every month
deployed changes in larger batches than the teams that were doing continuous deployment to production.
Thus we could assume that for the survey population at the time, deployment frequency was a proxy for batch size,
But this does not mean that increasing deployment frequency reduces batch size. In fact we can have high deployment
frequency while having very large batches - especially if we dont integrate continuously to the main branch.
So if we read the headline and simply mandate that teams increase deployment frequency without doing all the work needed
to reduce batch size first, then we not likely to see the combination of higher deployement frequency and high stability.
This means that the headline about the relationship between the deployment frequency and stability that the survey data
shows has many hidden variables that also need to measured and modeled before we can get an explanatory theory.
This is still steps away from a theory why lower batch size leads to fewer incidents, and even further away from a mathematical theoru
that lets us measure and see this explanation in action against real world data.
