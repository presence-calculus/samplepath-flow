---
title: "<strong>Sample Path Analysis</strong>"
subtitle: "The Physics of Flow"
author: |
  <a href=""><em></em></a>

document-root: "../.."
header-image: "-root/assets/"
---

The 2025 Edition of The Polaris Flow Dispatch was devoted to showing the significance of Little's Law in
reasoning about operational processes in software development.

Little's Law is familiar to most practitioners of Lean software development, but it plays
only a peripheral role in how we measure or reason about operational processes in our field.
The conventional explanation is that law is not directly applicable in our operating environments, because it requires
stable processes, and the conditions under which Little's Law applies are not often satisfied.

Dan Vacanti [@vacanti2015], gave one of the clearest set of guidelines on how to think about the law in our domain.
- Think of the law as an elegant mathematical mathematical equation that expresses a relationship between the flow metrics we measure.
- Recognize that there are conditions under which this law is satisfied.  
- Treat the violations of the conditions as operational signals, use them to shape policies that govern the flow of work.
- Even when the conditions are not satisfied, use Little's Law for rough estimation.

All of this is directionally correct, but we rarely see Little's Law rarely being used this limited way in practice. One reason is that despite being an intuitive relationship, Little's Law is quite a subtle result and it has a long history of being deeply misunderstood even by the experts.

As Shaler Stidham  [@stidham72] noted when he gave a deterministic proof of the law in 1972, ".. there are many who feel that the $L=\LambdaW$" is such an 'obvious' relation that a rigorous proof is superfluous, at most a heuristic argument is required. ..(such) arguments can be deceptive. In any case, they are not instructive when the ascribe the validity of the result to properties that turn out, upon rigorous examination, to be superfluous, while failing to document the properties that turn out to be essential."

Little's Law has traditionally been described as a queueing formula. As Dr. Little described in [@little2011],  "A queuing system consists of discrete
objects we shall call items, which arrive at some rate to the
system...The stream of
arrivals enters the system, joins one or more queues and
eventually receives service, and exits in a stream of depar-
tures. ". Even in Stidham's 1972 proof, the problem domain is (somewhat awkwardly) defined as one of "customers arriving at an input-output system, at a sequence of time instants."

One of the profound insights from Stidham's approach to proving Little's Law was that much of this description of the "system" to which law applies is superfluous.
Stidham’s original proof showed that the familiar probabilistic assumptions and much of the internal queueing structure were unnecessary to establish Little’s Law.

Subsequent work by Stidham, El-Taha, and others developed this insight into the more general technique we call sample path analysis.  They showed the only requirements to Little's Law is the sequence of timestamps, each mark indicating whether it is one of two types of events (which we may call arrival and departure). Sample path analysis shows how we can derive all the quantities in Little's Law from just this  minimal representation. Items and queues, as a structural nexus for the events were unnecessary to state and prove Little's Law.

Stidham's proof also showed that the relationships in Little's Law were deterministic in nature and did not require assumptions about probability distributions of item arrivals  or service times in queues, as was long believed. In fact, contrary to what we assume today,  the "averages" in  Little's Law are not even averages of empirical distributions. What sample path analysis shows is that the quantities in Little's Law are limiting values of functions computed deterministically from the sequence of timestamps. When those limits exist, Little's Law holds.

While this is a mathematically elegant result, the way we measure and use flow metrics today is much simpler and intuitive. Like in Dr. Little's model, we assume we have a set of discrete work-items that arrive and leave some operational boundary. We compute flow metrics from item metrics: The number of items that arrive or depart at the boundary over some observation period, the average number of items that remain in the boundary over the period, the average time it takes items to transit the boundary from start to finish. Checking whether Little's Law holds is as simple as plugging in the numbers and seeing if the identity holds.

Except, its not as simple as that. Once you commit to this definition of flow metrics as statistical averages, you have to bring in much of Dr. Little's original machinery of queues and probabilistic assumptions such as stationary service processes, and ergodic arrival processes to apply Little's Law. The deterministic conditions for Little's Law which were introduced in our domain in [@Vacanti2015], and which to the best of my understanding can be traced to a claim in Little and Graves [@graves2008], are unproven for the case when WIP never goes to zero. This is likely one reason they have been hard to operationalize reliably.

While the sample path metrics are not as intuitive, in our series last year, we showed how to calculate flow metrics as functionals over a sample path of a marked point process. We showed how we could check for process stability by testing for local convergence of these functions over finite windows. We showed how metrics let us use invariants similar to Little's Law to reason about flow as a process level behavior even when it was unstable. We showed that conditions required to satisfy Little's Law are also the ones where the sample path metrics become equivalent to item level distributional averages.

Those posts demonstrated several proofs of concept for how we might use these metrics.
 Since last years posts, I have released and field tested an open source toolkit that formalizes the sample path method of measuring flow using the minimal event based measurement model. There is extensive technical documentation at [] for those who want to dig deeper into the theory and mathematics.

We will use this years edition of the Polaris Flow Dispatch to demystify the mathematics and show how we can apply and build upon these ideas.

Quite aside from the theoretical arguments, there is a very practical reason why these techniques are important. Separating the concept of flow from items and queuing systems makes it quite a bit simpler to model the flow of work in real world software development with making strong commitments about the structures through which that work flows.

In our domain, items are a modeling choice. They are placeholders for the actual work. While queueing is endemic in our system of work is endemic, the queues themselves are ephemeral. Any representation of a software development workflow as items arriving at statically defined set of queues, ends up being an inaccurate representation of the underlying queueing behavior in the process. When we represent the flow of work in a Kanban board, we are choosing to model it as a particular static projection of a dynamically changing queueing network. No single such projection will fully capture all the queueing behavior in the system.
Better then, if we can reason about flow without having to commit to the items of the internal structure of queues within a boundary. In fact, even the concept of static boundary can be dispensed with for the most part under the model that sample path analysis gives us.

By helping us reason about flow as a process level behavior, without having to first commit to the structures that generate that behavior, sample path analysis gives us new ways to model and reason about the flow of work in software development. This will be a topic we will explore in greater detail this year.
