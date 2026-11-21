---
title: "<strong>Towards a Theory of Measurement for Software Development</strong>"
subtitle: ""
author: |
  <a href=""><em></em></a>

document-root: "../.."
header-image: "-root/assets/"
---

Towards a Theory of Measurement for Software Development

In the field of software engineering management, some of the most impactful ideas have also often been counter-intuitive.

For example, when a team is is perceived as not moving fast enough, a common response is to add more work to their queue. The principles
of flow tell us this much more likely to compound the problem. As more work is started, unfinished work sits idle as it competes for attention.
If more work is started as a result, it kicks off a self-reinforcing loop where the perceived delays through the delivery process get worse over time.

To take another example, when deployments are considered high-risk affairs, a common strategy to mitigate the risk is to deploy less frequently. But
teams that develop the engineering practices that let them deploy smaller changes more frequently, see fewer release related incidents, but are also able to recover from them faster.

Neither of these solutions is intuitively obvious, but over the last 20 years the industry has built up a
significant body of knowledge, both empirical and theoretical, to support these claims.






Our series explores the role that measurement systems play here. Most engineering managers today rely on a combination of experience and instinct, popular but unproven theories and metrics to drive decisions. When they conflict, more often than not, instinct and experience wins. Our theories and metrics generally lack the scientific rigor that would convince them to act differently. Knowing that a practice has worked elsewhere doe not tell us whether the mechanisms behind
their success operate in our context. The theories and measurements available to managers do little to resolve that uncertainty.

Let's consider the deployment example.

The DORA research established an empirical association between deployment frequency and change failure rate: specifically, it found one population of survey respondents who reported low change failure rates despite deploying frequently, and another population that reported high change failure rates when deploying infrequently. This by itself does not help a manager in the second group achieve the results of the first one.

The theory is that when you deploy frequently you are deploying smaller changes with each deployment, and thus deployment frequency can be treated as a proxy for batch size. So the survey asked respondents about deployment frequency because it is easier to answer. But in fact, deployment frequency is not a proxy for batch size. Deploying code frequently does not mean that you cannot release large batches. Further, even if we release in small batches, it does not mean that the change failure rate is going to go down. That is a separate question involving the distribution of defects in source code and probability that a change in any part of the source code will produce a visible incident at deployment. The empirically observed relationship between deployment frequency and change failure rate requires
specific mechanisms to exist that _maintain_ this relationship, and it is those mechanisms that need to be identified and measured in order to reliably move populations in the second group to the first.

To be sure, DORA guidance is clear that achieving the results of the first group mean changing engineering practices around continuous integration, automated testing, reducing batch sizes, automating deployment processes, etc. But how can the manager trying to make the transition, reliably measure and track progress along these dimensions and know how and where to intervene when progress stalls. This is the gap that a robust operational measurement system should fill.











While DORA research established an empirical correlation between higher deployment frequencies and lower change failure rates, there is not a direct causal relationship between the two. A team can maintain a high deployment frequency without increasing the change failure rate only if they have adopted the rest of the practices necessary to support continuous delivery: integrating code to the mainline frequently, releasing from the mainline frequently, maintain the stability of the mainline through exhaustive automated tests that gate merging into the mainline chief among them. In my experience, managers rarely ask for or have access to rigorously developed metrics that help them reason about the processes that drive their day to day decisions. Yet, rigorous metrics at that level of detail that can be provably connected to movement in the headline metrics like deployment frequency and change failure rate are precisely what are needed to ensure that the
the headline metrics maintain the desired correlation.



For better or worse, management decision making remains heavily influenced by metrics, but the techniques we use to measure our operational systems and drive those decisions have lagged far behind our ability to create theories of organizational change.



Consider first the problem of delivery commitments.

Managers need to make promises about what a team will deliver and when. Yet the team’s capacity is difficult to establish directly, and authorizing additional work is much easier than increasing its ability to finish. Commitments can therefore grow faster than completions.

The consequences are not always immediately visible. Individual assignments look reasonable. Each initiative has a justification. People remain busy. Meanwhile, unfinished work accumulates, delivery takes longer, and the organization responds to missed commitments by applying more pressure or starting additional initiatives.

Kanban made this accumulation visible. Representing work on a board gave teams and managers a shared view of what had been requested, what had started, and what remained unfinished. The gap between commitments and completions became something they could inspect. The buildup of work in particular stages gave them places to investigate delays and constraints.

That visibility supported a different approach to management. Instead of continually authorizing more starts, teams could regulate how much work entered the process, concentrate on finishing, and attend to aging or blocked work. Policies such as work-in-progress limits gave them practical ways to act.

There is already an explanatory idea here. Starting work does not necessarily increase the rate at which work finishes. When starts exceed completions, unfinished work accumulates. Depending on how the system operates, that accumulation can contribute to further delays.

But several kinds of claim are contained in that explanation, and they need to be separated.

The accounting relationship is exact. Within a defined boundary, when every entry and exit is accounted for, current unfinished work equals initial unfinished work plus entries minus exits. No assumption about a stable process or a probability distribution is needed to establish that relationship.

The consequences of accumulation require more explanation. Whether additional work creates competition for attention, lengthens queues, increases coordination demands, or leaves completion capacity unchanged depends on how the process operates.

The effects of a policy require further investigation. A work-in-progress limit may change how work is selected and finished. Whether it improves delivery, and which limit is appropriate, depends on the conditions under which it is applied.

Mathematics can establish constraints that any such explanation must respect. It cannot, by itself, establish every behavioral consequence of a management policy.

Now consider a second familiar problem: the risk of releasing software.

A release can introduce failures into production. One understandable response is to deploy less frequently and subject each release to more extensive preparation and control. This reduces the number of occasions on which a deployment occurs.

But postponing deployment also changes what accumulates between deployments. More changes can be grouped together, feedback arrives later, and diagnosing a failure may require examining a larger set of interacting modifications.

Continuous delivery and continuous deployment challenged the assumption that speed and stability must be traded against each other. Teams demonstrated that frequent delivery could coexist with reliable operation, and DORA’s research supplied broader empirical evidence for that combination. DORA also identifies small batches and supporting technical capabilities as important parts of the delivery system, rather than treating deployment frequency alone as the intervention. [DORA on continuous delivery](https://dora.dev/capabilities/continuous-delivery/)

The empirical finding gives us something important to explain. But it does not, by itself, tell us what to change in a particular organization.

Increasing deployment frequency is not equivalent to reducing batch size. A deployment count tells us how often deployment events occur. It does not tell us how much change each event contains, how long those changes have accumulated, or how they were integrated and tested.

Even the prescription to reduce batch size needs to be examined carefully. Smaller batches do not, by themselves, establish a lower change failure rate.

The process that accumulates changes into batches is distinct from the processes that create defects, detect them, and determine whether they become production failures. These processes can interact, but an intervention on one does not automatically produce the desired change in the others.

A policy that produces smaller deployments can therefore leave important sources of failure untouched. Conversely, policies that improve defect detection may reduce failures without resolving excessive batch accumulation. To improve delivery flow and reliability together, we need to understand and act on both.

This is where developing the explanation step by step becomes useful.

For batch accumulation, we can construct a marked point process that represents the relevant batch quantity dynamically. Instead of inferring batch size from deployment frequency, we can measure its presence mass directly. The generalized relationship \(H=\Lambda G\) then gives us a mathematical basis for reasoning about that accumulation and policies intended to reduce it.

The construction matters. We have to specify what the events represent, what quantity their marks carry, how that quantity changes, and what each term in the relationship means operationally. The equation becomes useful through those definitions.

Explaining changes in failure outcomes requires additional models. If smaller batches improve feedback, simplify diagnosis, or change the effectiveness of testing, those mechanisms need to be represented and examined. We cannot substitute the phrase “small batches” for that explanatory work.

The same scrutiny therefore applies to management intuitions and practitioner prescriptions. Either can become misleading when the mechanism is left implicit.

These examples illustrate several different kinds of knowledge that a rigorous approach to operational management must connect.

Practical experience tells us that an approach has worked in particular circumstances. It supplies observations, identifies problems, and suggests possible interventions.

Empirical research examines patterns across observations. It can establish associations, estimate effects under appropriate research designs, and give us evidence about how broadly a finding may apply.

Explanatory models propose the processes through which those patterns and outcomes arise. They identify mechanisms and conditions: how accumulation develops, how feedback affects decisions, or how defects become failures.

Mathematical models make definitions and assumptions explicit enough to derive consequences. Some consequences are exact accounting relationships. Others follow only under additional assumptions about the behavior of the process.

These forms of knowledge develop through an ongoing exchange. An observation suggests an explanation. A model reveals a missing variable or an unexpected consequence. That consequence suggests a new measurement or an empirical test. The evidence may then require us to revise the model.

A proof establishes what follows within a mathematical model. Whether that model adequately represents the process we are managing remains an empirical and conceptual question.

Measurement is central to this exchange.

Before we explain an observed relationship, we need to know what our measurements represent. A count of completed work, a distribution of completion times, a deployment frequency, and a change failure rate each describe something specific. None automatically supplies an account of the process that produced it.

The choices behind those measurements matter: which events count, where observation begins and ends, what state existed at the beginning, and how incomplete activity is treated. So do the relationships between quantities. If two metrics summarize different populations or different portions of a history, an apparent relationship between them may not mean what we think it means.

This is the foundation we are seeking to develop through sample path analysis.

We begin with the realized history of a process, represented through events and, where necessary, their marks. Given an explicit observation model and the information needed to reconstruct state, we can calculate how the represented process evolved over time.

From that history, we can derive measurements of state, accumulation, and elapsed time. We can establish exact relationships between these measurements over finite observation windows. This lets us reason about processes while they are changing, rather than first requiring their averages to settle into a steady state.

The familiar flow of work is our first application. It gives us concrete quantities, existing practices, and mathematical relationships through which the approach can be demonstrated. Batch accumulation provides a further application of the marked-process framework.

The larger ambition is to build a measurement foundation on which explanatory models of software-development processes can be developed and examined.

That foundation will not make every operational question deterministic. Statistical inference remains necessary when we make claims beyond the observed history. Forecasting requires assumptions about histories that have not yet occurred. Explanations of management interventions require evidence about the mechanisms through which they act.

What sample path analysis offers is a way to make these transitions explicit: what we measured, what follows exactly from the representation, what requires additional assumptions, and what evidence would support those assumptions.

This matters because the practical objective is to develop methods that work reliably under identifiable conditions.

A theory should help us explain why an intuitive response can backfire. It should help us identify which quantity a policy actually changes, distinguish that quantity from a convenient proxy, and understand what additional interventions are needed to achieve the intended outcome.

It should also help us recognize the limits of a successful practice and develop alternatives when the context changes.

The path from observation to effective policy therefore runs through careful measurement, explicit mechanisms, mathematical reasoning, and empirical validation. Sample path analysis is the foundation we are developing to connect those parts, beginning with the flow of work and extending toward a broader account of the processes through which software is developed, delivered, and made valuable.
