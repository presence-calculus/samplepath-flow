---
ID: 36
Task: New revision of the documentation
Branch: docs-v2
---

Spec: I want to update the chart reference document in `docs/articles/chart-reference/index.md` to reflect the full set of charts we have in `plots/core`  and `plots/convergence`  as well as the top level stacks. The current version is mix of theory, applications and reference and so it is a bit incoherent.It also does not highlight the event indexed and calendar indexed views and their significance clearly. The section on cause effect reasoning here is good but it probably beongs in a separate theory/backgorund doc that we can share across the chart reference and the command line interface doc `docs/cli/index.md`.  That document has a section on the metrics and their definitions and the mapping to Lean/Kanban metrics that also probably belongs in a shared theory doc.

Still, review the current versions of the cli doc and chart reference to understand the starting point. The goal in this task is to refactor all this information into three more coherent documents, and overall theory/background doc on sample  path analysis, a cli reference, and a chart reference. The cli-reference should focus on the cli and the chart reference should focus  on teh charts and each should link to the other and to the common theory/backgorund docs where needed without duplicating content.

For the overall narrative arc of the thoery doc, review the presentation `docs/articles/chart-reference/Sample-Path-Analysis-Presentation (1).pdf`

This is a presentation I gave recently and it has a good story arcc that starts to connect key concepts that should form the backbone of the thoery/overview doc. We can flesh this out as needed.

One this doc is in place, we will build the chart-reference and cli docs to be consistent with it. All docs will be pandoc compatible markdown and follow the existing structures and conventions of the documentation in this project. So the first goal is to produce a markdown version of the presentation that we can iterate on.

The examples in this presentation are drawn from the outputs in `chart-reference/chart_reference_small` There are two scenarios, one with events and one without events. In the chart reference and cli docs we will interpolate these images in context. We'll also need versions for calendar indexed charts but we'll get to that later.

You can use these charts as the examples we pull into all docs.

Before beginning any work, review the code for the plots and the cli and metrics calculations and make sure you understand the current implementation. The docs and code must stay in syync.

------

Before beginning anything review this brief and all supporting documentation and sketch out an initial set of tasks that need to be done to get going. Write them down below so that I can review. Ask any questions you need. We wont begin work until we have a good plan in place.
