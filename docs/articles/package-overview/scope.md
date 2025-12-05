---
title: <strong>Package Scope</strong>
---

This package is a part of [The Presence Calculus Project](https://docs.pcalc.org): an
open source computational toolkit that is intended to make sample path methods and
concepts more accessible to practitioners working on operations management problems in
the software industry including engineering/product/sales/marketing operations and
related disciplines: value stream management, developer experience and platforms, and
lean continuous process improvement.

This library and toolkit is intended to be used by practitioners to understand the
theory and _develop their intuition about the dynamics of flow processes_, using their
own environments. Understanding the context behind the data greatly helps makes the
abstract ideas here concrete and there in no substitute for getting your hands dirty and
trying things out directly. This toolkit is designed for that.

It is not ready nor intended to support production quality operations management
tooling.

## In Scope

You will get the most benefits out of this library in the near term by using it to look
closely and carefully, at how flow processes in your environment behave and evolve using
the machinery of this package and learning how to model them.

The toolkit is intentionally designed to be low-tech and force you think about how to
model flow processes in a complex system more broadly before jumping to talking about
metrics and building dashboards.

You should be able to apply it anywhere you can extract a csv file with start and end
dates for various operational processes using the native reporting tools built into the
production tools you use today.

Compare what you see here, with whatever your current operational dashboards show today
and think about the differences. It will soon become apparent why sample path analysis
represents a fundamentally different way of thinking about flow and operations
management.

We will continues to provide more examples and talking about applications in The Polaris
Flow Dispatch so it becomes easier to imagine the possibilities. But the potential
applications are really very large so we welcome contributions from others who have
interesting examples to show once they grasp the ideas here and understand why they
matter.

The barrier to modeling and measuring a flow process using this toolkit are minimal. So
please try it out and see what you find. I would love to publish your use cases. Please
raise a PR to tell us if you have a novel use case and show us how you are using these
techniques.

We plan to extend this library to support the analysis of general flow processes, which
allow arbitrary functions as marks, and are governed by the H=λG form of Little's Law.
This will allow us to directly model the economic impacts of flow processes.

## Out of scope

We do _not_ plan to directly implement online analysis of real time flow processes in
this package nor build web apps or UIs here in this package. It will remain an offline,
command line toolkit for the foreseeable future.

However, the underlying modules implement the low-level machinery and code required to
build these more production ready and richer applications, and our licensing permits you
to build those applications using the concepts in this package if you so desire.
