---
ID: 10
Task: Metric derivations
Branch: metric-derivations
---

Spec: We want the key charts and stacks to show the derivations of key metrics. For example N(t)= A(T) - D(T),
A(T) = #{arrivals in [0,T]} (suitably formatted in unicode etc). We want to control whether this is shown or not using a command line
argument. Each panel or stack that supports this argument should declare it in the signature, and if so shows the definition next the chart title.
For example N(t) without the derivation will have the title "Sample Path - N(T)". With the derivation will have title  "Sample Path - N(t): A(T) - D(T)"
and so on. I want to keep the definitions of the derivations themselves centralized in a single class in the metrics module. This way that class can also serve as documentatiion of the derivations. All usages of the derivations must pull from the definitions in the central class.

1. Ask questions and then write a behavioral spec for this feature in specs/00010-metric-derivations.md
2. In the CFD, add the derivations to the legends since there are two series in that chart.


Progress:
- Added behavioral spec for metric derivations.
- Implemented derivation titles for core panels and added tests.
- Added tests for title construction, additional renderer derivations, and CLI flag parsing.
- Added CFD legend derivation support and tests.
