---
ID: 11
Task: Panel Classes
Branch: panel-classes
---

Spec:
We want to explore a new API pattern for core plotting:
- Panels become classes with shared state (defaults + options).
- Stacks remain functions (stateless layout orchestration).

We will defer an ADR until after a prototype and decision.

Phase 0 (prototype): refactor N(t) to a panel class and update call sites.
Phase 1: migrate all call sites of render_N and remove render_N.
Phase 2.x: migrate remaining core panels in order of appearance in core.py,
one per commit (L, Lambda, w, H, CFD). Remove each wrapper after migration.

Status:
- N, L, Lambda, w, H panel migrations committed.
- CFD panel migration committed.
- LLW panel migration in progress (core + tests).
