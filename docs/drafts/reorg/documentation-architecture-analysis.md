# Documentation Architecture Analysis (Sample Path Toolkit)

_Audit date: February 21, 2026_

## Scope and Intent

This document analyzes the current documentation landscape across:

- `/README.md` (repository entry point)
- `/docs/site/index.html` (site landing page)
- Core articles:
  - `/docs/articles/package-overview/index.md`
  - `/docs/articles/theory/index.md`
  - `/docs/articles/cli/index.md`
  - `/docs/articles/chart-reference/index.md`
  - `/docs/articles/not-statistics/index.md`
  - `/docs/articles/a-methodological-contrast/index.md`
- Supporting docs relevant to reader flow:
  - `/examples/README.md`
  - `/examples/polaris/README.md`
  - selected docs drafts and build docs for context

It also reviews the Polaris Flow Dispatch links currently referenced in this repo (including directly linked posts and the series context they indicate), to propose a coherent, progressive learning journey.

The goal is not to reduce technical depth. The goal is to preserve full depth while introducing clear on-ramps, consistent framing, and explicit paths from introductory to advanced material.

---

## Executive Summary

The content quality is strong and technically differentiated, but the documentation system is currently optimized for readers who already share the model. New readers face too many parallel entry points, repeated theory with varying emphasis, and framing drift (especially around CAS language) that can increase cognitive load before the value proposition is internalized.

The strongest opportunity is to introduce a **deliberate progression architecture**:

1. **Orientation**: what this is, when it is useful, and why mainstream flow metrics fail in volatile settings.
2. **Core model**: finite-window pathwise definitions and invariants (minimal but rigorous).
3. **Operation**: how to run the tool and read outputs.
4. **Deep reference**: complete chart-by-chart semantics and derivations.
5. **Case studies and narrative exposition**: Polaris Flow Dispatch and examples.

The current docs already contain almost all needed material. Most of the work is information architecture, role clarity per document, and cross-link hygiene.

---

## 1. How Sample Path Analysis Differs from Statistical and Probabilistic Flow Measurement

This distinction should be the central through-line of the documentation set.

### 1.1 Difference in object of analysis

Mainstream flow tooling typically analyzes **interval aggregates** (calendar buckets) as primary objects.

Sample path analysis analyzes a **realized event-time trajectory** and computes deterministic functionals on that path.

### 1.2 Difference in measurement semantics

Mainstream: reporting windows often define the metric itself (pre-aggregation before measurement).

Sample path: measurement is defined first on event-resolved cumulative processes; optional calendar views are sub-samples of already computed metrics.

### 1.3 Difference in mathematical guarantees

Mainstream: statistical summaries (means/percentiles/trends) whose comparability and inference quality depend on distributional/stationarity conditions.

Sample path: finite-window identities/invariants that hold pathwise at each admissible horizon when quantities are defined consistently on the same observation prefix.

### 1.4 Difference in causal interpretability

Mainstream: generally correlation-oriented dashboard interpretation over aggregates.

Sample path: deterministic dependency structure among metrics measured on the same realized path, enabling constrained cause-effect reasoning.

### 1.5 Difference in operational posture

Mainstream: often assumes stability-like behavior to justify interpretation.

Sample path: treats instability/transience as first-class measurable behavior; convergence and coherence are empirical properties to detect, not assumptions.

### 1.6 Difference in treatment of elapsed time

Mainstream: completion-only elapsed-time averages can exclude active work and create window mismatch with WIP and throughput measurements.

Sample path: residence-time constructions remain aligned to invariant structure over finite windows and only collapse to classical sojourn interpretations under convergence conditions.

### 1.7 Suggested one-line framing for consistency

Use a canonical sentence across docs:

> Sample path analysis is a deterministic, finite-window measurement framework on realized event histories; it is not an interval-aggregated statistical summary workflow.

---

## Current-State Map: What Each Document Is Doing Today

## `/README.md` (303 lines)

**Current role**: mixed role (pitch + conceptual introduction + install/run + mini-reference + links).

**Strengths**:

- Strong problem statement and motivation.
- Clear quick start and command examples.
- Good high-level metric table.

**Issues**:

- Competes with package overview/theory for conceptual ownership.
- Carries significant conceptual load before user intent split (install vs theory reader).
- CAS/VUCA framing mixed with older wording.

## `/docs/site/index.html` (landing page)

**Current role**: curated hub across docs and Polaris links.

**Strengths**:

- Useful external reading curation.
- Immediate visibility into conceptual ecosystem.

**Issues**:

- CAS-heavy language in multiple sections; not aligned with desired VUCA-forward framing.
- No explicit “start here by reader type” routing.
- Includes manual lists that can drift from article architecture.

## `/docs/articles/package-overview/index.md` (135 lines)

**Current role**: historical and conceptual orientation.

**Strengths**:

- Strong historical narrative and significance.
- Bridges to broader Presence Calculus context.

**Issues**:

- Significant overlap with README and theory.
- Heavier than an “overview” for first-time readers.
- Contains CAS-centric framing you now want to de-emphasize.
- Link issue: `./examples/polaris` appears incorrect relative to published article path.

## `/docs/articles/theory/index.md` (189 lines)

**Current role**: compact technical theory spine.

**Strengths**:

- Good concise derivation chain from event counts to invariants.
- Clear finite-window orientation.

**Issues**:

- Some duplicate explanatory burden still appears in chart reference and package overview.
- Link issue: points to `#the-presence-invariant-charts` anchor in chart reference that does not currently exist.

## `/docs/articles/cli/index.md` (277 lines)

**Current role**: command and schema reference.

**Strengths**:

- Clear option contracts and export schemas.
- Good separation of event-indexed vs calendar-indexed mode semantics.

**Issues**:

- Could better surface reader intent (“I just want to run this once” vs full spec).
- Dense for first-time operational users without quick recipes near top.

## `/docs/articles/chart-reference/index.md` (1488 lines)

**Current role**: deep semantic reference and derivation atlas.

**Strengths**:

- Very strong technical depth and explicit terminology.
- Improved internal organization (Sample Path Analysis, Flow Dynamics, Flow Geometry, Reasoning about Flow, Convergence/Stability, Appendices).
- Excellent potential as canonical detailed reference.

**Issues**:

- Entry cognitive load remains very high for new readers.
- Some high-level context is repeated from theory/overview instead of linked out.
- Could use explicit “if this is your first visit, read X first” gates at key chapter transitions.

## `/docs/articles/not-statistics/index.md` + `/docs/articles/a-methodological-contrast/index.md`

**Current role**: methodological positioning.

**Strengths**:

- Contains your most important strategic differentiation.

**Issues**:

- Split across two pages with overlapping argumentation and different polish levels.
- Not clearly integrated as a single canonical “methodological contrast” path.

## `/examples/README.md` and `/examples/polaris/README.md`

**Current role**: case-study entry.

**Strengths**:

- Useful bridge from concepts to concrete outputs.

**Issues**:

- Minimal orientation; assumes prior conceptual context.
- Polaris links partially use `/i/...` forms that are currently less reliable for automated retrieval.

---

## Reader Journey Analysis (Current)

There are currently at least five first-entry paths:

1. PyPI/GitHub user starts at README.
2. Docs site user starts at `/docs/site/index.html`.
3. Theoretical reader starts at package overview or theory.
4. Tool user starts at CLI doc.
5. Existing community reader comes from Polaris posts into chart reference/examples.

These paths do not converge quickly onto a single canonical progression. They repeatedly re-explain similar concepts in slightly different framing, which increases perceived complexity.

---

## Major Coherence and Consistency Gaps

## 1. Framing drift: CAS vs VUCA

You have decided to shift from “complex adaptive systems” as the default rhetorical frame to a broader, less fraught VUCA-oriented context. Current docs still use CAS-heavy language in key top-of-funnel locations (README, package overview, site landing).

## 2. Role ambiguity between README, Package Overview, and Theory

All three explain “what/why/how” of core ideas. None is strictly designated as the canonical conceptual spine.

## 3. Entry load too high for new readers

Chart reference is deep and strong, but there is no compact intermediate bridge from “why this matters” to “how to read these charts without overload.”

## 4. Methodology content split

“Not statistics” and “methodological contrast” both carry the key differentiation but are not clearly sequenced as short + long forms.

## 5. Link integrity and navigational drift

Notable issues:

- Theory points to non-existent chart-reference anchor (`#the-presence-invariant-charts`).
- Package overview example link appears path-broken.
- Mixed Polaris URL styles (`/p/...`, `/i/...`, `open.substack.com/...`) reduce consistency and can create maintenance/retrieval friction.

## 6. Inconsistent conceptual granularity by page

Some reference pages include introductory rhetoric; some introductory pages include advanced derivational claims. This blurs boundaries and makes docs harder to scan.

---

## Polaris Flow Dispatch Alignment Analysis

The repo links to these Polaris Flow Dispatch posts (directly or via site landing):

- The Many Faces of Little’s Law
- A Brief History of Little’s Law
- The Causal Arrow in Little’s Law
- Little’s Law in a Complex Adaptive System
- How long does it take?
- What is Residence Time?
- How Flows Stabilize
- Stabilizing Flow with Timeboxes
- Sample Path Construction for L=λW (referenced in links/sections)
- Flow Processes (referenced in links)
- Polaris case-study link (`/i/.../the-polaris-case-study`)

### Observed narrative ladder in Polaris

The Polaris sequence naturally forms a good progressive ladder:

1. High-level reframing (Many Faces).
2. Legitimacy/history (Brief History).
3. Causal interpretation (Causal Arrow).
4. Full applied walkthrough (LL in CAS; includes sample path construction and finite invariant details).
5. Process-time deepening (How long / Residence time).
6. Stabilization mechanisms (How Flows Stabilize / Timeboxes).

This is already close to the onboarding ladder you want. The toolkit docs should explicitly mirror this progression, not compete with it.

### Recommended boundary between Polaris and Toolkit docs

- **Polaris**: exposition, intuition, contextual narrative, applied stories.
- **Toolkit docs**: definitions, formulas, semantics, contracts, reproducible procedure, chart/file references.
- **Presence Calculus docs**: generalization beyond binary flow process and broader theory.

That boundary should be spelled out in the top-level “Start Here” guidance.

---

## Proposed Target Information Architecture

## A. Canonical doc roles

Assign one non-overlapping primary role per core doc.

1. **README**: product-level orientation + quick start + shortest path to first output.
2. **Package Overview**: context, scope, lineage, where this fits in broader ecosystem.
3. **Theory**: canonical conceptual and mathematical spine (compact).
4. **CLI**: operational and schema contract.
5. **Chart Reference**: complete semantic reference and deep interpretation.
6. **Methodological Contrast**: definitive difference from statistical/interval-aggregate approaches.
7. **Examples**: concrete case-study navigation and replication pointers.

## B. Progressive discovery paths

Provide explicit pathways in README and site home:

1. **Fast path (15-20 min)**
   - Run command
   - Open one stack chart
   - Read “what changed and why” cheat sheet

2. **Practitioner path (60-90 min)**
   - Theory summary
   - Methodological contrast
   - Arrival/departure stacks
   - Convergence section

3. **Deep path (reference)**
   - Full chart reference
   - CLI schema details
   - proofs/appendices

## C. Cross-link rules

- Every deep section links “up” to one conceptual section.
- Every conceptual section links “down” to one concrete chart/CLI section.
- Avoid repeated full explanations when a stable canonical explanation exists elsewhere.

---

## Recommended Structural Changes (Doc Set Level)

## Phase 1: Clarify entry points without deleting detail

1. Add a “Start Here” block at top of README and site home with the three paths above.
2. Add one-sentence role statements at top of each core article:
   - “This page is conceptual spine”
   - “This page is reference”
   - etc.
3. Add “Prerequisites” callouts in chart reference chapters for first-time readers.

## Phase 2: Unify methodology positioning

1. Make one page canonical (likely `a-methodological-contrast`).
2. Reduce `not-statistics` to concise summary + link to canonical long form.
3. Reuse canonical phrasing for event-indexed vs pre-aggregated distinction.

## Phase 3: Align framing language (CAS -> VUCA-forward)

1. Keep CAS as specific case where useful, not as universal lead frame.
2. Adopt VUCA language in top-of-funnel docs.
3. Maintain technical neutrality: claims should be about measurement validity conditions, not domain labels.

## Phase 4: Link hygiene and navigational consistency

1. Fix broken/incorrect anchors and relative links.
2. Normalize Polaris links with a preferred URL style.
3. Add “related reading” blocks with stable ordering (intro -> applied -> advanced).

## Phase 5: Reduce duplication by reference discipline

1. Keep full derivations in theory/chart reference.
2. Keep README concise and directive.
3. Convert repeated explanatory paragraphs in overview pages into short summaries with pointers.

---

## Proposed High-Level Site TOC (Reader-facing)

1. **Start Here**
2. **Why This Is Different** (Methodological Contrast)
3. **Sample Path Theory**
4. **Using the Toolkit (CLI)**
5. **Reasoning with Charts (Chart Reference)**
6. **Examples and Case Studies**
7. **Project Scope and Roadmap**

This TOC preserves all existing depth while making first exposure less intimidating.

---

## Document-by-Document Suggested Repositioning

## README

- Keep: problem, value proposition, quick start, install/run.
- Move out: long conceptual exposition duplicated in package overview/theory.
- Add: explicit link ladders by reader type.

## Package Overview

- Keep: history/origins/significance/scope boundary.
- Tighten: overlap with theory and not-statistics.
- Reframe: VUCA-forward, CAS as specific technical category where needed.

## Theory

- Keep as canonical mathematical spine.
- Add: direct map to chart reference chapter anchors that actually exist.
- Ensure all symbols and interval conventions match chart reference exactly.

## CLI

- Add an early “quick recipes” subsection before full option catalog.
- Keep full contracts untouched.

## Chart Reference

- Preserve depth.
- Add stronger chapter-level onboarding notes:
  - “If new, read Sample Path Theory sections X/Y first.”
- Ensure top matter clearly distinguishes conceptual summary vs reference semantics.

## Methodological docs

- Consolidate to one canonical long form.
- Keep a short technical note as entry teaser.

## Examples

- Expand orientation: what each scenario teaches.
- Add mapping from scenario -> recommended charts -> questions answered.

---

## Voice, Terminology, and Style Consistency Guide (Recommended)

Use these consistently across all docs:

- “Sample path analysis” (not alternating with multiple near-synonyms unless defined).
- “Event-indexed” vs “calendar-indexed” with explicit definition once, reused everywhere.
- “Finite-window invariant” as default phrase for structural identity.
- “Residence time” and “sojourn time” with a stable one-line distinction in every first mention.

Avoid top-of-funnel overloading with:

- Excessive jargon before motivation.
- Unqualified claims that imply mainstream methods are universally invalid; keep claims scoped to measurement semantics/assumptions.

---

## Risks if Left As-Is

1. Strong potential reader drop-off before reaching the practical value.
2. Continued confusion about which doc is canonical for core definitions.
3. High maintenance overhead from duplicated conceptual content.
4. Messaging drift as new sections are added at different times.

---

## Priority Backlog (Suggested)

## Priority 0 (Immediate)

1. Fix broken links/anchors.
2. Add “Start Here” routing to README and site home.
3. Add role statement at top of each core doc.

## Priority 1

1. Methodology page consolidation.
2. VUCA-forward framing pass in top-level docs.
3. Chart reference onboarding callouts.

## Priority 2

1. Examples orientation enhancement.
2. Create short “concept map” page linking theory symbols to chart reference sections.

## Priority 3

1. Editorial pass for style/terminology normalization across docs.
2. Optional docs lint checks for anchor/link consistency.

---

## Concrete Next Step Proposal

If we proceed incrementally with minimal disruption, the first reorganization sprint should produce:

1. A single “Docs Entry Map” section added to README and site home.
2. Theory/chart-reference/CLI role statements and prerequisite links.
3. Link repairs and URL normalization.

That gives immediate coherence gains without touching substantive technical content.

---

## Notes on Source Coverage

Polaris post pages used in this analysis were reviewed from currently retrievable linked content and metadata snapshots for the URLs referenced in this repository. A small subset of linked `/i/...` and `open.substack.com` variants was not directly retrievable in this environment; where needed, role inference for those links is based on link context and overlapping sections visible in accessible companion posts.
