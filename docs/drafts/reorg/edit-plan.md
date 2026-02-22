# Documentation Reorganization: File-by-File Edit Plan

_Draft for review — February 21, 2026_

## Design Principle

The existing CLI doc (277 lines) and Chart Reference (1,488 lines) contain all the
substantive technical content. The Theory doc (189 lines) is the formal backbone. The
Polaris Flow Dispatch articles provide the richest conceptual exposition. None of these
need to be rewritten.

What is missing is a thin layer of orientation: a way to land new readers, give them the
core intellectual move in a few minutes, and route them to the right depth level. And a
set of small surgical edits so that when they arrive at the denser material, they know
why they are there and what to read first.

The plan has three categories of work:

1. **New pages** (you write, I review): 1–2 short documents, ~1500 words total.
2. **Surgical edits to existing docs** (I draft, you review): callouts, link fixes,
   a recipes block, a consolidation, and a README trim.
3. **Link and terminology normalization** (I do, you spot-check): mechanical cleanup
   across all in-scope files.

---

## Category 1: New Pages

### 1A. Start Here

**File:** `docs/articles/start-here/index.md` (new)

**Role:** The single simplest entry point in the ecosystem. Simpler than the Dispatch
articles. A reader who knows conventional flow metrics should understand why this is
different and why they should care within five minutes.

**Sections (approximate):**

- The diagnostic gap (why reporting isn't enough)
- The measurement problem (aggregation before measurement destroys event-level
  structure; one tight paragraph, not the full argument)
- What sample path analysis does differently (deterministic functionals, finite-window
  invariant, stability as empirical property)
- Reading paths (three intent-driven routes with direct links)
- The broader ecosystem (four layers: Dispatch, Toolkit docs, Presence Calculus,
  Polaris Advisor)
- Current scope (binary flow processes, offline)

**Length target:** 800–1000 words.

**Author:** Krishna. This is the voice-critical page.

**Routing contract:** This page links forward to:

- Chart Reference (deep reference path)
- Theory (practitioner path)
- CLI / Quick Start in README (fast path)
- Methodological Contrast (full argument path)
- Polaris Dispatch articles (second-level intuition building)

Every other doc links back to this page as the canonical "start here."

### 1B. Why This Is Different (short methodological orientation)

**File:** `docs/articles/why-different/index.md` (new)

**Role:** The compressed methodological argument. Not the full Methodological Contrast
article (which stays as the long form). This is ~500–800 words that establishes the
single most important intellectual move: the distinction between aggregation-before-measurement
(statistical summary workflow) and measurement-on-the-event-history (deterministic
sample path analysis).

**Why this exists as a separate page:** So that the Chart Reference, Theory doc, and
Dispatch article links can all point to it as a shared premise. A reader who has seen
this page will arrive at denser material already holding the key contrast. Without it,
every deep document must re-explain this distinction locally, which is the current
duplication problem.

**Sections (approximate):**

- What current tools do (aggregate, then measure)
- What is lost (event-level deterministic structure becomes statistical correlation)
- What sample path analysis does instead (measure on the path, then optionally sample for display)
- Why this matters operationally (cause-effect reasoning, stability as observable, not assumed)
- Where to go next (links to full Methodological Contrast for detail, Dispatch articles
  for intuition, Chart Reference for application)

**Length target:** 500–800 words.

**Author:** Krishna. This is also voice-critical.

**Decision needed:** You may decide to merge 1A and 1B into a single page. The argument
for keeping them separate is that 1A is a routing page (its job is to orient and
dispatch) while 1B is a conceptual page (its job is to plant one idea). If combined,
the page gets longer but the reader makes fewer clicks. Either works; the content is
the same.

---

## Category 2: Surgical Edits to Existing Docs

### 2A. Chart Reference — chapter-level orientation callouts

**File:** `docs/articles/chart-reference/index.md`

**What changes:**

Add a short callout block (3–5 lines) at the top of each major chapter heading:

- **Flow Dynamics** (line 136): callout noting this chapter builds from the event
  stream up to time-average WIP, and that readers new to the framework should read
  Start Here and/or Theory first.
- **Flow Geometry** (line 527): callout noting this chapter introduces the
  arrival/departure factorizations that decompose L(T) into diagnostic coordinates,
  and links back to Theory §Finite-Window Little's Law.
- **Reasoning about Flow** (line 1019): callout noting the stacks combine dynamics
  and geometry on a shared timeline for operational interpretation.
- **Convergence and Stability** (line 1181): callout noting this chapter addresses when
  and whether the process is approaching stable behavior, and that readers should be
  comfortable with Flow Geometry first.
- **Appendices** (line 1363): no callout needed.

Also add a "New here?" note in the opening section (around line 22–48) pointing to
Start Here.

**What does NOT change:** All chart entries, derivations, formulas, and technical prose
stay untouched.

### 2B. CLI Doc — quick recipes block

**File:** `docs/articles/cli/index.md`

**What changes:**

Insert a "Common Recipes" section between the current Scope section (ends ~line 36) and
the Invocation section (starts line 38). Four recipes, each with a one-line command and
a one-line description of what question it answers:

1. "Run a basic completed-items analysis" — `flow analyze events.csv --completed`
2. "Generate a weekly calendar-indexed report" — `flow analyze events.csv --sampling-frequency week --anchor MON --completed`
3. "Trim sojourn-time outliers before analysis" — `flow analyze events.csv --outlier-iqr 1.5 --completed`
4. "Export metrics data without generating charts" — `flow analyze events.csv --export-only --completed`

Add a "New here?" line at the top of Scope pointing to Start Here for conceptual
context.

**What does NOT change:** All option semantics, export schemas, and contracts stay
untouched.

### 2C. Theory Doc — link fix + role statement + orientation pointer

**File:** `docs/articles/theory/index.md`

**What changes:**

1. **Fix broken link** (line 137): Change
   `chart-reference#the-presence-invariant-charts` to
   `chart-reference#sample-path-flow-metrics` (the actual heading anchor).

2. **Add role statement** at the top of the document, after the YAML front matter:
   something like "This article is the formal mathematical backbone for the toolkit.
   For a gentler introduction to why this approach exists, see Start Here. For
   extended intuition and worked examples, see the Polaris Flow Dispatch articles."

3. **Add a forward pointer** from the "How This Maps to the Docs" section to the
   new Start Here page.

**What does NOT change:** All formulas, derivations, and mathematical content.

### 2D. Methodology Consolidation

**Files:**
- `docs/articles/a-methodological-contrast/index.md` (becomes canonical long form)
- `docs/articles/not-statistics/index.md` (becomes short entry point)

**What changes:**

**not-statistics:** Trim to ~15–20 lines. Keep the core assertion (sample path analysis
works with deterministic functionals of the observed history, not statistical summaries).
Add an explicit forward pointer: "For the detailed methodological argument, see
[Sample Path Analysis vs Statistics](../a-methodological-contrast)."

**a-methodological-contrast:** Add a role statement at top: "This is the detailed
methodological argument for why sample path analysis uses a fundamentally different
measurement approach than current flow metrics tools." Add a "New here?" pointer to
Start Here. Light copy cleanup (a few typos/missing periods noted in current text).
Content and argument stay as-is.

**What does NOT change:** The core argument in the methodological contrast piece. It is
already strong and in your voice.

### 2E. README — trim and refocus

**File:** `README.md`

**What changes:**

1. **Add a "Documentation" pointer** near the top (after the badges/image block,
   before "The Problem with Flow Metrics Today"), linking to Start Here as the
   recommended entry point for understanding the approach.

2. **Trim "The Problem with Flow Metrics Today" section:** Keep the opening hook
   (lines 16–35, the strongest prose in the repo). Trim or tighten the
   second paragraph of theory explanation (lines 37–42) since this now lives in
   Start Here and Why This Is Different.

3. **Trim "What This Toolkit Does":** Keep the three properties list
   (distribution-free, finite-window, deterministic). Trim the surrounding
   explanatory paragraphs that duplicate Theory and Start Here. Add a pointer:
   "For the full conceptual foundation, see Start Here."

4. **Keep untouched:** Quick Start, Input Format, Examples, Installation, Development
   Setup, Package Layout. These are the README's core job and they work well.

**What does NOT change:** The README remains the product-level entry point. It still
opens with the problem statement. It still has quick start. It gains a clear pointer to
the docs for readers who want depth.

### 2F. Package Overview — role reassignment

**File:** `docs/articles/package-overview/index.md`

**What changes:**

1. **Add role statement** at top: "This article covers the history, origins, and broader
   context of sample path analysis. For a practical introduction, see Start Here."

2. **Fix broken link** (line 105): `./examples/polaris` needs to be corrected to the
   right relative path from the published article location.

3. **VUCA-forward framing pass:** Replace CAS-first framing in the opening paragraph
   and "Why this is significant" section with VUCA-forward language. Keep CAS as a
   specific technical category where used precisely (e.g., "conditions that
   characterize complex adaptive systems").

4. **Trim overlap** with Theory and Start Here. Where the package overview currently
   re-explains the finite-window formulation or the distribution-free property, replace
   with a one-sentence summary and a pointer to the canonical source.

**What does NOT change:** History and Origins section (unique content). Key Concepts
section structure (but tighten pointers). Flow processes section (unique content).

---

## Category 3: Link and Terminology Normalization

### 3A. Polaris URL normalization

**Decision needed:** Choose one canonical URL format for Polaris links. Current state
mixes `/p/...`, `/i/...`, and `open.substack.com/...` styles.

**Recommendation:** Use `/p/...` for article-level links (most stable, most readable).
Use `/i/...` only for section-level deep links where no `/p/` equivalent exists.
Eliminate `open.substack.com` variants.

**Files affected:** README.md, package-overview, chart-reference, site index.html,
examples READMEs.

### 3B. Cross-link audit

After the new pages exist and edits are applied, verify:

- Every deep section (Chart Reference chapter, CLI section) links "up" to at least one
  orientation page (Start Here, Theory, or Why Different).
- Every orientation page links "down" to at least one concrete reference section.
- No internal markdown link targets a non-existent anchor.
- All relative paths resolve correctly from the published article location.

### 3C. Terminology pass

Normalize across all in-scope docs:

- "sample path analysis" as the consistent term (not alternating with near-synonyms).
- "event-indexed" vs "calendar-indexed" with consistent phrasing.
- "finite-window invariant" or "presence invariant" as the default phrase for the
  structural identity (pick one primary, define the other as alias).
- "residence time" vs "sojourn time" distinction stated once in each doc where both
  appear, then used consistently.

---

## Execution Sequence

### Phase 1: New pages (depends on Krishna drafting)

- Krishna writes Start Here and Why This Is Different (or combined page).
- I review for structural coherence, link targets, and progressive-exposure design.
- Codex reviews for voice consistency and terminology.

### Phase 2: Surgical edits (can begin in parallel with Phase 1)

- 2C (Theory link fix + role statement): no dependency, can start immediately.
- 2D (Methodology consolidation): no dependency on new pages, can start immediately.
- 2B (CLI recipes block): no dependency, can start immediately.
- 2A (Chart Reference callouts): needs Start Here page path finalized for "New here?"
  links.
- 2E (README trim): needs Start Here page path finalized.
- 2F (Package Overview): needs Start Here page path finalized.

### Phase 3: Normalization (after Phase 1 and 2 are stable)

- 3A, 3B, 3C run as a single pass after all content edits are done.

### Phase 4: Build and publish

- Run pandoc build for all changed articles.
- Stage generated `docs/site/*` outputs.
- Pre-commit pass.
- Review rendered output.
- Squash merge to main.

---

## What This Plan Does Not Do

- Does not rewrite the Chart Reference. The internal structure (Flow Dynamics → Flow
  Geometry → Reasoning about Flow → Convergence/Stability) is already the right
  progressive arc.
- Does not create parallel "companion chapters" for existing Chart Reference sections.
- Does not create reader-facing bridge/migration/advisor-entry documents. Those concepts
  from the Codex plan are useful as internal design notes for link strategy but are not
  content the reader needs.
- Does not touch the Polaris Flow Dispatch articles (separate publication channel).
- Does not restructure the CLI doc's option catalog or export schemas.

---

## Open Decisions

1. **One page or two?** Merge Start Here + Why This Is Different, or keep separate?
2. **Canonical URL path for new pages?** Proposed: `docs/articles/start-here/` and
   `docs/articles/why-different/`. Alternatives welcome.
3. **Polaris URL format?** `/p/...` as primary. Confirm or override.
4. **Primary invariant term?** "Presence invariant" vs "finite-window invariant" as the
   default across docs. The chart reference currently uses "Presence Invariant."
   Recommend keeping that as primary.
5. **Package overview fate?** The plan trims and refocuses it. An alternative is to
   eventually retire it and absorb its unique content (history/origins) into Start Here.
   Not proposing that now, but flagging it.

---

## Estimated Scope

| Category | Files touched | New lines written | Lines edited/trimmed |
| --- | --- | --- | --- |
| New pages | 1–2 new files | ~1500 | 0 |
| Surgical edits | 6 existing files | ~80–100 (callouts, recipes, role statements) | ~60–80 (trims, link fixes) |
| Normalization | ~8 files | 0 | ~30–50 (link/term substitutions) |
| **Total** | **8–10 files** | **~1600** | **~100–130** |

The heavy writing is the ~1500 words of new pages. Everything else is small, targeted
edits to existing content.
