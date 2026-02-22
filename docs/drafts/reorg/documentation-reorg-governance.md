# Documentation Reorganization Governance Policy

_Last updated: February 21, 2026_

## Purpose

This policy defines how the holistic documentation reorganization is executed so decisions stay consistent across commits, reviews, and merges.

## 1) Branching and Merge Policy

- Active implementation branch: `codex/docs-holistic-reorg`
- Integration target: `main`
- Merge style to `main`: **squash merge only** for this reorg effort
- Safety policy: before any history rewrite on `main`, create a backup branch pointer

Rationale: this keeps the docs-reorg history readable and avoids replaying large fragmented doc-only commit chains on `main`.

## 2) Commit Cadence Policy

- Default cadence: **one commit per coherent workstream slice**
- Acceptable commit granularity:
  - one focused workstream checkpoint, or
  - one tightly-related doc set update (for example README + site home entry-map alignment)
- Avoid “mega commits” spanning unrelated workstreams

Commit message convention:

- `docs(reorg): <what changed>`
- `docs(voice): <what changed>`
- `docs(links): <what changed>`

## 3) Scope Policy (This Reorg Pass)

In-scope sources:

- `/README.md`
- `/docs/site/index.html`
- `/docs/articles/package-overview/index.md`
- `/docs/articles/theory/index.md`
- `/docs/articles/cli/index.md`
- `/docs/articles/chart-reference/index.md`
- `/docs/articles/not-statistics/index.md`
- `/docs/articles/a-methodological-contrast/index.md`
- `/examples/README.md`
- `/examples/polaris/README.md`

Generated outputs in scope when corresponding sources change:

- matching files under `/docs/site/**`

Draft policy:

- `/docs/drafts/*` is treated as **internal working design space** for this effort.
- Drafts may be rendered under `/docs/site/drafts/*` by build tooling, but they are not publication targets.

Publishing-state policy:

- **State A: Internal Draft (not deployed)**  
  Source under `/docs/drafts/*`, rendered under `/docs/site/drafts/*`, excluded from Pages deploy.
- **State B: Deployed-Unlinked (soft launch)**  
  Source under a reorg path in `/docs/articles/reorg/*`, deployed to production but intentionally not linked from nav/home/README.
- **State C: Live-Linked (public path)**  
  Promoted canonical path under `/docs/articles/*` and linked from entry points.

Path policy for this reorg:

- New reorg documents are authored in `/docs/drafts/reorg/*` first.
- When stable enough for production QA, promote to `/docs/articles/reorg/*` (State B).
- When approved for end-user discovery, wire links from entry points and keep or rename final path as needed (State C).

## 4) Build and Validation Gates

For docs-content commits:

1. Run docs build after markdown changes that affect published pages:
   - `./docs/build/pandocs.sh`
2. Stage corresponding generated outputs under `/docs/site/*`
3. Run pre-commit before commit:
   - `PRE_COMMIT_HOME=.pre-commit-cache pre-commit run --all-files`

Tests policy:

- Run pytest only if non-doc code is changed.
- If changes are docs-only, pytest is not required.

## 5) Review Policy

Review order for each checkpoint:

1. Structural coherence (reader flow and role clarity)
2. Technical accuracy and consistency of terminology
3. Voice consistency against `/docs/drafts/author-voice-profile.md`
4. Copy cleanup (typos/grammar)

Escalation policy:

- Raise only major issues in review summaries.
- Auto-fix typos/grammar directly without blocking discussion.

## 6) Voice and Style Enforcement

Author voice source of truth:

- `/docs/drafts/author-voice-profile.md`

Mandatory constraints:

- Minimize formulaic `not X, it is Y` constructions.
- Minimize em-dashes; prefer commas, semicolons, and colons.
- Preserve thesis-led argument structure and contrast-driven reasoning.
- Preserve some rhetorical texture; do not over-normalize into generic docs prose.

## 7) Link and Citation Policy

- Internal links must resolve after pandoc conversion.
- Anchor links must target existing IDs.
- Polaris links should use one normalized URL style across docs (decision tracked in checklist).
- Bibliographic references must render correctly in pandoc output when citations are enabled.

Visibility policy:

- A document is considered “visible” only when linked from at least one entry point:
  - `README.md`
  - `/docs/site/index.html`
  - article top-nav links in `docs/build/pandoc_template.html`
- Deployed-unlinked pages are allowed during transition, but must carry a tracking entry in the checklist.

## 8) Definition of Done (Per Workstream)

A workstream is complete only when all are true:

1. Source docs updated
2. Generated `/docs/site/*` outputs updated as needed
3. Pre-commit passes
4. Checklist statuses updated
5. Decision log updated if a policy choice was made

## 9) Decision Log

- **2026-02-21**: Reorg branch designated as `codex/docs-holistic-reorg`.
- **2026-02-21**: Merge strategy set to squash-only for reorg to `main`.
- **2026-02-21**: Commit cadence set to one coherent workstream slice.
- **2026-02-21**: Draft docs designated internal working artifacts for this pass.
- **2026-02-21**: Voice constraints adopted to minimize LLM markers and em-dashes.
- **2026-02-21**: Three-state publishing model adopted (Internal Draft -> Deployed-Unlinked -> Live-Linked).
- **2026-02-21**: Reorg path convention adopted: `/docs/drafts/reorg/*` then `/docs/articles/reorg/*`.

## 10) Outside-In Rewiring Policy

Rewiring order is outside-in, based on end-user context and decision stage:

1. **External context entry points** (Polaris Flow Dispatch, Polaris Advisor site)
2. **Toolkit top-level entry points** (`README.md`, `/docs/site/index.html`)
3. **Orientation and theory bridge** (overview + methodology + theory)
4. **Operational docs** (CLI, examples)
5. **Deep reference docs** (chart reference and appendices)

Rule:

- Do not rewire deeper sections before the upstream entry context is coherent.
- Every newly linked deep page must have at least one backlink to a higher-level orienting page.

## 11) Cross-Site Pathway Policy

The communication pathway is explicitly multi-site:

- Polaris Flow Dispatch (narrative onboarding),
- Polaris Advisor site (practice context and business framing),
- Sample Path Toolkit docs (formal reference and operational implementation),
- Presence Calculus docs (generalization layer).

Integration requirements:

- Maintain consistent thesis and terminology across all four layers.
- Track cross-site entry pages and intended user intent in the checklist.
- If a source site cannot be reviewed (network or DNS limits), mark pathway decisions as provisional until verified.
